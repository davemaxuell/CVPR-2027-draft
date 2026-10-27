"""What-if branches: at one recorded decision, force each alternative action, let the model continue on its own,
and record where the episode ends.

The rollouts run on the GPU in whatif_worker.py subprocesses, one per checkpoint. They exist only while the
viewer's "What-if · GPU" switch is on. The switch is off whenever the server starts (--no-whatif removes it).
Turning it off cancels queued work and stops the workers, freeing the GPU. Workers left idle also exit
after IDLE_SECONDS. Results are appended to exprs_map/whatif/<model>/<split>.jsonl and stay viewable with the
switch off. Standard library only; the worker runs in the scalevln environment (env.sh).
"""
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import parse_trajectories as pt

WORKER = Path(__file__).resolve().parent / 'whatif_worker.py'
STORE = pt.DUET_DATA / 'R2R' / 'exprs_map' / 'whatif'
LOG = Path(__file__).resolve().parent / 'cache' / 'whatif_worker.log'
MAX_ALTERNATIVES = 8     # most likely alternatives per decision; the reference-best option is always added
CHUNK = 32               # branches per worker request (progress is reported per branch)
REPLAY_CHECKS = 2        # unforced replays per job, compared with the recorded episodes
IDLE_SECONDS = 300


def force_key(vp):
    return vp or 'STOP'


class Worker:
    """One whatif_worker.py process (one checkpoint loaded on the GPU)."""

    def __init__(self, model, checkpoint, gpu):
        LOG.parent.mkdir(parents=True, exist_ok=True)
        cmd = f'source env.sh >&2 && exec python {shlex.quote(str(WORKER))} --model {model}'
        if checkpoint:
            cmd += f' --checkpoint {shlex.quote(str(checkpoint))}'
        self.log, self.started = open(LOG, 'ab'), False      # started: the checkpoint has loaded
        self.proc = subprocess.Popen(['bash', '-c', cmd], cwd=pt.ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1, start_new_session=True,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))

    def request(self, req):
        """Send one request and yield its replies, ending with {'done': n}."""
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        for line in self.proc.stdout:
            msg = json.loads(line)
            yield msg
            if 'done' in msg:
                return
        raise RuntimeError(f'the GPU worker exited (code {self.proc.wait()}); see {LOG}')

    def stop(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.log.close()


class WhatIf:
    def __init__(self, gpu=0, allowed=True):
        self.gpu, self.allowed, self.enabled = gpu, allowed, False
        self.cv = threading.Condition()
        self.queue, self.job, self.last, self.seq = deque(), None, None, 0
        self.workers = {}             # (model, checkpoint) -> Worker
        self.results = {}             # run name -> {(instr_id, t, force key): record}
        self.store_lock = threading.Lock()
        self.idle_since = time.time()
        threading.Thread(target=self._loop, daemon=True).start()

    # ---------------------------------------------------------------------------------------- switch

    def status(self):
        with self.cv:
            job = self.job and {k: v for k, v in self.job.items() if k not in ('branches', 'run_obj', 'cancel')}
            state = 'off' if not self.enabled else job['phase'] if job else 'idle'
            return {'allowed': self.allowed, 'enabled': self.enabled, 'gpu': self.gpu, 'state': state, 'job': job,
                    'queued': [j['label'] for j in self.queue], 'last': self.last,
                    'loaded': [m for m, _ in self.workers], 'log': str(LOG)}

    def switch(self, on):
        if on and not self.allowed:
            raise PermissionError('what-if branches are disabled on this server (--no-whatif)')
        with self.cv:
            self.enabled = bool(on)
            if not on:
                for job in self.queue:
                    job.update(phase='cancelled', finished=time.time())
                self.queue.clear()
                if self.job:
                    self.job['cancel'] = True
            self.cv.notify_all()
        if not on:
            self._stop_workers()      # a running request fails at once; its finished branches are kept
        return self.status()

    def _stop_workers(self):
        with self.cv:
            workers, self.workers = list(self.workers.values()), {}
        for w in workers:
            w.stop()

    # --------------------------------------------------------------------------------------- results

    def _file(self, run):
        return STORE / run.model / f'{run.split}.jsonl'

    def stored(self, run):
        """{(instr_id, t, force key): record} for a run, read once from its results file."""
        with self.store_lock:
            if run.name not in self.results:
                recs, f = {}, self._file(run)
                if f.is_file():
                    for line in f.read_text().splitlines():
                        if line.strip():
                            r = json.loads(line)
                            recs[(r['instr_id'], r['t'], r['force'])] = r
                self.results[run.name] = recs
            return self.results[run.name]

    def _save(self, run, rec):
        with self.store_lock:
            f = self._file(run)
            f.parent.mkdir(parents=True, exist_ok=True)
            with open(f, 'a') as out:
                out.write(json.dumps(rec) + '\n')
            self.results.setdefault(run.name, {})[(rec['instr_id'], rec['t'], rec['force'])] = rec

    def summary(self, run):
        """Compact results for the split view: {instr_id: {t: {force: [success, NE, nDTW, is reference-best]}}}."""
        out = {}
        for (iid, t, force), r in self.stored(run).items():
            out.setdefault(iid, {}).setdefault(t, {})[force] = [r['sr'], r['ne'], r['ndtw'], int(r.get('ref', False))]
        return out

    def for_episode(self, run, instr_id):
        """{t: {force key: record without the trajectory}} plus each branch's hop path, for the episode view."""
        out = {}
        for (iid, t, force), r in self.stored(run).items():
            if iid == instr_id:
                rec = {k: v for k, v in r.items() if k != 'trajectory'}
                rec['path'] = [vp for seg in r['trajectory'] for vp in seg]
                out.setdefault(t, {})[force] = rec
        return out

    # ------------------------------------------------------------------------------------------- jobs

    def plan(self, run, items):
        """Branches still to run for decisions [(instr_id, t), ...]: the MAX_ALTERNATIVES most likely
        options other than the recorded one, plus the reference-best option. Needs the run's step log."""
        if getattr(run, 'ce', False):
            raise ValueError('what-if branches re-run DUET on the discrete graph; continuous (VLN-CE) runs have none')
        if not run.steplog:
            raise ValueError('what-if needs the step log of this run (log_duet_steps.py)')
        stored, branches, cached, skipped = self.stored(run), [], 0, 0
        for iid, t in items:
            if iid not in run.preds or iid not in run.gt or not run.gt[iid].get('path') or len(run.gt[iid]['path']) < 2:
                skipped += 1
                continue
            ep = run.episode(iid)
            if not 0 <= t < len(ep['decisions']) or t >= run.max_action_len - 1:
                skipped += 1           # at the last step the budget forces STOP: there is nothing to change
                continue
            d = ep['decisions'][t]
            md = d.get('model')
            if not md:
                skipped += 1
                continue
            chosen = None if d['action'] == 'stop' else d['target']
            options = [(None, md['fused'][0])] + [(vp, md['fused'][i + 1]) for i, (vp, v) in
                                                   enumerate(zip(md['gmap'], md['visited'])) if not v]
            alts = sorted((o for o in options if o[0] != chosen), key=lambda o: -o[1])
            best = (d.get('ref') or {}).get('best', chosen)
            keep = alts[:MAX_ALTERNATIVES] + [o for o in alts[MAX_ALTERNATIVES:] if o[0] == best]
            prefix = [x['target'] for x in ep['decisions'][:t]]
            for vp, p in keep:
                if (iid, t, force_key(vp)) in stored:
                    cached += 1
                    continue
                branches.append({'id': f'{iid}|{t}|{force_key(vp)}', 'instr_id': iid, 't': t, 'force': force_key(vp),
                                 'forced': prefix + [vp], 'p': round(p, 5), 'ref': vp == best})
        return branches, cached, skipped

    def submit(self, run, items, label):
        if not self.enabled:
            raise PermissionError('turn on the What-if · GPU switch first')
        branches, cached, skipped = self.plan(run, items)
        if not branches:
            return {'job': None, 'branches': 0, 'cached': cached, 'skipped': skipped}
        model = (getattr(run, 'settings', {}) or {}).get('model') or run.model.replace('released_', '')
        if model not in ('h14', 'b16'):
            raise ValueError(f'cannot tell which ScaleVLN checkpoint produced {run.name}')
        with self.cv:
            self.seq += 1
            job = {'id': self.seq, 'run': run.name, 'label': label, 'done': 0, 'total': len(branches), 'failed': 0,
                   'phase': 'queued', 'cached': cached, 'skipped': skipped, 'check': None,
                   'model': model, 'checkpoint': (getattr(run, 'settings', {}) or {}).get('checkpoint'),
                   'branches': branches, 'run_obj': run, 'cancel': False}
            self.queue.append(job)
            self.cv.notify_all()
        return {'job': job['id'], 'branches': len(branches), 'cached': cached, 'skipped': skipped}

    def cancel(self):
        with self.cv:
            for job in self.queue:
                job.update(phase='cancelled', finished=time.time())
            self.queue.clear()
            if self.job:
                self.job['cancel'] = True
        return self.status()

    def _loop(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.enabled and self.queue, timeout=30)
                job = self.job = self.queue.popleft() if self.enabled and self.queue else None
                idle = job is None and self.workers and time.time() - self.idle_since > IDLE_SECONDS
            if idle:          # only this thread starts jobs, so no job can be using these workers
                self._stop_workers()
            if job is None:
                continue
            try:
                self._run(job)
                job['phase'] = 'cancelled' if job['cancel'] else 'finished'
            except Exception as e:    # worker crash, or the switch was turned off mid-request
                if job['cancel']:
                    job['phase'] = 'cancelled'
                else:
                    job.update(phase='failed', error=f'{type(e).__name__}: {e}')
                self._stop_workers()
            with self.cv:
                job['finished'] = time.time()
                self.last = {k: v for k, v in job.items() if k not in ('branches', 'run_obj', 'cancel')}
                self.job, self.idle_since = None, time.time()

    def _run(self, job):
        run, key = job['run_obj'], (job['model'], job['checkpoint'])
        with self.cv:
            worker = self.workers.get(key)
            if worker is None or worker.proc.poll() is not None:
                worker = self.workers[key] = Worker(job['model'], job['checkpoint'], self.gpu)
        settings = getattr(run, 'settings', {}) or {}
        req = {'anno': run.anno_file, 'languages': settings.get('languages') or '',
               'max_instr_len': settings.get('max_instr_len', 200), 'max_action_len': run.max_action_len}
        # replaying a few recorded episodes unforced shows whether batched re-runs reproduce the recording
        checks = list(dict.fromkeys(b['instr_id'] for b in job['branches']))[:REPLAY_CHECKS]
        job['phase'] = 'loading model' if not worker.started else 'running'
        todo = [{'id': f'check|{iid}', 'instr_id': iid, 'forced': []} for iid in checks] + job['branches']
        by_id = {b['id']: b for b in job['branches']}
        job['check'] = {'n': 0, 'match': 0}
        for s in range(0, len(todo), CHUNK):
            if job['cancel'] or not self.enabled:
                job['cancel'] = True
                return
            chunk = [{k: b[k] for k in ('id', 'instr_id', 'forced')} for b in todo[s:s + CHUNK]]
            for msg in worker.request(dict(req, branches=chunk)):
                worker.started = True
                job['phase'] = 'running'
                if 'id' not in msg:
                    continue
                if msg['id'].startswith('check|'):
                    iid = msg['id'][len('check|'):]
                    job['check']['n'] += 1
                    job['check']['match'] += msg.get('trajectory') == run.preds[iid]['trajectory']
                    continue
                b = by_id[msg['id']]
                if 'error' in msg:
                    job['failed'] += 1
                else:
                    self._save(run, self._score(run, b, msg))
                job['done'] += 1

    def _score(self, run, b, msg):
        gt = run.gt[b['instr_id']]
        ep = pt.analyse_episode({'instr_id': b['instr_id'], 'trajectory': msg['trajectory']}, gt,
                                run.graphs[gt['scan']], run.max_action_len, full=False,
                                route_reference=run.route_reference)
        m = ep['metrics']
        return {'instr_id': b['instr_id'], 't': b['t'], 'force': b['force'], 'p': b['p'], 'ref': b['ref'],
                'outcome': ep['outcome'], 'sr': m['sr'], 'ne': m['ne'], 'ndtw': m['ndtw'], 'spl': m['spl'],
                'tl': m['tl'], 'osr': m['osr'], 'decisions': m['decisions'],
                'prefix_agree': msg.get('prefix_agree', True), 'trajectory': msg['trajectory']}
