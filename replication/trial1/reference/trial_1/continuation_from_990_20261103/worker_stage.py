"""Reused owned-child execution and bounded stage recovery; no supervisor lock."""
from pathlib import Path
import os
import subprocess
import time
import json

def live(identity):
    current = process_identity(identity['pid']) if identity else None
    return bool(current and current['start_ticks'] == identity['start_ticks'])

def stage(name, command, ready, progress_path=None, gpu=False, extra_env=None):
    """Resume finished artifacts or adopt a recorded worker, never duplicate it."""
    path = HERE / 'stages' / name / 'state.json'
    state = read_json(path, {})
    child_identity = state.get('process')
    child = None
    if not live(child_identity) and ready():
        state.update(status='complete', verified_utc=utc())
        state.pop('process', None)
        atomic_json(path, state)
        progress(name)
        return
    if not live(child_identity):
        attempts = state.get('attempts', 0)
        if attempts >= 3:
            raise RuntimeError('Bounded recovery exhausted for ' + name)
        log = HERE / 'logs' / (name + '_attempt_' + str(attempts + 1) + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            log = log.with_name(log.stem + '_' + str(time.time_ns()) + log.suffix)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0' if gpu else '', HF_HOME='/data/shared/hf_cache',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
            TOKENIZERS_PARALLELISM='false', MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet',
            GLOG_minloglevel='2', PYTHONDONTWRITEBYTECODE='1')
        if extra_env:
            env.update(extra_env)
        actual = command if gpu else ['ionice','-c','3','nice','-n','10',*command]
        with log.open('x') as stream:
            child = subprocess.Popen(actual, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        # All new children remain in this worker's session for owned-process recovery.
        child_identity = capture_process(child.pid, str(HERE))
        if not child_identity:
            raise RuntimeError('Child exited before identity capture: ' + str(log))
        history = state.setdefault('attempt_history', [])
        history.append(dict(attempt=attempts + 1, started_utc=utc(), process=child_identity, log=str(log)))
        state.update(status='running', attempts=attempts + 1, process=child_identity, command=command,
                     log=str(log), started_utc=state.get('started_utc', utc()))
        atomic_json(path, state)
        print(json.dumps(dict(stage=name, event='started', process=child_identity, log=str(log))), flush=True)
    while live(child_identity):
        progress(name, child_identity, state.get('log'), progress_path)
        if child is not None and child.poll() is not None:
            break
        time.sleep(15)
    code = child.wait() if child is not None else None
    if code not in (0, None) or not ready():
        state.update(status='failed', finished_utc=utc(), returncode=code)
        state.pop('process', None)
        atomic_json(path, state)
        log = Path(state['log'])
        with log.open('rb') as source:
            source.seek(max(0, log.stat().st_size - 12000))
            print(source.read().decode(errors='replace'), flush=True)
        raise RuntimeError('Stage failed without its verified completion: ' + name)
    state.update(status='complete', finished_utc=utc(), returncode=code)
    state.pop('process', None)
    atomic_json(path, state)
    progress(name)
    print(json.dumps(dict(stage=name, event='complete')), flush=True)
