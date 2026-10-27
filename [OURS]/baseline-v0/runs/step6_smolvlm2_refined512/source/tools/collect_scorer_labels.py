"""Resumable group-B/C four-action collection; four independent native batch-one workers."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
for key in ('MAGNUM_LOG', 'HABITAT_SIM_LOG'):
    os.environ.setdefault(key, 'quiet')
os.environ.setdefault('GLOG_minloglevel', '2')

import habitat
import torch
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from policy.experimental_reference import ExperimentalReference
from scorer_data.collection import carrier, collect_root
from scorer_data.common import DATA, REFERENCE, RUN, job_seed, now, read, sha, verify_protocol, write
from sim.config import load_expert, make_ce_config, scene_group


def worker(args):
    protocol = verify_protocol()
    signature = sha(RUN / 'protocol.json')
    selection = read(RUN / 'selection.json')
    name, mode = [('r2r', 'expert'), ('r2r', 'policy'), ('rxr', 'expert'), ('rxr', 'policy')][args.worker_index]
    policy = ExperimentalReference(REFERENCE)
    assert not any(p.requires_grad for p in policy.model.parameters())
    assert not any(module.training for module in policy.model.modules())
    expert = load_expert(name, 'train')
    stage = RUN / args.stage
    for group in args.groups:
        count = args.roots_b if group == 'B' else args.roots_c
        quota = count // 4
        cfg = make_ce_config(name, 'train', scenes=scene_group(group), gpu=0)
        full = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        lookup = {str(ep.episode_id): ep for ep in full.episodes}
        ordered = selection[group][name][mode]
        ds = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
        ds.episodes = [lookup[ordered[0]]]
        env = habitat.Env(config=cfg, dataset=ds)
        completed, attempted, roots, started = 0, 0, [], time.time()
        try:
            assert list(cfg.habitat.task.actions) == ['stop', 'move_forward', 'turn_left', 'turn_right']
            for eid in ordered:
                if completed == quota:
                    break
                ep = lookup[eid]
                assert ep.scene_id.split('/')[-2] in scene_group(group)
                job = DATA / group / name / mode / eid
                seed = job_seed(group, name, mode, eid)
                recorded, online = carrier(env, ep, policy, mode, expert[eid], job / 'carrier', seed,
                                           protocol['simulator']['max_episode_steps'][name], signature)
                attempted += 1
                for t in recorded['root_steps']:
                    if completed == quota:
                        break
                    directory = job / 'roots' / f'{t:04d}'
                    row = collect_root(env, ep, recorded, online, policy, expert[eid], directory, t, group, name, mode, signature)
                    roots.append(str((directory / 'labels.json').relative_to(DATA)))
                    completed += 1
                    progress = {'status': 'collecting', 'group': group, 'dataset': name, 'mode': mode,
                                'roots': completed, 'target': quota, 'attempted_episodes': attempted,
                                'last_root': roots[-1], 'seconds': time.time() - started, 'updated_utc': now()}
                    write(stage / f'progress_{group}_{args.worker_index}.json', progress)
                    print(str(progress), flush=True)
                del online
                torch.cuda.empty_cache()
            assert completed == quota, ('Episode pool exhausted', group, name, mode, completed, quota)
            write(stage / f'index_{group}_{args.worker_index}.json', {'status': 'complete', 'group': group,
                  'dataset': name, 'mode': mode, 'roots': roots, 'attempted_episodes': attempted,
                  'seconds': time.time() - started, 'protocol_sha256': signature, 'completed_utc': now()})
        finally:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--groups', nargs='+', choices=['B', 'C'], default=['B'])
    ap.add_argument('--roots-b', type=int, default=64)
    ap.add_argument('--roots-c', type=int, default=0)
    ap.add_argument('--worker-index', type=int)
    args = ap.parse_args()
    assert Path(args.stage).name == args.stage
    for group in args.groups:
        target = args.roots_b if group == 'B' else args.roots_c
        assert target > 0 and target % 4 == 0
    assert len(set(args.groups)) == len(args.groups)
    if args.worker_index is not None:
        assert 0 <= args.worker_index < 4
        return worker(args)
    verify_protocol()
    stage = RUN / args.stage
    stage.mkdir(parents=True, exist_ok=True)
    lock = open(RUN / 'collection.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = {'stage': args.stage, 'groups': args.groups, 'roots_B': args.roots_b, 'roots_C': args.roots_c,
               'workers': 4, 'protocol_sha256': sha(RUN / 'protocol.json')}
    if (stage / 'request.json').exists():
        assert read(stage / 'request.json') == request
    else:
        write(stage / 'request.json', request)
    if (stage / 'status.json').exists() and read(stage / 'status.json')['status'] == 'complete':
        return
    children, logs, started = [], [], time.time()
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Collection interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        write(stage / 'status.json', {'status': 'running', 'pid': os.getpid(), 'started_utc': now(), **request})
        for rank in range(4):
            log = open(stage / f'worker_{rank}.log', 'a')
            logs.append(log)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', args.stage,
                       '--groups', *args.groups, '--roots-b', str(args.roots_b), '--roots-c', str(args.roots_c),
                       '--worker-index', str(rank)]
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        write(stage / 'workers.json', {'pids': [p.pid for p in children]})
        while True:
            codes = [p.poll() for p in children]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f'Worker failure: {codes}; see {stage}/worker_*.log')
            if all(c == 0 for c in codes):
                break
            time.sleep(5)
        roots = []
        for group in args.groups:
            for rank in range(4):
                roots += read(stage / f'index_{group}_{rank}.json')['roots']
        assert len(roots) == len(set(roots)) == sum(args.roots_b if g == 'B' else args.roots_c for g in args.groups)
        write(stage / 'index.json', {'protocol_sha256': sha(RUN / 'protocol.json'), 'roots': roots})
        write(stage / 'status.json', {'status': 'complete', 'completed_utc': now(), 'seconds': time.time() - started,
                                     'roots': len(roots), 'branches': 4 * len(roots), **request})
    except BaseException as exc:
        write(stage / 'status.json', {'status': 'failed', 'updated_utc': now(), 'error': repr(exc), **request})
        raise
    finally:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in children:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
