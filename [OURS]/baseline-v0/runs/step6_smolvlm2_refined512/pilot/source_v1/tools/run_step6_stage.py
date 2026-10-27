"""Durable collection followed by complete label audit for one declared stage."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scorer_data.common import RUN, now, read, sha, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--groups', nargs='+', choices=['B', 'C'], default=['B'])
    ap.add_argument('--roots-b', type=int, default=64)
    ap.add_argument('--roots-c', type=int, default=0)
    args = ap.parse_args()
    assert Path(args.stage).name == args.stage
    check = read(RUN / 'check_runtime.json')
    assert check['status'] == 'passed' and check['protocol_sha256'] == sha(RUN / 'protocol.json')
    stage = RUN / args.stage
    stage.mkdir(parents=True, exist_ok=True)
    lock = open(RUN / 'pipeline.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    extra = {p: sha(ROOT / p) for p in ('tools/run_step6_stage.py', 'tools/verify_step6.py')}
    if (stage / 'audit_source_sha256.json').exists():
        assert read(stage / 'audit_source_sha256.json') == extra
    else:
        write(stage / 'audit_source_sha256.json', extra)
        for name in extra:
            dst = stage / 'source' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT / name).read_bytes())
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Stage interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        write(RUN / 'status.json', {'status': 'running', 'stage': args.stage, 'task': 'collection', 'pid': os.getpid(), 'updated_utc': now()})
        tasks = [('collection', ['tools/collect_scorer_labels.py', '--stage', args.stage, '--groups', *args.groups,
                                 '--roots-b', str(args.roots_b), '--roots-c', str(args.roots_c)]),
                 ('verification', ['tools/verify_step6.py', '--stage', args.stage])]
        for task, command in tasks:
            with open(stage / f'{task}.log', 'a') as log:
                child = subprocess.Popen([sys.executable, '-u', *command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(RUN / 'status.json', {'status': 'running', 'stage': args.stage, 'task': task, 'pid': os.getpid(), 'child_pid': child.pid, 'updated_utc': now()})
                code = child.wait()
                if code:
                    raise RuntimeError(f'{task} exited {code}; see {stage}/{task}.log')
        assert read(stage / 'verification.json')['status'] == 'passed'
        write(RUN / 'status.json', {'status': 'stage_complete', 'stage': args.stage, 'task': 'verified', 'completed_utc': now()})
    except BaseException as exc:
        write(RUN / 'status.json', {'status': 'failed', 'stage': args.stage, 'error': repr(exc), 'updated_utc': now()})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=60)


if __name__ == '__main__':
    main()
