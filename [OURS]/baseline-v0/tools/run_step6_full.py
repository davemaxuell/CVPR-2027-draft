"""Run the locked full Step 6 budget, audit it, then publish its catalogs."""
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
    lock = open(RUN / 'full.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(RUN / 'plan.json')
    assert read(RUN / 'pilot/verification.json')['status'] == 'passed'
    assert sha(RUN / 'pilot/verification.json') == plan['pilot_verification_sha256']
    assert plan['protocol_sha256'] == sha(RUN / 'protocol.json')
    for path, expected in plan['orchestration_source_sha256'].items():
        assert sha(ROOT / path) == expected
    if (RUN / 'result.json').exists() and read(RUN / 'result.json')['status'] == 'complete':
        return
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Step 6 interrupted')
    signal.signal(signal.SIGTERM, interrupt)
    try:
        commands = [('full_pipeline', ['tools/run_step6_stage.py', '--stage', 'full', '--groups', 'B', 'C',
                     '--roots-b', str(plan['roots_B']), '--roots-c', str(plan['roots_C'])]),
                    ('finalize', ['tools/finalize_step6.py', '--stage', 'full'])]
        for name, command in commands:
            with open(RUN / f'{name}.log', 'a') as log:
                child = subprocess.Popen([sys.executable, '-u', *command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(RUN / 'full_process.json', {'pid': os.getpid(), 'child_pid': child.pid, 'task': name, 'updated_utc': now()})
                code = child.wait()
                if code:
                    raise RuntimeError(f'{name} exited {code}; see {RUN}/{name}.log')
    except BaseException as exc:
        write(RUN / 'status.json', {'status': 'failed', 'stage': 'full', 'error': repr(exc), 'updated_utc': now()})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=60)


if __name__ == '__main__':
    main()
