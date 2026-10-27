"""Continue automatically to the next feature stage after NWM/CDiT verification."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scorer_four.common import FEATURES_RUN, NWM_RUN, now, read, write


def main():
    folder = ROOT / 'runs/automatic_four_features'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        while not (NWM_RUN / 'result.json').exists():
            for path in (NWM_RUN / 'status.json', ROOT / 'runs/automatic_nwm/status.json'):
                if path.exists(): assert read(path)['status'] != 'failed', f'Predecessor failed: {path}'
            write(folder / 'status.json', {'status': 'waiting_for_NWM_pair', 'pid': os.getpid(), 'updated_utc': now()})
            time.sleep(10)
        assert read(NWM_RUN / 'result.json')['status'] == 'complete'
        if not (FEATURES_RUN / 'result.json').exists():
            for stage in ('prepare','cache','verify'):
                with (folder / f'{stage}.log').open('a') as log:
                    child = subprocess.Popen([sys.executable, '-u', f'tools/{stage}_four_features.py'], cwd=ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    write(folder / 'status.json', {'status': 'running', 'stage': stage, 'child_pid': child.pid, 'updated_utc': now()})
                    assert child.wait() == 0, f'{stage} failed; inspect {stage}.log'
        assert read(FEATURES_RUN / 'result.json')['status'] == 'complete'
        write(folder / 'status.json', {'status': 'four_model_features_complete', 'updated_utc': now()})
    except BaseException as error:
        write(folder / 'status.json', {'status': 'failed', 'error': repr(error), 'updated_utc': now()}); raise


if __name__ == '__main__':
    main()
