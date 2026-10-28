"""Honor standing progression authorization after the initial DINO pilot completes."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from nwm_training.common import RUN, now, read, write


def main():
    folder = ROOT / 'runs/automatic_nwm'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    predecessor = ROOT / 'runs/step10_smolvlm2_refined512_dino_pair_v1'
    try:
        while not (predecessor / 'result.json').exists():
            for source in (ROOT / 'runs/step9_smolvlm2_refined512_dino_pair_v1/status.json', predecessor / 'status.json'):
                if source.exists(): assert read(source)['status'] != 'failed', f'Predecessor failed: {source}'
            write(folder / 'status.json', {'status': 'waiting_for_DINO_E_completion', 'pid': os.getpid(), 'updated_utc': now()})
            time.sleep(10)
        assert read(predecessor / 'result.json')['status'] == 'complete'
        with (folder / 'run.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', 'tools/run_nwm_pair.py'], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(folder / 'status.json', {'status': 'running_NWM_repeat', 'pid': os.getpid(), 'child_pid': child.pid, 'updated_utc': now()})
            write(RUN / 'job.json', {'pid': child.pid, 'started_utc': now(), 'command': 'tools/run_nwm_pair.py',
                'authorization': 'Standing user request to proceed automatically after each completed step'})
            assert child.wait() == 0, 'NWM repeat failed; inspect its stage status and logs'
        assert read(RUN / 'result.json')['status'] == 'complete'
        write(folder / 'status.json', {'status': 'NWM_pair_complete', 'next': 'Step 7B pooled four-model scorer features', 'updated_utc': now()})
    except BaseException as error:
        write(folder / 'status.json', {'status': 'failed', 'error': repr(error), 'updated_utc': now()}); raise


if __name__ == '__main__':
    main()
