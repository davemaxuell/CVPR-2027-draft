"""Advance from verified four-model features into matched scorer training."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scorer_four.common import FEATURES_RUN, now, read, write
from scorer_four.training_common import RUN


def main():
    folder = ROOT / 'runs/automatic_four_training'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'watch.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        while not (FEATURES_RUN / 'result.json').exists():
            for path in (FEATURES_RUN / 'status.json', ROOT / 'runs/automatic_four_features/status.json'):
                if path.exists(): assert read(path)['status'] != 'failed', f'Predecessor failed: {path}'
            write(folder / 'status.json', {'status':'waiting_for_four_model_features', 'pid':os.getpid(), 'updated_utc':now()})
            time.sleep(10)
        assert read(FEATURES_RUN / 'result.json')['status'] == 'complete'
        if not (RUN / 'training_complete.json').exists():
            with (folder / 'run.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', 'tools/run_four_training.py'], cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'status.json', {'status':'running', 'child_pid':child.pid, 'updated_utc':now()})
                assert child.wait() == 0, 'Scorer training failed; inspect run.log'
        assert read(RUN / 'training_complete.json')['status'] == 'trained'
        write(folder / 'status.json', {'status':'four_model_scorer_trained', 'updated_utc':now()})
    except BaseException as error:
        write(folder / 'status.json', {'status':'failed', 'error':repr(error), 'updated_utc':now()}); raise


if __name__ == '__main__':
    main()
