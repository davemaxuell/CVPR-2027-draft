"""Run and audit the predeclared four-model scorer fit, then hand off to C."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scorer_four.training_common import RUN, now, read, sha, verify_protocol, write


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'training_complete.json').exists():
        verify_protocol(); assert read(RUN / 'training_complete.json')['status'] == 'trained'; return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    child = None
    def interrupted(signum, frame): raise KeyboardInterrupt('Four-model scorer fit interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    def stage(name, arguments):
        nonlocal child
        with (RUN / f'{name}.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(RUN / 'status.json', {'status':'running', 'stage':name, 'pid':os.getpid(),
                'child_pid':child.pid, 'updated_utc':now()})
            while child.poll() is None: time.sleep(5)
            assert child.returncode == 0, f'{name} failed; inspect {name}.log'
            child = None
    try:
        if not (RUN / 'protocol.json').exists():
            stage('check', ['tools/check_four_training.py'])
            stage('prepare', ['tools/prepare_four_training.py'])
        protocol = verify_protocol(); variants = {}
        for variant in ('pooled', 'null'):
            stage('train_' + variant, ['tools/train_four.py', '--variant', variant])
            result = read(RUN / variant / 'result.json')
            assert result['status'] == 'trained' and result['updates'] == protocol['total_updates']
            assert result['protocol_sha256'] == sha(RUN / 'protocol.json')
            for record in result['all_epoch_metrics']:
                path = RUN / variant / 'checkpoints' / f"step_{record['step']:06d}.pt"
                assert sha(path) == read(path.with_suffix('.json'))['sha256']
                assert sha(RUN / variant / record['evaluation']) == record['evaluation_sha256']
                if variant == 'null' or record['stage'] == 'actual': assert record['inherited_DINO_fit_bit_exact']
            variants[variant] = {'result':str(RUN / variant / 'result.json'),
                'sha256':sha(RUN / variant / 'result.json'), 'shortlist':result['closed_loop_shortlist']}
        verify_protocol()
        write(RUN / 'training_complete.json', {'status':'trained', 'completed_utc':now(),
            'protocol_sha256':sha(RUN / 'protocol.json'), 'variants':variants,
            'no_E_fitting_or_selection':True, 'next':'C closed-loop calibration of one scorer shared across four world models'})
        write(RUN / 'status.json', {'status':'trained_pending_closed_loop_selection', 'updated_utc':now()})
    except BaseException as error:
        write(RUN / 'status.json', {'status':'failed', 'error':repr(error), 'updated_utc':now()}); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()


if __name__ == '__main__':
    main()
