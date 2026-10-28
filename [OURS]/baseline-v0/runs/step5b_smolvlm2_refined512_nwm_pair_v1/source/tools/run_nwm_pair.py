"""Sequential supervised NWM/CDiT repeat with failure records and resumable stages."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from nwm_training.common import CACHE, RUN, now, read, write


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'pipeline.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RUN / 'result.json').exists():
        assert read(RUN / 'result.json')['status'] == 'complete'; return
    predecessor = ROOT / 'runs/step10_smolvlm2_refined512_dino_pair_v1/result.json'
    assert read(predecessor)['status'] == 'complete'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', HF_HOME='/data/shared/hf_cache',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    child = None
    def interrupted(signum, frame): raise KeyboardInterrupt('NWM/CDiT repeat interrupted')
    signal.signal(signal.SIGTERM, interrupted)

    def stage(name, arguments):
        nonlocal child
        with (RUN / f'{name}.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(RUN / 'status.json', {'status': 'running', 'stage': name, 'pid': os.getpid(),
                'child_pid': child.pid, 'updated_utc': now()})
            while child.poll() is None: time.sleep(5)
            assert child.returncode == 0, f'{name} failed ({child.returncode}); inspect {name}.log'
            child = None

    try:
        if not (RUN / 'protocol.json').exists():
            stage('check_cpu', ['tools/check_nwm_cpu.py'])
            stage('check_gpu', ['tools/check_nwm_gpu.py'])
            stage('prepare', ['tools/prepare_nwm_protocol.py'])
        from nwm_training.protocol import verify_protocol
        plan = verify_protocol()
        stage('cache', ['tools/cache_nwm_posteriors.py', '--batch-size', str(plan['cache_batch_size'])])
        for trial in ('transfer','fresh'):
            stage('train_' + trial, ['tools/train_nwm.py', '--trial', trial])
        stage('freeze', ['tools/freeze_nwm.py'])
        stage('quality', ['tools/evaluate_nwm_features.py'])
        stage('finalize', ['tools/finalize_nwm.py'])
        assert read(RUN / 'result.json')['status'] == 'complete'
    except BaseException as error:
        write(RUN / 'status.json', {'status': 'failed', 'error': repr(error), 'updated_utc': now()}); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()


if __name__ == '__main__':
    main()
