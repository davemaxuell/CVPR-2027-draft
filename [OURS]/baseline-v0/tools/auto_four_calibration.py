"""Automatically verify, calibrate on C, and run the predeclared seven-system E repeat."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from navigation_four.common import CALIBRATION, EVALUATION, TRAIN, now, read, write


def main():
    folder=ROOT/'runs/automatic_four_calibration';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    def interrupted(signum,frame):raise KeyboardInterrupt('Automatic four-model continuation interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        HF_HOME='/data/shared/hf_cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def stage(name,script):
        nonlocal child
        with (folder/f'{name}.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',script],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(folder/'status.json',{'status':'running','stage':name,'child_pid':child.pid,'updated_utc':now()})
            assert child.wait()==0,f'{name} failed; inspect {name}.log'
            child=None
    try:
        while not (TRAIN/'training_complete.json').exists():
            for path in (TRAIN/'status.json',ROOT/'runs/automatic_four_training/status.json'):
                if path.exists():assert read(path)['status']!='failed',f'Predecessor failed: {path}'
            write(folder/'status.json',{'status':'waiting_for_four_model_training','pid':os.getpid(),'updated_utc':now()})
            time.sleep(10)
        assert read(TRAIN/'training_complete.json')['status']=='trained'
        if not (CALIBRATION/'protocol.json').exists():
            stage('check_runtime','tools/check_four_navigation.py')
            stage('prepare_C','tools/prepare_four_calibration.py')
        stage('calibrate_C','tools/run_four_calibration.py')
        assert read(CALIBRATION/'result.json')['status']=='complete'
        if not (EVALUATION/'protocol.json').exists():
            stage('check_reporting','tools/check_four_evaluation.py')
            stage('prepare_E','tools/prepare_four_evaluation.py')
        stage('evaluate_E','tools/run_four_evaluation.py')
        assert read(EVALUATION/'result.json')['status']=='complete'
        write(folder/'status.json',{'status':'four_model_SmolVLM2_trial_complete','updated_utc':now(),
            'next':'Remaining recorded policy repeats under standing automatic-progression authorization'})
    except BaseException as error:
        write(folder/'status.json',{'status':'failed','error':repr(error),'updated_utc':now()});raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
