"""Continue from running C calibration into the predeclared frozen E comparison."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scorer_features.common import now,read,sha,write

FLOW=ROOT/'runs/automatic_smolvlm2';C=ROOT/'runs/step9_smolvlm2_refined512_dino_pair_v1';E=ROOT/'runs/step10_smolvlm2_refined512_dino_pair_v1'


def main():
    FLOW.mkdir(parents=True,exist_ok=True)
    lock=(FLOW/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'HF_HOME':'/data/shared/hf_cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    try:
        while not (C/'result.json').exists():
            status=read(C/'status.json')
            if status['status']=='failed':raise RuntimeError('Step 9 failed; inspect its status/logs before continuing')
            write(FLOW/'status.json',{'status':'waiting_for_calibration','pid':os.getpid(),'updated_utc':now()})
            time.sleep(10)
        assert read(C/'result.json')['status']=='complete'
        for stage,script in [('prepare_step10','tools/prepare_step10.py'),('run_step10','tools/run_step10.py')]:
            with (FLOW/f'{stage}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',script],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(FLOW/'status.json',{'status':'running','stage':stage,'pid':os.getpid(),'child':child.pid,'updated_utc':now()})
                if stage=='run_step10':write(E/'job.json',{'pid':child.pid,'started_utc':now(),'protocol_sha256':sha(E/'protocol.json'),'automatic_parent':os.getpid()})
                code=child.wait()
                if code:raise RuntimeError(f'{stage} exited {code}')
        assert read(E/'result.json')['status']=='complete'
        write(FLOW/'status.json',{'status':'initial_smol_dino_trial_complete','completed_utc':now(),
            'next':'Remaining declared model/policy trials under automatic progression authorization'})
    except BaseException as exc:
        write(FLOW/'status.json',{'status':'failed','error':repr(exc),'updated_utc':now()});raise

if __name__=='__main__':main()
