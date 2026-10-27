"""Run both matched training variants; closed-loop selection follows automatically."""
import fcntl
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scorer_training.common import RUN,now,verify_protocol,write


def main():
    lock=(RUN/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_protocol()
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'HF_HOME':'/data/shared/hf_cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    try:
        for variant in ('pooled','null'):
            with (RUN/f'{variant}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u','tools/train_step8.py',variant],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(RUN/'status.json',{'status':'training','variant':variant,'pid':os.getpid(),'child':child.pid,'updated_utc':now()})
                code=child.wait()
                if code:raise RuntimeError(f'{variant} exited {code}; see {variant}.log')
        write(RUN/'status.json',{'status':'trained_pending_closed_loop_selection','completed_utc':now(),
            'next':'Continue automatically with C closed-loop candidate validation and Step 9 calibration'})
    except BaseException as exc:
        write(RUN/'status.json',{'status':'failed','updated_utc':now(),'error':repr(exc)});raise

if __name__=='__main__':main()
