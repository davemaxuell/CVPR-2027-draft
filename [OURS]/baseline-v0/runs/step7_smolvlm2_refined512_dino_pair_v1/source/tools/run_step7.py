"""Run frozen extraction and its audit; no optimizer or scorer training."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scorer_features.common import ROOT,RUN,now,read,verify_protocol,write

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify_protocol()
    if (RUN/'result.json').exists() and read(RUN/'result.json').get('status')=='complete':return
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','HF_HOME':'/data/shared/hf_cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'TOKENIZERS_PARALLELISM':'false'})
    try:
        for stage in ('policy','vision','predictions','verification'):
            commands=([[sys.executable,'-u','tools/cache_step7.py','policy','--rank',str(i),'--workers',str(p['runtime']['policy_workers'])]
                       for i in range(p['runtime']['policy_workers'])] if stage=='policy' else
                      [[sys.executable,'-u','tools/verify_step7.py']] if stage=='verification' else
                      [[sys.executable,'-u','tools/cache_step7.py',stage]])
            children=[]
            for i,cmd in enumerate(commands):
                log=(RUN/f'{stage}_{i}.log').open('a')
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append((proc,log))
            write(RUN/'status.json',{'status':'running','stage':stage,'pid':os.getpid(),
                'children':[c.pid for c,_ in children],'updated_utc':now()})
            codes=[]
            for child,log in children:codes.append(child.wait());log.close()
            if any(codes):raise RuntimeError(f'{stage} failed with exit codes {codes}; see stage logs')
    except BaseException as exc:
        write(RUN/'status.json',{'status':'failed','stage':stage,'error':str(exc),'updated_utc':now()});raise

if __name__=='__main__':main()
