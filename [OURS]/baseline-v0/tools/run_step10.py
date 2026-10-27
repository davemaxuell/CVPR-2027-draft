import fcntl
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiment_flow.resources import monitored_run
from navigation_eval.jobs import make_job
from scorer_features.common import now,read,sha,write

RUN=ROOT/'runs/step10_smolvlm2_refined512_dino_pair_v1'
CALIBRATION=ROOT/'runs/step9_smolvlm2_refined512_dino_pair_v1'


def verify():
    p=read(RUN/'protocol.json')
    assert sha(CALIBRATION/'result.json')==p['calibration_result_sha256']
    assert sha(CALIBRATION/'protocol.json')==p['calibration_protocol_sha256']
    assert sha(RUN/'selection.json')==p['selection_sha256']
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==sha(RUN/'source'/name)==digest,name
    for s in p['systems']:
        if s['checkpoint']:assert sha(s['checkpoint'])==s['checkpoint_sha256']
    return p


def main():
    lock=(RUN/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify()
    if (RUN/'result.json').exists():assert read(RUN/'result.json')['status']=='complete';return
    try:
        for system in p['systems']:
            verify();out=RUN/'evaluation'/system['name']
            job=make_job(out,RUN/'selection.json',mode=system['mode'],checkpoint=system['checkpoint'],
                world_model=system['world_model'],margin=system['margin'],workers=p['workers'],seed=p['sampling_seed'])
            write(RUN/'status.json',{'status':'running','system':system['name'],'job':str(job),'pid':os.getpid(),'updated_utc':now()})
            monitored_run(job)
        verify()
        write(RUN/'status.json',{'status':'verifying','updated_utc':now()})
        subprocess.run([sys.executable,'-u','tools/finalize_step10.py'],cwd=ROOT,check=True)
    except BaseException as exc:
        write(RUN/'status.json',{'status':'failed','error':repr(exc),'updated_utc':now()});raise

if __name__=='__main__':main()
