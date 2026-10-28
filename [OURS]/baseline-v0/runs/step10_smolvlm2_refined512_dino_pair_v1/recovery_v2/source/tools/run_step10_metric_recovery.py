"""Versioned reporting-only recovery; preserve C choices, E population, and old records."""
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scorer_features.common import now,read,sha,write
from navigation_metrics_v2.jobs import make_job,source_hashes
from navigation_metrics_v2.metrics import normalize_episode
from navigation_metrics_v2.resources import monitored_run
from tools.run_step10 import RUN as ORIGINAL_RUN,CALIBRATION,verify as verify_original

RUN=ORIGINAL_RUN/'recovery_v2'


def prepare():
    original=verify_original();checks=read(RUN/'check_metrics.json');assert checks['status']=='passed'
    for name,digest in checks['source_sha256'].items():assert sha(ROOT/name)==digest,name
    selection=ORIGINAL_RUN/'selection.json'
    if not (RUN/'selection.json').exists():shutil.copy2(selection,RUN/'selection.json')
    assert sha(RUN/'selection.json')==original['selection_sha256']
    old=ORIGINAL_RUN/'evaluation/policy'
    rows=sorted((old/'episodes').glob('*.json'));assert len(rows)==184
    inputs=[ORIGINAL_RUN/'protocol.json',RUN/'check_metrics.json',old/'job.json',old/'gpu_memory.json',*rows]
    sources=source_hashes()
    for name in ('tools/run_step10_metric_recovery.py','tools/finalize_step10_metric_recovery.py','tools/check_metric_reporting_v2.py'):
        sources[name]=sha(ROOT/name)
    protocol={**original,'format':'controlled_E_metric_reporting_recovery_v2',
        'original_protocol_sha256':sha(ORIGINAL_RUN/'protocol.json'),'initial_attempt':str(ORIGINAL_RUN),
        'amendment':'Post-rollout reporting only: retain infinite geodesic NE explicitly; zero SPL for native NaN in a failed episode with infinite initial distance. No action, model, preprocessing, selection or calibration changes.',
        'observed_failure':'RxR5734: infinite distance from initial pose; native SPL 0*(inf/inf) is NaN.',
        'declared_utc':now(),'imported_complete_policy_episodes':184,
        'E_population_unchanged':True,'no_E_tuning':True,
        'input_sha256':{str(path):sha(path) for path in inputs},'source_sha256':sources}
    if (RUN/'protocol.json').exists():return
    for name in sources:
        path=RUN/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    write(RUN/'protocol.json',protocol)


def verify():
    original=verify_original();p=read(RUN/'protocol.json')
    assert sha(ORIGINAL_RUN/'protocol.json')==p['original_protocol_sha256']
    assert p['systems']==original['systems'] and p['selection_sha256']==original['selection_sha256']
    assert sha(RUN/'selection.json')==p['selection_sha256']
    for path,digest in p['input_sha256'].items():assert sha(path)==digest,path
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==sha(RUN/'source'/name)==digest,name
    return p


def import_policy_rows(job):
    out=Path(read(job)['output']);old=ORIGINAL_RUN/'evaluation/policy'
    for path in sorted((old/'episodes').glob('*.json')):
        target=out/'episodes'/path.name
        if target.exists():continue
        row=read(path);assert row['job_sha256']==sha(old/'job.json')
        normalized=normalize_episode(row)
        normalized['imported_episode']={'file':str(path),'sha256':sha(path),'original_job_sha256':row['job_sha256']}
        normalized['job_sha256']=sha(job);write(target,normalized)


def combine_policy_memory(out):
    path=out/'gpu_memory.json';current=read(path)
    if current.get('imported_sampling_from'):return
    source=ORIGINAL_RUN/'evaluation/policy/gpu_memory.json';old=read(source)
    samples=old['samples']+current['samples'];peaks={}
    for sample in samples:
        for pid,value in sample['process_mib'].items():peaks[pid]=max(peaks.get(pid,0),value)
    write(path,{**current,'samples':samples,'peak_mib_by_process':peaks,
        'max_process_peak_mib':max(peaks.values()),
        'peak_sum_of_worker_processes_mib':max(sum(x['process_mib'].values()) for x in samples),
        'errors':old['errors']+current['errors'],'imported_sampling_from':str(source),'imported_sampling_sha256':sha(source)})


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (RUN/'protocol.json').exists():prepare()
    p=verify()
    if (RUN/'result.json').exists():assert read(RUN/'result.json')['status']=='complete';return
    try:
        for system in p['systems']:
            verify();out=RUN/'evaluation'/system['name']
            job=make_job(out,RUN/'selection.json',mode=system['mode'],checkpoint=system['checkpoint'],
                world_model=system['world_model'],margin=system['margin'],workers=p['workers'],seed=p['sampling_seed'])
            if system['name']=='policy':import_policy_rows(job)
            status={'status':'running','system':system['name'],'job':str(job),'pid':os.getpid(),'updated_utc':now(),
                'reporting_recovery':str(RUN),'original_attempt_preserved':True}
            write(RUN/'status.json',status);write(ORIGINAL_RUN/'status.json',status)
            monitored_run(job)
            if system['name']=='policy':combine_policy_memory(out)
        verify()
        subprocess.run([sys.executable,'-u','tools/finalize_step10_metric_recovery.py'],cwd=ROOT,check=True)
        assert read(RUN/'result.json')['status']=='complete'
        # Publish verified recovery as the Step10 completion record consumed by
        # the already-authorized successors. Original protocol/episodes stay intact.
        for name in ('verification.json','metrics.csv','RESULTS.md'):
            if (ORIGINAL_RUN/name).exists():assert sha(ORIGINAL_RUN/name)==sha(RUN/name)
            else:shutil.copy2(RUN/name,ORIGINAL_RUN/name)
        result={**read(RUN/'result.json'),'evaluation_directory':str(RUN/'evaluation'),
            'recovery_directory':str(RUN),'original_protocol_sha256':sha(ORIGINAL_RUN/'protocol.json')}
        write(ORIGINAL_RUN/'result.json',result)
        write(ORIGINAL_RUN/'status.json',{'status':'complete','completed_utc':now(),'reporting_recovery':str(RUN)})
        write(ROOT/'runs/automatic_smolvlm2/status.json',{'status':'initial_smol_dino_trial_complete','completed_utc':now(),'reporting_recovery':str(RUN)})
    except BaseException as error:
        status={'status':'failed','error':repr(error),'reporting_recovery':str(RUN),'updated_utc':now()}
        write(RUN/'status.json',status);write(ORIGINAL_RUN/'status.json',status);raise


if __name__=='__main__':main()
