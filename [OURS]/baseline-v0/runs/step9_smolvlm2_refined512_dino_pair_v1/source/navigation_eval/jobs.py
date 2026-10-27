"""Immutable evaluation jobs with independent native batch-one workers."""
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scorer_features.common import ROOT,POLICY_ID,now,read,sha,write
from sim.config import load_manifest
from .metrics import aggregate


def source_hashes():
    paths=list((ROOT/'navigation_eval').glob('*.py'))+[ROOT/'tools/eval_navigation.py']
    names=['Scorer/scorer.py','Scorer/interfaces.py','Scorer/world_models.py','policy/common.py','policy/online.py',
        'policy/smolvlm2.py','policy/experimental_reference.py','scorer_features/policy.py','scorer_features/common.py',
        'scorer_training/replay.py','wm/dino.py','wm/frozen.py','wm/common.py','sim/config.py','sim/episodes.py',
        'sim/rxr_dataset.py','sim/transition_branch.py','scorer_data/collection.py','scorer_data/metrics.py']
    paths += [ROOT/name for name in names]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}


def make_job(output,selection,*,mode,checkpoint=None,world_model=None,margin=None,workers=4,seed=0):
    output=Path(output).resolve();selection=Path(selection).resolve();s=read(selection)
    job={'format':'closed_loop_evaluation_job_v1','output':str(output),'group':s['group'],
        'selection_file':str(selection),'selection_sha256':sha(selection),
        'mode':mode,'checkpoint':str(Path(checkpoint).resolve()) if checkpoint else None,
        'checkpoint_sha256':sha(checkpoint) if checkpoint else None,'world_model':world_model,'margin':margin,
        'workers':workers,'seed':seed,'decoding_batch_size':1,'policy_identity_sha256':POLICY_ID,
        'STOP_bias':-.25,'scorer_dtype':'float32','matmul_tf32':False,'cudnn_tf32':True,
        'minimum_real_observations':4,'simulator':load_manifest()['simulator'],'source_sha256':source_hashes()}
    path=output/'job.json'
    if path.exists():assert read(path)==job
    else:write(path,job)
    return path


def run_job(path):
    path=Path(path);job=read(path);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest=sha(path)
    for name,h in job['source_sha256'].items():assert sha(ROOT/name)==h,name
    if (out/'summary.json').exists():
        result=read(out/'summary.json');assert result['status']=='complete' and result['job_sha256']==digest;return result
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'HF_HOME':'/data/shared/hf_cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
        'MAGNUM_LOG':'quiet','HABITAT_SIM_LOG':'quiet','GLOG_minloglevel':'2'})
    children=[];logs=[];started=time.time()
    try:
        for rank in range(job['workers']):
            folder=out/'workers'/str(rank);folder.mkdir(parents=True,exist_ok=True)
            log=(folder/'stdout.log').open('a');logs.append(log)
            children.append(subprocess.Popen([sys.executable,'-u','tools/eval_navigation.py','--job',str(path),'--rank',str(rank)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        write(out/'status.json',{'status':'running','pid':os.getpid(),'children':[c.pid for c in children],'started_utc':now()})
        previous=-1
        while True:
            codes=[c.poll() for c in children]
            if any(c is not None and c!=0 for c in codes):raise RuntimeError(f'Worker failure {codes}: {out}')
            rows=[read(p) for p in sorted((out/'episodes').glob('*.json'))] if (out/'episodes').exists() else []
            if len(rows)!=previous:
                write(out/'progress.json',{'status':'running','completed':len(rows),'seconds':time.time()-started,'updated_utc':now()})
                print({'job':out.name,'completed':len(rows)},flush=True);previous=len(rows)
            if all(c==0 for c in codes):break
            time.sleep(5)
        selection=read(job['selection_file']);expected={(n,eid) for n,ids in selection['episodes'].items() for eid in ids}
        assert len(rows)==len(expected) and {(r['dataset'],r['episode_id']) for r in rows}==expected
        assert all(r['job_sha256']==digest for r in rows)
        latencies=[v for r in rows for v in r['decision_seconds']]
        report={'status':'complete','completed_utc':now(),'job_sha256':digest,'seconds_this_run':time.time()-started,
            'group':job['group'],'mode':job['mode'],'world_model':job['world_model'],'margin':job['margin'],
            'checkpoint_sha256':job['checkpoint_sha256'],**aggregate(rows,bootstrap=True),
            'decision_latency_seconds':{'mean':float(np.mean(latencies)),'median':float(np.median(latencies)),
                'p95':float(np.quantile(latencies,.95)),'observations':len(latencies),'concurrent_workers':job['workers']},
            'torch_peak_allocated_bytes_per_process':max(r['torch_peak_allocated_bytes_process'] for r in rows),
            'torch_peak_reserved_bytes_per_process':max(r['torch_peak_reserved_bytes_process'] for r in rows),
            'resource_scope':'PyTorch allocator memory excludes Habitat allocations; measured decision latency includes concurrent-worker contention',
            'episode_sha256':{p.name:sha(p) for p in sorted((out/'episodes').glob('*.json'))}}
        write(out/'summary.json',report);write(out/'status.json',{'status':'complete','completed_utc':now()})
        return report
    except BaseException as exc:
        write(out/'status.json',{'status':'failed','error':repr(exc),'updated_utc':now()});raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()
