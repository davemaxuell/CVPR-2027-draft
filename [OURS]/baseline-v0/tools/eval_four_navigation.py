"""Fixed-selection, resumable real navigation for calibration and final controls."""
import argparse
from collections import Counter
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('MAGNUM_LOG','quiet');os.environ.setdefault('HABITAT_SIM_LOG','quiet');os.environ.setdefault('GLOG_minloglevel','2')
import habitat
import numpy as np
import torch
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from PIL import Image
from navigation_four.controller import NavigationController
from navigation_eval.metrics import aggregate
from scorer_data.collection import reset_episode,pixel_sha
from scorer_data.metrics import terminal_ndtw
from scorer_features.common import POLICY_ID,now,read,sha,write
from sim.config import load_expert,load_manifest,make_ce_config,scene_group
from sim.transition_branch import movement_branches,pose


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--rank',type=int,default=0)
    args=ap.parse_args();job_path=Path(args.job).resolve();job=read(job_path);out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    assert 0<=args.rank<job['workers'] and job['policy_identity_sha256']==POLICY_ID
    for name,digest in job['source_sha256'].items():assert sha(ROOT/name)==digest,name
    if job.get('checkpoint'):assert sha(job['checkpoint'])==job['checkpoint_sha256']
    selection=read(job['selection_file']);assert sha(job['selection_file'])==job['selection_sha256']
    assert load_manifest()['simulator']==job['simulator']
    assert sha(ROOT/'manifest/scene_partition.json')==selection['partition_sha256']
    for path,digest in selection.get('dataset_file_sha256',{}).items():assert sha(path)==digest
    worker=out/'workers'/str(args.rank);worker.mkdir(parents=True,exist_ok=True);(out/'episodes').mkdir(exist_ok=True)
    group=job['group'];split='val_unseen' if group=='E' else 'train'
    assert selection['group']==group and selection['split']==split
    controller=NavigationController(job['mode'],job.get('checkpoint'),job.get('world_model'),job.get('margin'))
    torch.cuda.reset_peak_memory_stats();started=time.time();completed=[]
    for name in ('r2r','rxr'):
        cfg=make_ce_config(name,split,scenes=scene_group(group),gpu=0)
        full=make_dataset(cfg.habitat.dataset.type,config=cfg.habitat.dataset)
        lookup={str(e.episode_id):e for e in full.episodes};ids=selection['episodes'][name][args.rank::job['workers']]
        pending=[]
        for eid in ids:
            path=out/'episodes'/f'{name}_{eid}.json'
            if path.exists():
                row=read(path);assert row['job_sha256']==sha(job_path);completed.append(row)
            else:pending.append(lookup[eid])
        if not pending:continue
        expert=load_expert(name,split)
        ds=registry.get_dataset(cfg.habitat.dataset.type)(config=None)
        ds.episodes=[pending[0]];env=habitat.Env(config=cfg,dataset=ds)
        try:
            assert list(cfg.habitat.task.actions)==['stop','move_forward','turn_left','turn_right']
            for ep in sorted(pending,key=lambda e:(e.scene_id,str(e.episode_id))):
                assert ep.scene_id.split('/')[-2] in scene_group(group)
                obs=reset_episode(env,ep,job['seed']);budget=int(cfg.habitat.environment.max_episode_steps)
                episode_started=time.perf_counter();controller.reset(ep.instruction.instruction_text,budget)
                controller.set_episode_identity(group, name, str(ep.episode_id))
                actions=[];events=[];collisions=[];positions=[pose(env.sim)['position']];hashes=[pixel_sha(obs['rgb'])]
                distances=[float(env.get_metrics()['distance_to_goal'])];latencies=[]
                while not env.episode_over:
                    image=Image.fromarray(np.array(obs['rgb'],copy=True))
                    torch.cuda.synchronize();before=time.perf_counter()
                    controller.observe(image,actions)
                    preview=lambda:[Image.fromarray(x['rgb']) for x in movement_branches(env)]
                    action,event=controller.decide(preview)
                    torch.cuda.synchronize();latencies.append(time.perf_counter()-before)
                    obs=env.step(action);controller.commit(action)
                    actions.append(action);events.append(event)
                    collisions.append(bool(action==1 and env.sim.previous_step_collided))
                    positions.append(pose(env.sim)['position']);hashes.append(pixel_sha(obs['rgb']))
                    distances.append(float(env.get_metrics()['distance_to_goal']))
                    assert len(actions)==env._elapsed_steps<=budget
                metric=env.get_metrics();success=float(metric['success'])
                ndtw=terminal_ndtw(positions,expert[str(ep.episode_id)]['locations'])
                row={'format':'four_WM_closed_loop_scorer_evaluation_v1','job_sha256':sha(job_path),
                    'dataset':name,'episode_id':str(ep.episode_id),'scene':ep.scene_id.split('/')[-2],
                    'group':group,'mode':job['mode'],'world_model':job.get('world_model'),'margin':job.get('margin'),
                    'policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,'scorer_sha256':job.get('checkpoint_sha256'),
                    'success':success,'spl':float(metric['spl']),'distance_to_goal':float(metric['distance_to_goal']),
                    'ndtw':ndtw,'sdtw':success*ndtw,'steps':len(actions),'collisions':sum(collisions),
                    'forward_attempts':actions.count(1),'forward_collisions':sum(collisions),'stopped':actions[-1]==0,
                    'termination':'STOP' if actions[-1]==0 else 'step_limit','original_budget':budget,
                    'executed_actions':actions,'collided_after_action':collisions,'positions':positions,
                    'distances_to_goal':distances,'rgb_sha256':hashes,'events':events,
                    'overrides':sum(e['selected_live']!=e['base_live'] for e in events),
                    'reason_counts':dict(Counter(e['reason'] for e in events)),
                    'world_model_calls':sum(e['world_model_calls'] for e in events),
                    'decision_seconds':latencies,'decision_seconds_mean':float(np.mean(latencies)),
                    'decision_seconds_p95':float(np.quantile(latencies,.95)),
                    'seconds':time.perf_counter()-episode_started,'completed_utc':now(),
                    'torch_peak_allocated_bytes_process':torch.cuda.max_memory_allocated(),
                    'torch_peak_reserved_bytes_process':torch.cuda.max_memory_reserved(),
                    'resource_scope':'Per-process PyTorch memory excludes Habitat allocations; decision wall latency measured with declared concurrent workers'}
                assert all(np.isfinite(row[k]) for k in ('success','spl','distance_to_goal','ndtw','sdtw'))
                assert actions[-1]==0 or len(actions)==budget
                write(out/'episodes'/f'{name}_{ep.episode_id}.json',row);completed.append(row)
                progress={'status':'running','rank':args.rank,'completed':len(completed),'last_episode':f'{name}/{ep.episode_id}',
                    'seconds':time.time()-started,'updated_utc':now()}
                write(worker/'progress.json',progress);print(json.dumps(progress),flush=True)
        finally:env.close()
    write(worker/'status.json',{'status':'complete','episodes':len(completed),'completed_utc':now()})

if __name__=='__main__':main()
