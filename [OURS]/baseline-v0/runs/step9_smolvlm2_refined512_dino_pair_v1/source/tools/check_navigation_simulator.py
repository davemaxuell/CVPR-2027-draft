"""Complete native base-policy episodes must reproduce the selected reference."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
from navigation_eval.jobs import make_job,run_job
from scorer_features.common import now,read,sha,write

RUN=ROOT/'runs/step9_smolvlm2_refined512_dino_pair_v1'

def main():
    source=ROOT/'runs/step3b_smolvlm2_stop/confirmation_A/candidate'
    config=read(source/'eval_config.json')
    selection={'group':'C','split':'train','episodes':{n:ids[:1] for n,ids in config['selected_episodes'].items()},
        'partition_sha256':sha(ROOT/'manifest/scene_partition.json'),'selection':'First episode of each dataset in the existing locked confirmation set; runtime parity only'}
    selection_path=RUN/'runtime_selection.json'
    if selection_path.exists():assert read(selection_path)==selection
    else:write(selection_path,selection)
    path=make_job(RUN/'runtime_policy',selection_path,mode='policy',workers=2)
    result=run_job(path);checks=[]
    for name,ids in selection['episodes'].items():
        for eid in ids:
            previous=read(source/'episodes'/f'{name}_{eid}.json')
            current=read(RUN/'runtime_policy/episodes'/f'{name}_{eid}.json')
            for key in ('executed_actions','collided_after_action','positions','success','spl','distance_to_goal'):
                assert current[key]==previous[key],(name,eid,key)
            assert abs(current['ndtw']-previous['ndtw'])<1e-12
            assert [e['raw_action_scores'] for e in current['events']]==previous['action_scores']
            checks.append({'dataset':name,'episode_id':eid,'steps':current['steps'],'exact':True})
    write(RUN/'check_simulator.json',{'status':'passed','completed_utc':now(),'complete_reference_episode_parity':checks,
        'runtime_job_sha256':sha(path),'source_sha256':read(path)['source_sha256']})

if __name__=='__main__':main()
