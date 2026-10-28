"""Select an untouched E pilot only after C deployment settings are frozen."""
import os
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
from habitat.datasets import make_dataset
from navigation_eval.jobs import source_hashes
from scorer_features.common import POLICY_ID,TRIALS,read,sha,write
from sim.config import EPISODES,EXPERT,make_ce_config,scene_group
from sim.episodes import select_episodes
from tools.run_step9 import verify as verify_calibration

RUN=ROOT/'runs/step10_smolvlm2_refined512_dino_pair_v1'
CALIBRATION=ROOT/'runs/step9_smolvlm2_refined512_dino_pair_v1'


def main():
    checks=read(RUN/'check_reporting.json');assert checks['status']=='passed' and checks['no_E_data_used']
    assert sha(ROOT/'tools/finalize_step10.py')==checks['source_sha256']
    calibration=verify_calibration();result=read(CALIBRATION/'result.json')
    assert result['status']=='complete' and result['E_not_used_for_training_or_selection']
    assert result['protocol_sha256']==sha(CALIBRATION/'protocol.json')
    plan=calibration['step10_plan'];assert plan['group']=='E' and plan['episodes_per_dataset']==112
    selections={};scene_ids={}
    for name in ('r2r','rxr'):
        cfg=make_ce_config(name,'val_unseen',scenes=scene_group('E'),gpu=0)
        dataset=make_dataset(cfg.habitat.dataset.type,config=cfg.habitat.dataset)
        chosen=select_episodes(dataset.episodes,plan['episodes_per_dataset'],plan['selection_seed'])
        assert len(chosen)==112 and {e.scene_id.split('/')[-2] for e in chosen}==set(scene_group('E'))
        selections[name]=[str(e.episode_id) for e in chosen]
        scene_ids[name]={str(e.episode_id):e.scene_id.split('/')[-2] for e in chosen}
    selection={'group':'E','split':'val_unseen','episodes':selections,'episode_scenes':scene_ids,
        'partition_sha256':sha(ROOT/'manifest/scene_partition.json'),'seed':plan['selection_seed'],
        'selection':'Scene-balanced seeded subset, unfiltered by outcomes or expert replay; same episodes for all systems',
        'scope':plan['scope'],'dataset_file_sha256':{}}
    for name in ('r2r','rxr'):
        for table in (EPISODES,EXPERT):
            file=table[name].format(split='val_unseen',role='guide');selection['dataset_file_sha256'][file]=sha(file)
    if (RUN/'selection.json').exists():assert read(RUN/'selection.json')==selection
    else:write(RUN/'selection.json',selection)
    frozen={}
    for variant in ('pooled','null'):
        folder=CALIBRATION/'frozen'/variant
        for name,digest in read(folder/'file_sha256.json').items():assert sha(folder/name)==digest
        frozen[variant]=read(folder/'bundle.json')
        assert frozen[variant]['policy_identity_sha256']==POLICY_ID and frozen[variant]['STOP_bias']==-.25
    systems=[{'name':'policy','mode':'policy','checkpoint':None,'world_model':None,'margin':None},
        {'name':'null','mode':'null','checkpoint':frozen['null']['checkpoint'],'world_model':None,'margin':frozen['null']['margin']},
        {'name':'actual','mode':'actual','checkpoint':frozen['pooled']['checkpoint'],'world_model':None,'margin':frozen['pooled']['margin']}]
    for name in TRIALS:systems.append({'name':name,'mode':'predicted','checkpoint':frozen['pooled']['checkpoint'],'world_model':name,'margin':frozen['pooled']['margin']})
    for system in systems:system['checkpoint_sha256']=sha(system['checkpoint']) if system['checkpoint'] else None
    sources=source_hashes()
    for p in list((ROOT/'experiment_flow').glob('*.py'))+list((ROOT/'tools').glob('*step10*.py')):
        sources[str(p.relative_to(ROOT))]=sha(p)
    sources['tools/run_step9.py']=sha(ROOT/'tools/run_step9.py')
    protocol={'format':'controlled_E_evaluation_protocol_v1','policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,
        'calibration_result_sha256':sha(CALIBRATION/'result.json'),'calibration_protocol_sha256':sha(CALIBRATION/'protocol.json'),
        'selection_sha256':sha(RUN/'selection.json'),'systems':systems,'workers':4,'sampling_seed':0,
        'episodes_per_system':224,'scope':plan['scope'],'RxR_protocol':'English guide, unified 15-degree turns; not official 30-degree RxR-CE',
        'metrics':['SR','SPL','navigation_error','nDTW','SDTW','forward_collision_rate','steps','override_and_fallback_frequency','latency','GPU_memory'],
        'confidence_intervals':'Paired scene-cluster bootstrap, 2000 resamples, seed 10; no branch-as-independent-episode intervals',
        'actual_control':'Privileged actual branch-next RGB with the identical pooled scorer and margin; never a deployable comparison',
        'no_E_tuning':True,'source_sha256':sources}
    protocol['reporting_checks_sha256']=sha(RUN/'check_reporting.json')
    if (RUN/'protocol.json').exists():assert read(RUN/'protocol.json')==protocol
    else:
        for name in sources:
            dest=RUN/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        write(RUN/'protocol.json',protocol)
    print('Locked',sha(RUN/'protocol.json'))

if __name__=='__main__':main()
