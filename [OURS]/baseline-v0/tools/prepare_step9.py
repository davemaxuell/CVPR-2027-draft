"""Lock C-only candidate/margin screening before any new scorer navigation results."""
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scorer_features.common import CACHE,POLICY_ID,TRIALS,now,read,sha,write
from scorer_training.common import RUN as TRAIN,verify_protocol as verify_training
from navigation_eval.jobs import source_hashes
from sim.config import EPISODES,EXPERT

RUN=ROOT/'runs/step9_smolvlm2_refined512_dino_pair_v1'
GRID=[0.,.025,.05,.1,.2,.4,.8,1.]


def curves(evaluation,variant):
    assert evaluation['group']=='C'
    roots={r['root_id']:r for r in read(Path(read(CACHE/'manifest.json')['index'])/'roots.json')}
    names=['null'] if variant=='null' else list(TRIALS);result=[]
    for margin in GRID:
        regrets=[];overrides=[]
        for name in names:
            rr=[];oo=[]
            for row in evaluation['rows'][name]:
                assert row['root_id'].startswith('C/')
                root=roots[row['root_id']];scores=np.asarray(row['utility'],dtype=np.float32);base=root['base_action_canonical']
                best=int(np.argmax(scores));selected=best if scores[best]-scores[base]>np.float32(margin) else base
                target=np.array([root['candidates_canonical'][str(a)]['utility'] for a in range(4)])
                rr.append(float(target.max()-target[selected]));oo.append(selected!=base)
            regrets.append(float(np.mean(rr)));overrides.append(float(np.mean(oo)))
        result.append({'margin':margin,'mean_regret':float(np.mean(regrets)),
            'mean_override_fraction':float(np.mean(overrides)),'regret_by_representation':dict(zip(names,regrets))})
    return sorted(result,key=lambda x:(x['mean_regret'],-x['margin']))


def main():
    verify_training()
    for name in ('check_runtime.json','check_simulator.json'):
        report=read(RUN/name);assert report['status']=='passed'
        for source,digest in report['source_sha256'].items():assert sha(ROOT/source)==digest,source
    null_check=read(RUN/'check_null_runtime.json');assert null_check['status']=='passed'
    for source,digest in null_check['source_sha256'].items():assert sha(ROOT/source)==digest,source
    candidates=[];training_results={}
    for variant in ('pooled','null'):
        result=read(TRAIN/variant/'result.json');assert result['status']=='trained' and result['updates']==2048
        training_results[variant]=sha(TRAIN/variant/'result.json')
        for candidate in result['closed_loop_shortlist']:
            assert candidate['stage']=='predicted'
            evaluation=TRAIN/variant/candidate['evaluation'];assert sha(evaluation)==candidate['evaluation_sha256']
            table=curves(read(evaluation),variant)
            checkpoint=TRAIN/variant/'checkpoints'/f"step_{candidate['step']:06d}.pt"
            for option in table[:2]:
                identifier=f"{variant}_s{candidate['step']:06d}_m{option['margin']:g}".replace('.','p')
                candidates.append({'id':identifier,'variant':variant,'step':candidate['step'],
                    'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                    'branch_curve':table,'margin':option['margin'],'branch_regret':option['mean_regret']})
    confirmation=ROOT/'runs/step3b_smolvlm2_stop/confirmation_A/candidate/eval_config.json'
    config=read(confirmation);base={'group':'C','split':'train','partition_sha256':sha(ROOT/'manifest/scene_partition.json'),
        'confirmation_source':str(confirmation),'confirmation_source_sha256':sha(confirmation),
        'dataset_file_sha256':{EPISODES[n].format(split='train',role='guide'):sha(EPISODES[n].format(split='train',role='guide')) for n in ('r2r','rxr')}}
    base['dataset_file_sha256'].update({EXPERT[n].format(split='train',role='guide'):sha(EXPERT[n].format(split='train',role='guide')) for n in ('r2r','rxr')})
    for label,subset in [('screen',slice(None,32)),('remaining',slice(32,None)),('full',slice(None))]:
        selection={**base,'episodes':{n:ids[subset] for n,ids in config['selected_episodes'].items()},
            'selection':'Nested fixed confirmation episodes; no filtering by success, route length or expert replay'}
        path=RUN/f'selection_{label}.json'
        if path.exists():assert read(path)==selection
        else:write(path,selection)
    protocol={'format':'scorer_calibration_protocol_v1','policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,
        'training_protocol_sha256':sha(TRAIN/'protocol.json'),'training_result_sha256':training_results,
        'candidates':candidates,'margin_grid':GRID,'screen_episodes':64,'confirmation_episodes':224,
        'screen_selection_sha256':sha(RUN/'selection_screen.json'),
        'remaining_selection_sha256':sha(RUN/'selection_remaining.json'),'full_selection_sha256':sha(RUN/'selection_full.json'),
        'selection_rule':'Within each variant: highest C64 macro SR, then SPL, then nDTW (average equally over DINO pair for pooled); then lower branch regret, larger margin, earlier checkpoint',
        'shared_scorer_and_margin_across_DINO_models':True,
        'confirmation_rule':'Run the remaining 160 episodes for each selected variant/model and base; combine with C64. If SR is lower than base, or tied SR has lower SPL, validate margin 1.0 on all C224 before freezing that conservative fallback.',
        'fallback_rule':'Margin 1.0 must make zero overrides and reproduce base outcomes on C224; if it does not, validate finite margin 1000000.0 on C224. Failure to reproduce base then is a runtime error, not a valid calibration result.',
        'minimum_real_observations':4,'sampling_seed':0,'world_model_samples_per_movement':1,'workers':4,
        'step10_authorization':'Proceed automatically after settings freeze; E metrics cannot alter these choices',
        'step10_plan':{'group':'E','split':'val_unseen','episodes_per_dataset':112,'selection_seed':10,
            'systems':['policy','null','actual','dinowm_transfer','dinov2_fresh_dynamics'],
            'scope':'Scene-balanced 224-episode pilot on official unseen scenes, not the full 5508-episode benchmark',
            'actual_control':'Same pooled scorer and margin, privileged measured first-next views'},
        'scope':'Initial SmolVLM2 DINO pair; separate null control; held-out C is development only',
        'runtime_check_sha256':sha(RUN/'check_runtime.json'),'simulator_check_sha256':sha(RUN/'check_simulator.json'),
        'null_runtime_check_sha256':sha(RUN/'check_null_runtime.json')}
    sources=source_hashes()
    for p in list((ROOT/'experiment_flow').glob('*.py'))+list((ROOT/'tools').glob('*step9*.py'))+list((ROOT/'tools').glob('check_navigation*.py')):
        sources[str(p.relative_to(ROOT))]=sha(p)
    protocol['source_sha256']=sources
    if (RUN/'protocol.json').exists():assert read(RUN/'protocol.json')==protocol
    else:
        for name in sources:
            dest=RUN/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        write(RUN/'protocol.json',protocol)
    print('Locked',sha(RUN/'protocol.json'),len(candidates),'candidate/margin pairs')

if __name__=='__main__':main()
