"""Calibrate only on C under the four-model recipe declared before E results."""
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from navigation_four.common import BASE_CACHE as CACHE, CALIBRATION as RUN, PRIOR_CALIBRATION, TRAIN, TRIALS, POLICY_ID, read, sha, write
from navigation_four.jobs import source_hashes
from scorer_four.training_common import verify_protocol as verify_training

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
    verify_training(); recipe=read(RUN/'recipe.json')
    assert recipe['no_E_metrics_used_for_recipe'] and recipe['no_E_tuning_allowed']
    assert recipe['margin_grid']==GRID and recipe['trials']==list(TRIALS)
    assert sha(TRAIN/'recipe.json')==recipe['training_recipe_sha256']
    check=read(RUN/'check_runtime.json');assert check['status']=='passed' and check['no_E_used']
    for name,digest in check['source_sha256'].items():assert sha(ROOT/name)==digest,name
    prior=read(PRIOR_CALIBRATION/'result.json')
    assert prior['status']=='complete' and prior['base_C224_actions_reproduced_exactly']
    assert read(TRAIN/'training_complete.json')['status']=='trained'
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
    for label in ('screen','remaining','full'):
        assert sha(RUN/f'selection_{label}.json')==recipe['selection_sha256'][label]
    sources=source_hashes()
    for path in (ROOT/'tools').glob('*four*calibration*.py'):sources[str(path.relative_to(ROOT))]=sha(path)
    sources['tools/check_four_navigation.py']=sha(ROOT/'tools/check_four_navigation.py')
    protocol={**recipe,'format':'four_WM_C_calibration_protocol_v1','recipe_sha256':sha(RUN/'recipe.json'),
        'training_protocol_sha256':sha(TRAIN/'protocol.json'),'training_result_sha256':training_results,
        'training_complete_sha256':sha(TRAIN/'training_complete.json'),
        'candidates':candidates,'runtime_check_sha256':sha(RUN/'check_runtime.json'),
        'screen_selection_sha256':sha(RUN/'selection_screen.json'),
        'remaining_selection_sha256':sha(RUN/'selection_remaining.json'),
        'full_selection_sha256':sha(RUN/'selection_full.json'),
        'prior_calibration_result_sha256':sha(PRIOR_CALIBRATION/'result.json'),
        'prior_base_confirmation_sha256':sha(PRIOR_CALIBRATION/'confirmation/policy/summary.json'),
        'source_sha256':sources}
    if (RUN/'protocol.json').exists():assert read(RUN/'protocol.json')==protocol
    else:
        for name in sources:
            dest=RUN/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        write(RUN/'protocol.json',protocol)
    print('Locked four-model C calibration',sha(RUN/'protocol.json'),len(candidates),'candidates')

if __name__=='__main__':main()
