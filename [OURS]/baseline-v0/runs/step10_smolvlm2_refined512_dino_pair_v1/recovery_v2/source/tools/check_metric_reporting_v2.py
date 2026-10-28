"""Check all saved finite records and the exact reproduced unreachable episode."""
import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from navigation_eval.metrics import aggregate as original
from navigation_metrics_v2.metrics import aggregate,normalize_episode
from scorer_features.common import now,read,sha,write


def main():
    run=ROOT/'runs/step10_smolvlm2_refined512_dino_pair_v1'
    paths=sorted((run/'evaluation/policy/episodes').glob('*.json'));rows=[read(p) for p in paths]
    assert len(rows)==184
    converted=[normalize_episode(r) for r in rows]
    for a,b in zip(rows,converted):
        for key,value in a.items():assert b[key]==value,(a['episode_id'],key)
    before=original(rows,bootstrap=True);after=aggregate(converted,bootstrap=True)
    def contains_equal(a,b):
        if isinstance(a,dict):
            for key,value in a.items():contains_equal(value,b[key])
        else:assert a==b,(a,b)
    contains_equal(before,after)
    path=run/'diagnostics/nonfinite_rxr5734/episodes/rxr_5734.json'
    bad=read(path);bad['spl']=float('nan');bad['distance_to_goal']=float('inf')
    bad['distances_to_goal']=[float('inf') if v=='inf' else v for v in bad['distances_to_goal']]
    fixed=normalize_episode(bad)
    assert fixed['success']==fixed['spl']==0 and fixed['distance_to_goal'] is None
    assert fixed['initial_goal_unreachable'] and fixed['terminal_goal_unreachable']
    for key in ('executed_actions','positions','events','collided_after_action','rgb_sha256','ndtw'):
        assert fixed[key]==bad[key],key
    result=aggregate(converted+[fixed],bootstrap=True)
    assert result['overall']['episodes']==185 and result['overall']['terminal_unreachable_episodes']==1
    assert result['overall']['distance_to_goal'] is None and result['overall']['native_nan_SPL_failures']==1
    assert result['overall']['successes']==before['overall']['successes']
    # Other nonfinite metrics remain fatal; do not silently normalize arbitrary errors.
    for key in ('success','ndtw','sdtw','distance_to_goal'):
        value=copy.deepcopy(rows[0]);value[key]=float('nan')
        try:normalize_episode(value)
        except AssertionError:pass
        else:raise AssertionError(key)
    value=copy.deepcopy(rows[0]);value['spl']=float('nan')
    try:normalize_episode(value)
    except AssertionError:pass
    else:raise AssertionError('Unexpected NaN SPL accepted')
    write(run/'recovery_v2/check_metrics.json',{'status':'passed','completed_utc':now(),
        'finite_episodes_unchanged':len(rows),'all_original_finite_aggregate_and_bootstrap_values_exact':True,
        'unreachable_failure_retained_with_zero_SPL_and_infinite_NE':True,'actions_and_native_scores_unchanged':True,
        'unrelated_nonfinite_metrics_rejected':True,'diagnostic_source_sha256':sha(path),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'navigation_metrics_v2/metrics.py',Path(__file__).resolve()]}})
    print('Metric recovery checks passed for 184 finite records and reproduced RxR5734',flush=True)


if __name__=='__main__':main()
