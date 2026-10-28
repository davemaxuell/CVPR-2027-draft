"""Retain unreachable episodes and represent infinite navigation error in valid JSON."""
import copy
import math

import numpy as np
from navigation_eval.metrics import aggregate as original_aggregate


def normalize_episode(row):
    result=copy.deepcopy(row);exceptions={}
    assert all(math.isfinite(row[k]) for k in ('success','ndtw','sdtw'))
    assert all(math.isfinite(v) for point in row['positions'] for v in point)
    distances=row['distances_to_goal'];assert distances
    assert all(math.isfinite(v) and v>=0 or v==math.inf for v in distances)
    assert math.isfinite(row['distance_to_goal']) and row['distance_to_goal']>=0 or row['distance_to_goal']==math.inf
    if not math.isfinite(row['spl']):
        # Native SPL computes 0 * (inf/inf) for an unsuccessful episode whose
        # initial geodesic distance is infinite. Preserve it as a zero-success,
        # zero-SPL failure, and retain the original NaN explicitly.
        assert math.isnan(row['spl']) and row['success']==0. and distances[0]==math.inf
        exceptions['spl']='NaN (native 0 * infinity/infinity)';result['spl']=0.
    if row['distance_to_goal']==math.inf:
        exceptions['distance_to_goal']='+Infinity';result['distance_to_goal']=None
    result['distances_to_goal']=[None if v==math.inf else v for v in distances]
    result['infinite_distance_observations']=sum(v==math.inf for v in distances)
    result['initial_goal_unreachable']=distances[0]==math.inf
    result['terminal_goal_unreachable']=row['distance_to_goal']==math.inf
    result['native_metric_exceptions']=exceptions
    result['metric_reporting_version']='unreachable_goal_v2'
    assert math.isfinite(result['spl'])
    return result


def aggregate(rows,*,bootstrap=False):
    # Reuse the original SR/SPL/nDTW, counting, and bootstrap implementation.
    # Temporary distance placeholders are overwritten with explicit extended-
    # real statistics below; an infinite error is never dropped or capped.
    temporary=[{**r,'distance_to_goal':0. if r['distance_to_goal'] is None else r['distance_to_goal']} for r in rows]
    report=original_aggregate(temporary,bootstrap=bootstrap)
    for part,summary in [(rows,report['overall'])]+[([r for r in rows if r['dataset']==name],value) for name,value in report['by_dataset'].items()]:
        finite=[r['distance_to_goal'] for r in part if r['distance_to_goal'] is not None]
        unreachable=len(part)-len(finite)
        summary.update(distance_to_goal=None if unreachable else float(np.mean(finite)),
            distance_to_goal_mean_is_infinite=bool(unreachable),terminal_unreachable_episodes=unreachable,
            initial_unreachable_episodes=sum(r.get('initial_goal_unreachable',False) for r in part),
            finite_distance_to_goal_mean=float(np.mean(finite)) if finite else None,
            finite_distance_to_goal_episodes=len(finite),
            native_nan_SPL_failures=sum('spl' in r.get('native_metric_exceptions',{}) for r in part))
    report['unreachable_metric_convention']='Retain every selected episode; native failed-episode NaN SPL from infinite initial distance is zero, with original exception recorded; any terminal infinite distance makes mean NE infinite (JSON null plus explicit flag).'
    return report


def display_distance(value):
    return '∞' if value is None else f'{value:.2f}'
