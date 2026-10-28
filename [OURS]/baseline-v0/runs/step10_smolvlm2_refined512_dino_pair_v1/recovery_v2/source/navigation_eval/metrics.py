from collections import Counter
import numpy as np


def aggregate(rows,*,bootstrap=False):
    def metrics(part):
        forward=sum(r['forward_attempts'] for r in part)
        return {'episodes':len(part),'scenes':len({r['scene'] for r in part}),
            'successes':sum(int(r['success']) for r in part),
            **{k:float(np.mean([r[k] for r in part])) for k in ('success','spl','distance_to_goal','ndtw','sdtw','steps','collisions','seconds')},
            'forward_collision_rate':sum(r['forward_collisions'] for r in part)/max(1,forward),
            'override_rate':sum(r['overrides'] for r in part)/max(1,sum(r['steps'] for r in part)),
            'reasons':dict(sum((Counter(r['reason_counts']) for r in part),Counter()))}
    by={name:metrics([r for r in rows if r['dataset']==name]) for name in ('r2r','rxr') if any(r['dataset']==name for r in rows)}
    result={'overall':metrics(rows),'by_dataset':by,
        **{'macro_'+k:float(np.mean([m[k] for m in by.values()])) for k in ('success','spl','ndtw','sdtw')},
        'scope':'Closed-loop episode metrics; RxR uses the recorded unified 15-degree action protocol'}
    if bootstrap:
        rng=np.random.default_rng(10);scenes=sorted({r['scene'] for r in rows});samples=[]
        for _ in range(2000):
            picked=rng.choice(scenes,len(scenes),replace=True)
            part=[r for scene in picked for r in rows if r['scene']==scene]
            vals=[np.mean([r['success'] for r in part if r['dataset']==name]) for name in by]
            samples.append(float(np.mean(vals)))
        result['macro_success_scene_bootstrap_95']=np.quantile(samples,[.025,.975]).tolist()
    return result
