"""Measured branch diagnostics; these are not navigation success rates."""
import numpy as np
import torch
from Scorer.scorer import choose_action
from scorer_training.replay import replay
from .common import TRIALS


def summarize(data,indices,outputs):
    ids=torch.tensor(indices,device=data.device)
    utility=torch.tensor(outputs['utility'],device=data.device)
    labels=data.labels['utility'][ids];base=data.base[ids]
    actions=torch.arange(4,device=data.device)[None].expand(len(ids),-1)
    selected=choose_action(utility,actions,base,0.).actions
    best=labels.max(1).values;chosen=labels.gather(1,selected[:,None])[:,0]
    base_value=labels.gather(1,base[:,None])[:,0]
    regret=best-chosen
    success_logits=torch.tensor(outputs['success_logit'],device=data.device)
    collision_logits=torch.tensor(outputs['collision_logit'],device=data.device)
    success=data.labels['success'][ids];collisions=data.labels['immediate_collision'][ids]
    sorted_values=labels.sort(1,descending=True).values
    clear=(sorted_values[:,0]-sorted_values[:,1])>1e-3
    summary={'roots':len(indices),'mean_regret':float(regret.mean()),'base_mean_regret':float((best-base_value).mean()),
        'mean_selected_utility':float(chosen.mean()),'mean_base_utility':float(base_value.mean()),
        'mean_oracle_utility':float(best.mean()),'utility_mse':float(((utility-labels)**2).mean()),
        'success_bce':float(torch.nn.functional.binary_cross_entropy_with_logits(success_logits,success)),
        'forward_collision_bce':float(torch.nn.functional.binary_cross_entropy_with_logits(collision_logits[:,0],collisions[:,0])),
        'success_brier':float(((success_logits.sigmoid()-success)**2).mean()),
        'forward_collision_brier':float(((collision_logits[:,0].sigmoid()-collisions[:,0])**2).mean()),
        'ranking_accuracy_gap_gt_1e_minus_3':float((regret[clear]<=1e-6).float().mean()) if clear.any() else None,
        'clear_gap_roots':int(clear.sum()),'override_fraction_at_zero_margin':float((selected!=base).float().mean()),
        'selected_branch_success_fraction':float(success.gather(1,selected[:,None]).mean()),
        'base_branch_success_fraction':float(success.gather(1,base[:,None]).mean())}
    rows=[{'root_id':data.roots[i]['root_id'],'episode_index':data.roots[i]['episode_index'],
        'dataset':data.roots[i]['dataset'],'prefix_source':data.roots[i]['prefix_source'],
        'scene':data.roots[i]['scene'],'t':data.roots[i]['t'],'regret':float(regret[j]),
        'selected_action':int(selected[j]),'base_action':int(base[j]),
        **{k:outputs[k][j] for k in outputs}} for j,i in enumerate(indices)]
    strata={}
    for name in ('r2r','rxr'):
        for mode in ('expert','policy'):
            part=[r['regret'] for r in rows if r['dataset']==name and r['prefix_source']==mode]
            strata[f'{name}/{mode}']={'roots':len(part),'mean_regret':float(np.mean(part))}
    summary['strata']=strata
    return summary,rows


@torch.no_grad()
def evaluate(model,data,group='C',variant='pooled'):
    previous=model.training;model.eval();indices=data.groups[group]
    names=['null'] if variant=='null' else ['actual']+list(TRIALS)
    outputs={name:{k:[] for k in ('utility','success_logit','collision_logit')} for name in names}
    for start in range(0,len(indices),8):
        part=indices[start:start+8];context=replay(model,data,part)
        targets=data.targets(part)
        for name in names:
            if name=='null':future=context.current_patches[:,None].expand(-1,4,-1,-1)
            else:future=data.futures(part,[-1 if name=='actual' else TRIALS.index(name)]*len(part))
            values=model.score(context,future,targets['actions'])
            for k,v in values.items():
                assert torch.isfinite(v).all()
                outputs[name][k]+=v.cpu().tolist()
    summaries={};rows={}
    for name,values in outputs.items():summaries[name],rows[name]=summarize(data,indices,values)
    selection=float(np.mean([summaries[n]['mean_regret'] for n in (['null'] if variant=='null' else TRIALS)]))
    model.train(previous)
    return {'group':group,'variant':variant,'selection_mean_regret':selection,'representations':summaries,
            'scope':'Measured mixed-root branch outcomes; not closed-loop navigation performance','rows':rows}
