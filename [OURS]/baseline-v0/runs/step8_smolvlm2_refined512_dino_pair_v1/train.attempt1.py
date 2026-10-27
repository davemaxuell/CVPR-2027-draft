"""Matched fixed-budget scorer and null-future control training."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from Scorer.scorer import NavigationScorer
from .common import CACHE,RUN,flags,now,read,sha,torch_save,verify_protocol,write
from .data import ScorerData
from .evaluate import evaluate
from .replay import NullFutureScorer,replay


def schedule(epoch,protocol):
    stage=next(s for s in protocol['stages'] if s['start_epoch']<=epoch<s['end_epoch'])
    rng=np.random.default_rng(protocol['seed']+epoch*1009)
    order=rng.permutation(1024).tolist()
    if stage['name']=='actual':representations=[-1]*1024
    elif stage['name']=='mixed':representations=[-1]*512+[0]*256+[1]*256
    else:representations=[0]*512+[1]*512
    rng.shuffle(representations)
    assert len(set(order))==1024 and min(order)==0 and max(order)==1023
    return stage['name'],order,representations


def update(model,optimizer,data,indices,representations,variant):
    optimizer.zero_grad(set_to_none=True)
    context=replay(model,data,indices,training=True,bptt=32)
    targets=data.targets(indices)
    future=(context.current_patches[:,None].expand(-1,4,-1,-1) if variant=='null' else data.futures(indices,representations))
    output=model.score(context,future,targets['actions'])
    losses=model.loss(output,**{k:targets[k] for k in ('utility','success','immediate_collision','actions')})
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return {**{k:float(v.detach()) for k,v in losses.items()},'gradient_norm':float(norm)}


def train(variant):
    assert variant in ('pooled','null');flags();p=verify_protocol();signature=sha(RUN/'protocol.json')
    out=RUN/variant;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():assert read(out/'result.json')['status']=='trained';return
    random.seed(p['seed']);np.random.seed(p['seed']);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    data=ScorerData();assert data.groups['B']==list(range(1024))
    cls=NullFutureScorer if variant=='null' else NavigationScorer
    model=cls.from_config(CACHE/'scorer_config.json','smolvlm2').cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['optimizer']['learning_rate'],weight_decay=p['optimizer']['weight_decay'])
    epoch0=0;step=0;started=time.time();checkpoints=[]
    if (out/'latest.json').exists():
        latest=read(out/'latest.json');cp=out/latest['file'];assert sha(cp)==latest['sha256']
        state=torch.load(cp,map_location='cuda',weights_only=True)
        assert state['protocol_sha256']==signature and state['variant']==variant
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        epoch0=state['epochs_completed'];step=state['step'];checkpoints=state['checkpoints']
        torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    else:
        initial=evaluate(model,data,variant=variant)
        write(out/'initial_C.json',initial)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epoch0,p['epochs']):
        stage,order,reps=schedule(epoch,p);accum=[];epoch_start=time.time()
        for start in range(0,1024,8):
            values=update(model,optimizer,data,order[start:start+8],reps[start:start+8],variant)
            accum.append(values);step+=1
            if step%16==0:
                progress={'status':'training','variant':variant,'stage':stage,'step':step,'total_steps':p['total_updates'],
                    'epoch':epoch+1,'epochs':p['epochs'],'batch':start//8+1,'loss':values,
                    'seconds':time.time()-started,'updated_utc':now()}
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        validation=evaluate(model,data,variant=variant)
        evaluation=out/'validation'/f'step_{step:06d}.json';write(evaluation,validation)
        summary={'step':step,'epochs_completed':epoch+1,'stage':stage,
                 'selection_mean_regret':validation['selection_mean_regret'],
                 'evaluation':str(evaluation.relative_to(out)),'evaluation_sha256':sha(evaluation),
                 'train_mean_losses':{k:float(np.mean([x[k] for x in accum])) for k in accum[0]},
                 'epoch_seconds':time.time()-epoch_start}
        checkpoints.append(summary)
        cp=out/'checkpoints'/f'step_{step:06d}.pt'
        state={'format':'trained_navigation_scorer_v1','protocol_sha256':signature,'variant':variant,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'epochs_completed':epoch+1,
            'config':model.settings,'checkpoints':checkpoints,'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'stage':stage,'scorer_parameter_count':sum(x.numel() for x in model.parameters())}
        torch_save(cp,state);digest=sha(cp)
        write(cp.with_suffix('.json'),{**summary,'file':str(cp),'sha256':digest,'protocol_sha256':signature})
        write(out/'latest.json',{'file':str(cp.relative_to(out)),'sha256':digest,'step':step})
        write(out/'history.json',checkpoints);print(json.dumps({'validation':summary}),flush=True)
    candidates=sorted([c for c in checkpoints if c['stage']=='predicted'],key=lambda c:(c['selection_mean_regret'],c['step']))
    assert step==p['total_updates'] and candidates
    # Re-evaluate the selected branch candidate on B for fitting diagnostics only.
    best=out/'checkpoints'/f"step_{candidates[0]['step']:06d}.pt"
    model.load_state_dict(torch.load(best,map_location='cuda',weights_only=True)['model'])
    fit=evaluate(model,data,'B',variant);write(out/'selected_candidate_B.json',fit)
    result={'status':'trained','completed_utc':now(),'variant':variant,'updates':step,'epochs':p['epochs'],
        'protocol_sha256':signature,'branch_selected_candidate':str(best),'candidate_sha256':sha(best),
        'selection_mean_regret':candidates[0]['selection_mean_regret'],
        'closed_loop_shortlist':candidates[:2],'all_epoch_metrics':checkpoints,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'seconds_this_run':time.time()-started,'closed_loop_validation_pending':True,
        'note':'Candidate selection uses C measured branch regret; deployment selection still requires C closed-loop checks.'}
    write(out/'result.json',result);write(out/'progress.json',{'status':'trained','step':step,'completed_utc':now()})
    print(json.dumps(result,indent=2),flush=True)
