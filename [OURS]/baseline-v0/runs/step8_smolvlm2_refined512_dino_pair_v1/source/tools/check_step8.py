"""Real-input replay/gradient equivalence, null-future invariance and GPU timing."""
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from Scorer.scorer import NavigationScorer
from scorer_features.dataset import FrozenScorerDataset
from scorer_training.common import CACHE,RUN,flags,now,sha,verify_inputs,write
from scorer_training.data import ScorerData
from scorer_training.replay import NullFutureScorer,replay


def main():
    started=time.time();runtime=flags();verify_inputs();torch.manual_seed(8)
    data=ScorerData();reference=FrozenScorerDataset('B')
    chosen=[min(data.groups['B'],key=lambda i:abs(data.root_lengths[i]-n)) for n in (4,35,53)]
    model=NavigationScorer.from_config(CACHE/'scorer_config.json','smolvlm2').cuda().train()
    native=copy.deepcopy(model)
    context=replay(model,data,chosen,training=True)
    futures=data.futures(chosen,[-1]*len(chosen));targets=data.targets(chosen)
    output=model.score(context,futures,targets['actions'])
    losses=model.loss(output,**{k:targets[k] for k in ('utility','success','immediate_collision','actions')})
    losses['total'].backward()
    expected=[];memories=[]
    for i in chosen:
        prefix=reference.prefix(i);memory=None;n=len(prefix['policy_context'])
        for t in range(n):
            if t==max(0,n-32) and memory is not None:memory=memory.detach()
            with torch.set_grad_enabled(t>=max(0,n-32)):
                c=native.observe(**reference.observation(prefix,t),memory=memory);memory=c.memory
        memories.append(memory.detach())
        tar=data.targets([i]);out=native.score(c,data.futures([i],[-1]),tar['actions']);expected.append(out)
        loss=native.loss(out,**{k:tar[k] for k in ('utility','success','immediate_collision','actions')})
        (loss['total']/len(chosen)).backward()
    torch.testing.assert_close(context.memory,torch.cat(memories),atol=3e-6,rtol=3e-5)
    differences={}
    for k in output:
        joined=torch.cat([o[k] for o in expected]);torch.testing.assert_close(output[k],joined,atol=3e-6,rtol=3e-5)
        differences[k]=float((output[k]-joined).abs().max())
    gradient_error=0.
    for (name,a),(other,b) in zip(model.named_parameters(),native.named_parameters()):
        assert name==other and a.grad is not None and b.grad is not None
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=3e-4,msg=lambda m:name+': '+m)
        gradient_error=max(gradient_error,float((a.grad-b.grad).abs().max()))
    assert not any(v.requires_grad for v in vars(data).values() if isinstance(v,torch.Tensor))
    null=NullFutureScorer.from_config(CACHE/'scorer_config.json','smolvlm2').cuda()
    null.load_state_dict(model.state_dict());null.zero_grad(set_to_none=True)
    nc=replay(null,data,chosen,training=True)
    no=null.score(nc,futures,targets['actions'])
    alt=null.score(nc,torch.randn_like(futures),targets['actions'])
    for k in no:assert torch.equal(no[k],alt[k]),k
    null.loss(no,**{k:targets[k] for k in ('utility','success','immediate_collision','actions')})['total'].backward()
    assert null.type_embedding.weight.grad[4].abs().sum()>0
    assert null.visual_projection[1].bias.grad.abs().sum()>0
    assert sum(x.numel() for x in null.parameters())==sum(x.numel() for x in model.parameters())==2890115
    del native,null,context,expected,no,alt,nc,output,losses;torch.cuda.empty_cache()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    gen=torch.Generator().manual_seed(8);order=torch.randperm(1024,generator=gen).tolist()
    torch.cuda.reset_peak_memory_stats();timings=[];loss_values=[]
    for step in range(10):
        indices=order[step*8:(step+1)*8]
        torch.cuda.synchronize();begin=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        c=replay(model,data,indices,training=True)
        tar=data.targets(indices);future=data.futures(indices,[step%2]*8)
        out=model.score(c,future,tar['actions'])
        loss=model.loss(out,**{k:tar[k] for k in ('utility','success','immediate_collision','actions')})
        loss['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin;timings.append(elapsed);loss_values.append(float(loss['total']))
        print(json.dumps({'pilot_step':step+1,'seconds':elapsed,'loss':loss_values[-1],'gradient_norm':float(norm)}),flush=True)
    report={'status':'passed','completed_utc':now(),'seconds':time.time()-started,'runtime':runtime,
        'native_replay_roots':[data.roots[i]['root_id'] for i in chosen],
        'native_replay_lengths':[data.root_lengths[i] for i in chosen],
        'output_max_abs_errors':differences,'all_parameter_gradients_match_truncated_native_replay':True,
        'max_parameter_gradient_abs_error':gradient_error,'inactive_rows_preserve_memory':True,
        'external_inputs_frozen':True,'null_future_independent_of_supplied_views':True,
        'null_future_embedding_receives_gradients':True,'parameters_per_model':2890115,
        'training_step_seconds':timings,'pilot_losses':loss_values,
        'steady_step_seconds':sum(timings[2:])/len(timings[2:]),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'feature_manifest_sha256':sha(CACHE/'manifest.json'),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scorer_training').glob('*.py'))}}
    write(RUN/'check_training.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
