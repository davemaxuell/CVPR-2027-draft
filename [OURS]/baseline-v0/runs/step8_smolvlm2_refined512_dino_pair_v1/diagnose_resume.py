import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path('/data/team_a/dave-workspace/RoleVLN/[OURS]/baseline-v0');sys.path.insert(0,str(ROOT))
import torch
from Scorer.scorer import NavigationScorer
from scorer_training.common import CACHE,RUN,flags,now,sha,torch_save,write
from scorer_training.data import ScorerData
from scorer_training.train import schedule,update
from scorer_training.evaluate import evaluate

def main():
    flags();torch.manual_seed(8);data=ScorerData()
    p={'seed':8,'stages':[{'name':'actual','start_epoch':0,'end_epoch':4},
       {'name':'mixed','start_epoch':4,'end_epoch':8},{'name':'predicted','start_epoch':8,'end_epoch':16}]}
    for epoch in range(16):
        name,order,reps=schedule(epoch,p)
        assert sorted(order)==data.groups['B']
        assert [reps.count(i) for i in (-1,0,1)]=={'actual':[1024,0,0],'mixed':[512,256,256],'predicted':[0,512,512]}[name]
        assert schedule(epoch,p)==(name,order,reps)
    ids=sorted(data.groups['B'],key=lambda i:data.root_lengths[i])[:8]
    a=NavigationScorer.from_config(CACHE/'scorer_config.json','smolvlm2').cuda().train()
    opt=torch.optim.AdamW(a.parameters(),lr=1e-4,weight_decay=.01)
    update(a,opt,data,ids,[-1]*8,'pooled')
    path=RUN/'preflight_resume.pt';torch_save(path,{'model':a.state_dict(),'optimizer':opt.state_dict()})
    snapshot={k:v.clone() for k,v in a.state_dict().items()}
    left=update(a,opt,data,ids,[0,1]*4,'pooled')
    b=NavigationScorer.from_config(CACHE/'scorer_config.json','smolvlm2').cuda().train()
    other=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=.01)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    b.load_state_dict(restored['model']);other.load_state_dict(restored['optimizer'])
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in snapshot.items())
    print('step_devices',[(str(x['step'].device),str(y['step'].device)) for x,y in zip(opt.state.values(),other.state.values())][:4],flush=True)
    right=update(b,other,data,ids,[0,1]*4,'pooled')
    assert left==right
    errors=[{'name':k,'parameter_max_abs':float((v-b.state_dict()[k]).abs().max())} for k,v in a.state_dict().items() if not torch.equal(v,b.state_dict()[k])]
    gradients=[{'name':k,'gradient_max_abs':float((v.grad-dict(b.named_parameters())[k].grad).abs().max())} for k,v in a.named_parameters() if not torch.equal(v.grad,dict(b.named_parameters())[k].grad)]
    print(json.dumps({'parameter_errors':errors,'gradient_errors':gradients,'losses_equal':left==right}),flush=True)
    return
    path.unlink();del b,other,restored
    before={k:v.detach().clone() for k,v in a.state_dict().items()}
    torch.cuda.synchronize();start=time.perf_counter();report=evaluate(a,data)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    for key,value in a.state_dict().items():assert torch.equal(value,before[key])
    for name,rows in report['rows'].items():
        assert len(rows)==256 and len({r['root_id'] for r in rows})==256
        assert all(r['root_id'].startswith('C/') for r in rows)
    assert abs(report['representations']['actual']['base_mean_regret']-.1837984419579216)<1e-6
    result={'status':'passed','completed_utc':now(),'schedule_epochs_checked':16,
        'training_roots_group_B_only':True,'exact_equal_model_sampling_per_epoch':True,
        'checkpoint_optimizer_resume_exact':True,'validation_preserves_weights':True,
        'all_C_roots_all_three_representations_evaluated':True,'C_evaluation_seconds':elapsed,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scorer_training').glob('*.py'))}}
    write(RUN/'check_orchestration.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
