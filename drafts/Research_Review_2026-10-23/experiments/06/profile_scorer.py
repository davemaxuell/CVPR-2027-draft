from pathlib import Path
import json, sys, time
import psutil, torch
sys.path.insert(0,str(Path(__file__).parent/'scorer'))
from scorer import NavigationScorer, scorer_loss, START, select_action, pool_dino_patches

torch.set_num_threads(2); torch.manual_seed(606)
b,l,t,d,k=8,256,32,960,4
proc=psutil.Process(); start_rss=proc.memory_info().rss
model=NavigationScorer(d)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
text=torch.randn(b,l,d); valid=torch.ones(b,l,dtype=torch.bool)
view=torch.randn(b,64,384); q=torch.randn(b,d)
actions=torch.arange(k).expand(b,-1); previous=torch.full((b,),START,dtype=torch.long)
future=torch.randn(b,k,64,384,requires_grad=True)
labels=torch.zeros(b,k)
start=time.perf_counter(); memory=None
for i in range(t):
    ctx=model.observe(text,valid,view,q,previous,torch.full((b,1),i/500),memory)
    memory=ctx.memory
outs=model.score(ctx,future,actions)
loss=scorer_loss(outs,labels,labels,labels,actions)['total']
loss.backward(); optimizer.step()
train_seconds=time.perf_counter()-start
info=proc.memory_info()
model.eval()
with torch.inference_mode():
    one=model.observe(text[:1],valid[:1],view[:1],q[:1],previous[:1],torch.zeros(1,1))
    for _ in range(3): model.score(one,future[:1],actions[:1])
    ticks=[]
    for _ in range(20):
        ts=time.perf_counter(); model.score(one,future[:1],actions[:1]); ticks.append(time.perf_counter()-ts)
probe={}
try: select_action(torch.tensor([[float('nan'),0.,0.,0.]]),actions[:1],torch.tensor([0]),0.)
except ValueError as e: probe['nonfinite_selection_raises']=str(e)
try: pool_dino_patches(torch.zeros(1,196,384))
except ValueError as e: probe['dinowm_default_196_patches_rejected']=str(e)
probe['future_gradient_is_none']=future.grad is None
out={'scope':'Synthetic CPU scorer only; no GPU/CE/WM/encoder/model weights loaded. One optimizer step at last root after 32 recurrent observations, not 32 supervised roots.', 'python':sys.version,'torch':torch.__version__,'device':'cpu','threads':2,'batch_roots':b,'instruction_tokens':l,'recurrent_observations':t,'policy_width':d,'candidate_count':k,'dtype':'float32','train_step_seconds':train_seconds,'process_initial_rss_bytes':start_rss,'process_rss_after_step_bytes':info.rss,'process_peak_working_set_bytes':getattr(info,'peak_wset',None),'score_only_median_seconds':sorted(ticks)[len(ticks)//2],'model_parameter_count':sum(p.numel() for p in model.parameters()),'contract_probes':probe}
(Path(__file__).parent/'scorer_profile.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
