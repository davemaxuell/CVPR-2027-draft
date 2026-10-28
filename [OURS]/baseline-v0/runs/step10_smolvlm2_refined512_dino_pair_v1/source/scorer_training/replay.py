"""Batched exact architecture, with projections reused only within one weight version.

Inactive rows retain their memory. No imagined frame ever updates memory. The
no-gradient prefix is rebuilt on every call and discarded after this batch.
"""
import torch
from Scorer.scorer import NavigationScorer, RealContext


class NullFutureScorer(NavigationScorer):
    """Same parameter count. Constant movement inputs learn a shared null vector
    through the existing LayerNorm/linear biases and future-type embedding.
    STOP retains current-view identity, which is already available in context.
    """
    def score(self,context,future_patches,actions,*,future_grad=False):
        if future_grad:raise ValueError('Null future does not expose feature gradients')
        return super().score(context,torch.zeros_like(future_patches),actions)


def phase(model,batch,inputs,memory):
    if inputs is None:return memory,None
    instruction=batch['instruction'].detach().masked_fill(~batch['instruction_valid'][...,None],0)
    text=model.policy_projection(instruction)+model.type_embedding.weight[0]
    current=model.visual_projection(inputs['current'].detach())+model.positions+model.type_embedding.weight[1]
    state=model.policy_projection(inputs['context'].detach())+model.type_embedding.weight[2]
    at=model.action_embedding(inputs['previous'])+model.time_projection(inputs['elapsed'].detach())
    source=torch.cat((text[None].expand(len(current),-1,-1,-1),current,state[:,:,None],at[:,:,None]),dim=2)
    padding=torch.cat((~batch['instruction_valid'],torch.zeros(len(text),model.patch_count+2,dtype=torch.bool,device=text.device)),dim=1)
    if memory is None:memory=text.new_zeros(len(text),model.slots,model.width)
    for t in range(len(source)):
        query=model.memory_norm(memory+model.slot_queries)
        update=model.memory_attention(query,source[t],source[t],key_padding_mask=padding,need_weights=False)[0]
        update=update+model.memory_ff(update)
        proposal=model.memory_gru(update.reshape(-1,model.width),memory.reshape(-1,model.width)).reshape_as(memory)
        memory=torch.where(inputs['active'][t,:,None,None],proposal,memory)
    context=torch.cat((source[-1],memory+model.type_embedding.weight[3]),dim=1)
    context_padding=torch.cat((padding,torch.zeros(len(text),model.slots,dtype=torch.bool,device=text.device)),dim=1)
    return memory,RealContext(memory,context,context_padding,inputs['current'][-1].detach())


def replay(model,data,indices,*,training=False,bptt=32):
    batch=data.batch(indices)
    tail=[min(bptt,n) for n in batch['lengths_cpu']]
    warm=[n-k for n,k in zip(batch['lengths_cpu'],tail)]
    with torch.no_grad():
        memory,_=phase(model,batch,data.phase(batch,[0]*len(indices),warm),None)
    if memory is not None:memory=memory.detach()
    with torch.set_grad_enabled(training):
        _,context=phase(model,batch,data.phase(batch,warm,tail),memory)
    return context
