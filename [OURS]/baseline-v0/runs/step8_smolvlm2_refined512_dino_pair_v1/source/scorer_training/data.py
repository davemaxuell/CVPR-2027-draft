"""Keep only immutable external features resident; never store scorer memory."""
import numpy as np
import torch
from scorer_features.common import CACHE, RUN as FEATURES_RUN, TRIALS, read


class ScorerData:
    def __init__(self,device='cuda:0'):
        self.device=torch.device(device)
        self.episodes=read(FEATURES_RUN/'index/episodes.json')
        self.roots=read(FEATURES_RUN/'index/roots.json')
        values=[torch.load(CACHE/'policy'/f"{e['key']}.pt",map_location='cpu',weights_only=True) for e in self.episodes]
        assert all(not v['trainable_memory_cached'] for v in values)
        self.pooled=torch.from_numpy(np.load(CACHE/'vision/pooled.npy')).to(self.device)
        self.predictions=torch.stack([torch.from_numpy(np.load(CACHE/'predictions'/trial/'future.npy')) for trial in TRIALS]).to(self.device)
        self.offsets=[];offset=0
        for v in values:self.offsets.append(offset);offset+=len(v['context'])
        self.offsets=torch.tensor(self.offsets,device=self.device)
        self.context=torch.cat([v['context'] for v in values]).float().to(self.device)
        self.frames=torch.cat([v['frame_ids'] for v in values]).to(self.device)
        self.previous=torch.cat([v['previous_action_canonical'] for v in values]).to(self.device)
        self.elapsed=torch.cat([v['elapsed_fraction'] for v in values]).to(self.device)
        self.instruction_lengths=[len(v['instruction']) for v in values]
        self.instruction=torch.zeros(len(values),max(self.instruction_lengths),960,device=self.device)
        for i,v in enumerate(values):self.instruction[i,:len(v['instruction'])]=v['instruction'].to(self.device).float()
        self.episode_ids=torch.tensor([r['episode_index'] for r in self.roots],device=self.device)
        self.lengths=torch.tensor([r['t']+1 for r in self.roots],device=self.device)
        self.root_lengths=[r['t']+1 for r in self.roots]
        self.root_instruction_lengths=[self.instruction_lengths[r['episode_index']] for r in self.roots]
        self.actual_ids=torch.tensor([r['actual_next_image_ids'] for r in self.roots],device=self.device)
        self.labels={k:torch.tensor([[float(r['candidates_canonical'][str(a)][field]) for a in range(4)] for r in self.roots],device=self.device)
                     for k,field in [('utility','utility'),('success','success'),('immediate_collision','immediate_forward_collision')]}
        self.base=torch.tensor([r['base_action_canonical'] for r in self.roots],device=self.device)
        self.groups={g:[r['id'] for r in self.roots if r['group']==g] for g in ('B','C')}
        assert len(self.groups['B'])==1024 and len(self.groups['C'])==256
        assert not any(v.requires_grad for v in vars(self).values() if isinstance(v,torch.Tensor))

    def batch(self,indices):
        ids=torch.tensor(indices,device=self.device,dtype=torch.long)
        es=self.episode_ids[ids];length=max(self.root_instruction_lengths[i] for i in indices)
        il=torch.tensor([self.root_instruction_lengths[i] for i in indices],device=self.device)
        return {'ids':ids,'episodes':es,'lengths':self.lengths[ids],
                'lengths_cpu':[self.root_lengths[i] for i in indices],
                'instruction':self.instruction[es,:length],
                'instruction_valid':torch.arange(length,device=self.device)[None]<il[:,None]}

    def phase(self,batch,starts,counts):
        n=max(counts)
        if n==0:return None
        start=torch.tensor(starts,device=self.device)
        count=torch.tensor(counts,device=self.device)
        steps=torch.arange(n,device=self.device)[:,None]
        local=torch.minimum(start[None]+steps,batch['lengths'][None]-1)
        flat=self.offsets[batch['episodes']][None]+local
        return {'current':self.pooled[self.frames[flat]],'context':self.context[flat],
                'previous':self.previous[flat],'elapsed':self.elapsed[flat][...,None],
                'active':steps<count[None]}

    def futures(self,indices,representations):
        # -1 means observed first-next features; 0/1 are the frozen DINO pair.
        ids=torch.as_tensor(indices,device=self.device,dtype=torch.long)
        reps=torch.as_tensor(representations,device=self.device,dtype=torch.long)
        actual=self.pooled[self.actual_ids[ids]]
        predicted=self.predictions[reps.clamp_min(0),ids]
        return torch.where((reps<0)[:,None,None,None],actual,predicted)

    def targets(self,indices):
        ids=torch.as_tensor(indices,device=self.device,dtype=torch.long)
        return {**{k:v[ids] for k,v in self.labels.items()},'actions':torch.arange(4,device=self.device)[None].expand(len(ids),-1)}
