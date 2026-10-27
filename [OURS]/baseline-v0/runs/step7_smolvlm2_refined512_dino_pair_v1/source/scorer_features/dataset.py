"""Read frozen inputs. Scorer memory is deliberately absent and must be replayed."""
from collections import OrderedDict

import numpy as np
import torch

from .common import CACHE, RUN, TRIALS, read, sha, verify_protocol


class FrozenScorerDataset:
    def __init__(self, group, *, require_complete=True):
        assert group in ('B','C')
        verify_protocol()
        self.protocol_sha256=sha(RUN/'protocol.json')
        if require_complete:
            m=read(CACHE/'manifest.json')
            assert m['status']=='complete' and m['protocol_sha256']==self.protocol_sha256
        self.episodes=read(RUN/'index/episodes.json')
        self.roots=[r for r in read(RUN/'index/roots.json') if r['group']==group]
        self.pooled=np.load(CACHE/'vision/pooled.npy',mmap_mode='r')
        self.predictions={name:np.load(CACHE/'predictions'/name/'future.npy',mmap_mode='r') for name in TRIALS}
        self.loaded=OrderedDict()

    def __len__(self):return len(self.roots)

    def episode(self,index):
        e=self.episodes[index]
        if index not in self.loaded:
            value=torch.load(CACHE/'policy'/f"{e['key']}.pt",map_location='cpu',weights_only=True)
            assert value['protocol_sha256']==self.protocol_sha256 and not value['trainable_memory_cached']
            self.loaded[index]=value
            while len(self.loaded)>4:self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]

    def prefix(self,index):
        r=self.roots[index];e=self.episode(r['episode_index']);n=r['t']+1
        # Only actual observations up to the decision enter the memory replay.
        return {'instruction':e['instruction'].unsqueeze(0),'instruction_valid':e['instruction_valid'].unsqueeze(0),
                'current_patches':torch.from_numpy(np.array(self.pooled[e['frame_ids'][:n].numpy()],copy=True)),
                'policy_context':e['context'][:n], 'previous_action':e['previous_action_canonical'][:n],
                'elapsed_fraction':e['elapsed_fraction'][:n,None]}

    def futures(self,index,representation):
        r=self.roots[index]
        if representation=='actual':
            values=np.array(self.pooled[r['actual_next_image_ids']],copy=True)
            values[3]=self.pooled[r['current_image_id']]
        else:
            assert representation in TRIALS
            values=np.array(self.predictions[representation][r['id']],copy=True)
        return torch.from_numpy(values).unsqueeze(0)

    def targets(self,index):
        r=self.roots[index];c=r['candidates_canonical']
        return {'actions':torch.arange(4,dtype=torch.long)[None],
                'utility':torch.tensor([[c[str(a)]['utility'] for a in range(4)]],dtype=torch.float32),
                'success':torch.tensor([[c[str(a)]['success'] for a in range(4)]],dtype=torch.float32),
                'immediate_collision':torch.tensor([[float(c[str(a)]['immediate_forward_collision']) for a in range(4)]],dtype=torch.float32)}

    @staticmethod
    def observation(prefix,t):
        return {'instruction':prefix['instruction'],'instruction_valid':prefix['instruction_valid'],
                'current_patches':prefix['current_patches'][t:t+1],
                'policy_context':prefix['policy_context'][t:t+1],
                'previous_action':prefix['previous_action'][t:t+1],
                'elapsed_fraction':prefix['elapsed_fraction'][t:t+1]}
