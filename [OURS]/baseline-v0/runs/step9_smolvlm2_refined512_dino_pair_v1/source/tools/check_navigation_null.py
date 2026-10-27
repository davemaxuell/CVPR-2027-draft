import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from navigation_eval.controller import NavigationController
from scorer_features.cache import checked_image
from scorer_features.common import LABELS,RUN as FEATURES_RUN,now,read,sha,write
from scorer_features.dataset import FrozenScorerDataset

RUN=ROOT/'runs/step9_smolvlm2_refined512_dino_pair_v1'

def main():
    checkpoint=ROOT/'runs/step8_smolvlm2_refined512_dino_pair_v1/null/checkpoints/step_000128.pt'
    ctrl=NavigationController('null',str(checkpoint),margin=0.)
    assert ctrl.dynamics is None
    data=FrozenScorerDataset('C');images=read(FEATURES_RUN/'index/images.json');checks=[]
    for name in ('r2r','rxr'):
        index=next(i for i,r in enumerate(data.roots) if r['dataset']==name and r['prefix_source']=='policy')
        r=data.roots[index];e=data.episodes[r['episode_index']];prefix=data.prefix(index)
        ctrl.reset(e['instruction']['text'],e['budget']);memory=None
        for t in range(r['t']+1):
            item=images[e['frame_ids'][t]];image=checked_image(LABELS/item['path'],item['sha256'],item['raw_rgb_sha256'])
            native=ctrl.observe(image,e['actions_live'][:t])
            assert torch.equal(native['context'],prefix['policy_context'][t])
            with torch.inference_mode():
                expected=ctrl.scorer.observe(**data.observation(prefix,t),memory=memory);memory=expected.memory
            assert torch.equal(ctrl.memory,memory)
            action,event=ctrl.decide();assert event['world_model_calls']==0
            if t<r['t']:ctrl.commit(e['actions_live'][t])
        with torch.inference_mode():
            actions=torch.arange(4,device='cuda')[None]
            a=ctrl.scorer.score(ctrl.context,data.futures(index,'actual'),actions)
            b=ctrl.scorer.score(ctrl.context,torch.randn(1,4,49,384,device='cuda'),actions)
            for k in a:assert torch.equal(a[k],b[k])
        checks.append({'root':r['root_id'],'native_memory_exact':True,'future_independent':True})
    sources=[ROOT/'navigation_eval/controller.py',ROOT/'scorer_training/replay.py',Path(__file__).resolve()]
    write(RUN/'check_null_runtime.json',{'status':'passed','completed_utc':now(),'checkpoint':str(checkpoint),
        'checkpoint_sha256':sha(checkpoint),'checks':checks,'world_model_loaded':False,'world_model_calls':0,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}})

if __name__=='__main__':main()
