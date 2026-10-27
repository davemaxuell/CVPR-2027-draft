"""Live controller parity on saved real prefixes and explicit fallback checks."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navigation_eval.controller import NavigationController
from scorer_features.cache import checked_image
from scorer_features.common import CACHE,LABELS,RUN as FEATURES_RUN,TRIALS,now,read,sha,write
from scorer_features.dataset import FrozenScorerDataset
from Scorer.scorer import choose_action

RUN=ROOT/'runs/step9_smolvlm2_refined512_dino_pair_v1'


def main():
    checkpoint=ROOT/'runs/step8_smolvlm2_refined512_dino_pair_v1/pooled/checkpoints/step_000512.pt'
    roots=read(FEATURES_RUN/'index/roots.json');episodes=read(FEATURES_RUN/'index/episodes.json');images=read(FEATURES_RUN/'index/images.json')
    selected=[next(r for r in roots if (r['group'],r['dataset'],r['prefix_source'])==(g,d,m))
              for g in ('B','C') for d in ('r2r','rxr') for m in ('expert','policy')]
    completed=[]
    for trial in TRIALS:
        ctrl=NavigationController('predicted',str(checkpoint),trial,0.)
        for r in selected:
            e=episodes[r['episode_index']];dataset=FrozenScorerDataset(r['group'])
            index=next(i for i,x in enumerate(dataset.roots) if x['root_id']==r['root_id'])
            prefix=dataset.prefix(index);value=dataset.episode(r['episode_index'])
            ctrl.reset(e['instruction']['text'],e['budget']);reference_memory=None
            for t in range(r['t']+1):
                record=images[e['frame_ids'][t]];image=checked_image(LABELS/record['path'],record['sha256'],record['raw_rgb_sha256'])
                native=ctrl.observe(image,e['actions_live'][:t])
                assert torch.equal(native['context'],value['context'][t])
                assert torch.equal(native['raw_logits'],value['raw_logits_live'][t])
                with torch.inference_mode():
                    reference=ctrl.scorer.observe(**dataset.observation(prefix,t),memory=reference_memory)
                    reference_memory=reference.memory
                assert torch.equal(ctrl.memory,reference_memory)
                if t<r['t']:
                    live,event=ctrl.decide();assert event['reason']=='insufficient_real_context' and event['world_model_calls']==0
                    assert live==native['base_live'];ctrl.commit(e['actions_live'][t])
            ctrl.attempted_predictions=0;future=ctrl.futures()
            assert ctrl.attempted_predictions==3 and torch.equal(future.cpu(),dataset.futures(index,trial))
            with torch.inference_mode():
                actions=torch.arange(4,device='cuda')[None]
                out=ctrl.scorer.score(reference,dataset.futures(index,trial),actions)
                choice=choose_action(out['utility'],actions,torch.tensor([r['base_action_canonical']],device='cuda'),0.)
            live,event=ctrl.decide()
            assert event['selected_canonical']==int(choice.actions[0]) and event['world_model_calls']==3
            for k,v in out.items():assert np.array_equal(event['scores'][k],v[0].cpu().numpy())
            before=ctrl.memory.clone();normal=ctrl.futures
            def broken(*args):raise RuntimeError('Injected prediction failure for fallback check')
            ctrl.futures=broken
            live,event=ctrl.decide();assert live==ctrl.base_live and event['reason']=='prediction_unavailable'
            ctrl.margin=None
            live,event=ctrl.decide();assert live==ctrl.base_live and event['reason']=='disabled' and event['world_model_calls']==0
            ctrl.margin=0.;ctrl.futures=normal
            original=ctrl.scorer.score
            ctrl.scorer.score=lambda *args,**kwargs:{'utility':torch.full((1,4),float('nan'),device='cuda')}
            live,event=ctrl.decide();assert live==ctrl.base_live and event['reason']=='nonfinite_scores'
            ctrl.scorer.score=original
            assert torch.equal(before,ctrl.memory)
            try:ctrl.observe(image,e['actions_live'][:r['t']])
            except AssertionError:pass
            else:raise AssertionError('Duplicate real observation was accepted')
            # Actual-view diagnostics use the identical action labels and current STOP identity.
            ctrl.mode='actual'
            def actual_images():
                result=[]
                for image_id in r['actual_next_image_ids'][:3]:
                    item=images[image_id];result.append(checked_image(LABELS/item['path'],item['sha256'],item['raw_rgb_sha256']))
                return result
            ctrl.attempted_predictions=0
            assert torch.equal(ctrl.futures(actual_images).cpu(),dataset.futures(index,'actual'))
            ctrl.mode='predicted';ctrl.commit(0);assert ctrl.terminated
            completed.append({'trial':trial,'root':r['root_id'],'exact':True})
            print(json.dumps(completed[-1]),flush=True)
        del ctrl;torch.cuda.empty_cache()
    sources=[ROOT/'navigation_eval/controller.py',ROOT/'tools/check_navigation_runtime.py']
    report={'status':'passed','completed_utc':now(),'checkpoint_for_interface_check':str(checkpoint),
        'checkpoint_sha256':sha(checkpoint),'native_policy_memory_prediction_and_score_parity':completed,
        'actual_future_feature_parity':True,'warmup_and_disabled_skip_predictions':True,
        'prediction_failure_and_nonfinite_fallback':True,'candidate_calls_preserve_memory':True,
        'duplicate_observation_rejected':True,'STOP_terminates_lifecycle':True,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    write(RUN/'check_runtime.json',report)

if __name__=='__main__':main()
