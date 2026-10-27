"""Verify actual trained-policy taps, instruction spans, encoder and frozen predictions."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy.common import past_frame_indices
from policy.online import OnlineInputs
from Scorer.interfaces import last_valid_state, pool_dino_patches
from scorer_features.common import (ENCODER_ID,FROZEN,LABELS,POLICY_ID,RUN,TRIALS,
    native_flags,now,read,sha,verify_index,write)
from scorer_features.policy import FrozenPolicyFeatures,instruction_batch
from wm.dino import preprocessing
from wm.frozen import FrozenDinoWorldModel


def main():
    started=time.time();native_flags();verify_index()
    episodes=read(RUN/'index/episodes.json');roots=read(RUN/'index/roots.json')
    extractor=FrozenPolicyFeatures();policy=extractor.policy
    token_records=[]
    for e in episodes:
        _,_,info=instruction_batch(policy.inputs,e['instruction']['text'],'cpu')
        token_records.append({'episode':e['key'],'language':e['instruction']['language'],
            'instruction_tokens':info['instruction_tokens'],'prompt_tokens':info['prompt_tokens'],'truncated':False})
    selected=[]
    for g in ('B','C'):
        for d in ('r2r','rxr'):
            for m in ('expert','policy'):
                selected.append(next(e for e in episodes if (e['group'],e['dataset'],e['prefix_source'])==(g,d,m)))
    longest=max(episodes,key=lambda e:len(e['instruction']['text']))
    if longest not in selected:selected.append(longest)
    records=[];visual_episode=selected[0];visual_images=[]
    for e in selected:
        instruction=extractor.instruction(e['instruction']['text'],compare_hidden=True)
        online=OnlineInputs(policy.inputs,policy.device);frames=[]
        eroots=[r for r in roots if r['episode_index']==e['id']]
        chosen={eroots[0]['t']:eroots[0],eroots[-1]['t']:eroots[-1]}
        for t in range(max(chosen)+1):
            p=LABELS/Path(e['carrier']).parent/f'rgb/{t:04d}.png'
            with Image.open(p) as im:image=im.copy()
            frames.append(image);online.append(image)
            if t not in chosen:continue
            row=chosen[t];actions=e['actions_live'][:t]
            batch=online(e['instruction']['text'],actions)
            tapped=extractor.context(batch,compare_hidden=True)
            assert np.array_equal(tapped['raw_logits'].numpy(),row['base_raw_logits']),('Stored native logits differ',row['root_id'])
            assert tapped['prompt_length']==row['prompt_length']
            adjusted=torch.tensor(row['base_raw_logits']);adjusted[0]-=.25
            assert tapped['base_live']==int(adjusted.argmax())
            native=policy.inputs([{'instruction':e['instruction']['text'],'executed_actions':actions,'t':t,
                'images':[frames[i] for i in past_frame_indices(t,policy.cfg)+[t]]}],with_targets=False)
            assert set(native)==set(batch)
            for name in batch:assert torch.equal(batch[name].cpu(),native[name]),('Native processor drift',name)
            direct=policy.raw_action_scores({k:v.to(policy.device) for k,v in native.items()})
            assert torch.equal(direct[0].cpu(),tapped['raw_logits'])
            # A synthetic invalid padding tail must not replace the actual final state.
            hidden=extractor.tap.hidden
            padded=torch.cat((hidden,torch.zeros_like(hidden[:,:2])),dim=1)
            mask=torch.cat((batch['attention_mask'].bool(),torch.zeros(1,2,dtype=torch.bool,device=policy.device)),dim=1)
            assert torch.equal(last_valid_state(padded,mask)[0].cpu(),tapped['context'])
            records.append({'root':row['root_id'],'prompt_tokens':tapped['prompt_length'],
                'hook_equals_native_hidden':True,'saved_logits_exact':True,'native_processor_exact':True})
        if e is visual_episode:visual_images=frames[:4]
    assert not any(p.requires_grad for p in policy.model.parameters())
    hook_path=extractor.hook_path
    del extractor,policy,online,batch,native,direct,hidden,padded,mask
    torch.cuda.empty_cache()
    models={name:FrozenDinoWorldModel(FROZEN/name,expected_policy_identity=POLICY_ID,
                                    load_encoder=i==0) for i,name in enumerate(TRIALS)}
    first=models[TRIALS[0]]
    with torch.inference_mode():
        pixels=torch.stack([preprocessing()(im) for im in visual_images]).cuda()
        small=torch.cat([first.encoder.encoder.forward_features(x[None])['x_norm_patchtokens'] for x in pixels])
        full=first.encoder.encoder.forward_features(pixels.repeat(16,1,1,1))['x_norm_patchtokens'][:4]
        batch_mse=(small-full).square().mean().item()
        # Production uses one-image native forwards, matching online encoding.
        # Large batches can select TF32 cuDNN patch convolution (diagnosed separately).
        repeat=torch.cat([first.encoder.encoder.forward_features(x[None])['x_norm_patchtokens'] for x in pixels])
        assert torch.equal(small,repeat)
        raw=small.to(torch.float16)
        assert torch.equal(raw,torch.cat([first.encode_images([im]) for im in visual_images]))
        pooled=pool_dino_patches(raw.float())
        manual=raw.float().reshape(4,7,2,7,2,384).mean((2,4)).reshape(4,49,384)
        torch.testing.assert_close(pooled,manual,atol=2e-6,rtol=1e-6)
        predictions={}
        for name,model in models.items():
            x=raw[1:4][None];past=torch.tensor([[0,1]],device='cuda')
            values=[]
            for a in range(3):
                candidate=torch.tensor([a],device='cuda')
                y=model.predict(x,past,candidate,observations_seen=4,encoder_identity=ENCODER_ID)
                again=model.predict(x,past,candidate,observations_seen=4,encoder_identity=ENCODER_ID)
                assert y.shape==(1,196,384) and torch.isfinite(y).all() and torch.equal(y,again)
                values.append(pool_dino_patches(y.float()))
            assert max((values[a]-values[b]).square().mean().item() for a in range(3) for b in range(a))>0
            try:model.predict(x,past,candidate,observations_seen=3,encoder_identity=ENCODER_ID)
            except AssertionError:pass
            else:raise AssertionError('Insufficient context accepted')
            assert not any(p.requires_grad for p in model.model.parameters())
            predictions[name]={'strict_frozen_load':True,'three_commands':True,'repeat_exact':True,
                'rejects_insufficient_context':True,'pooled_shape':list(values[0].shape)}
    source=['scorer_features/common.py','scorer_features/policy.py','tools/check_step7_runtime.py',
            'Scorer/interfaces.py','policy/online.py','policy/experimental_reference.py','wm/dino.py','wm/frozen.py']
    report={'status':'passed','completed_utc':now(),'seconds':time.time()-started,
        'index_manifest_sha256':sha(RUN/'index/manifest.json'),'policy_hook':hook_path,
        'instruction_records':token_records,'instruction_token_range':[min(r['instruction_tokens'] for r in token_records),max(r['instruction_tokens'] for r in token_records)],
        'instruction_cases_with_native_hidden_parity':len(selected),'context_cases':records,
        'frozen_policy_parameters':True,'native_expanded_mask_verified':True,
        'DINO_unused_batch64_mse_vs_single_image':batch_mse,'DINO_production_batch_size':1,
        'DINO_single_image_repeat_exact':True,'DINO_FP16_storage_and_float32_pooling_verified':True,
        'world_models':predictions,'source_sha256':{f:sha(ROOT/f) for f in source}}
    write(RUN/'check_runtime.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('instruction_records','source_sha256')},indent=2),flush=True)

if __name__=='__main__':main()
