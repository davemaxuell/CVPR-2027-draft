"""Audit the complete frozen cache, test real replay inputs, and publish Step 8 handoff."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy.online import OnlineInputs
from policy.experimental_reference import verify_bundle
from Scorer.interfaces import pool_dino_patches
from Scorer.scorer import NavigationScorer
from scorer_features.cache import checked_image
from scorer_features.common import (CACHE,ENCODER_ID,FROZEN,LABELS,LIVE_TO_CANONICAL,POLICY_ID,REFERENCE,
    RUN,TRIALS,native_flags,now,read,sha,verify_protocol,write)
from scorer_features.dataset import FrozenScorerDataset
from scorer_features.policy import FrozenPolicyFeatures,instruction_batch
from wm.frozen import FrozenDinoWorldModel


def main():
    native_flags();p=verify_protocol();signature=sha(RUN/'protocol.json')
    episodes=read(RUN/'index/episodes.json');roots=read(RUN/'index/roots.json');images=read(RUN/'index/images.json')
    files={};totals={'prefix_updates':0,'instruction_tokens':0,'policy_trace_logits_checked':0,'root_logits_checked':0}
    for e in episodes:
        path=CACHE/'policy'/f"{e['key']}.pt";m=read(path.with_suffix('.json'))
        assert m['protocol_sha256']==signature and sha(path)==m['sha256']
        files[str(path.relative_to(CACHE))]={'sha256':m['sha256'],'bytes':path.stat().st_size}
        v=torch.load(path,map_location='cpu',weights_only=True);n=e['max_t']+1
        assert v['format']=='frozen_real_prefix_v1' and v['protocol_sha256']==signature
        assert v['policy_identity_sha256']==POLICY_ID and v['carrier_sha256']==e['carrier_sha256']
        assert not v['trainable_memory_cached'] and not any(k in v for k in ('poses','metrics','collisions','memory'))
        assert v['context'].shape==(n,960) and v['context'].dtype==torch.bfloat16
        assert v['instruction'].shape==(len(v['instruction_token_ids']),960) and v['instruction'].dtype==torch.bfloat16
        assert v['instruction_valid'].all() and v['instruction_valid'].shape==(len(v['instruction']),)
        assert v['instruction_text']==e['instruction']['text'].strip() and v['language']==e['instruction']['language']
        for key,value in v.items():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all() and not value.requires_grad,key
        assert v['frame_ids'].tolist()==e['frame_ids']
        expected=[4]+[LIVE_TO_CANONICAL[a] for a in e['actions_live']]
        assert v['previous_action_canonical'].tolist()==expected
        assert torch.equal(v['elapsed_fraction'],torch.arange(n,dtype=torch.float32)/e['budget'])
        adjusted=v['raw_logits_live'].clone();adjusted[:,0]-=.25
        assert torch.equal(adjusted,v['biased_logits_live'])
        assert v['base_action_canonical'].tolist()==[LIVE_TO_CANONICAL[a] for a in adjusted.argmax(1).tolist()]
        assert (v['prompt_lengths']<=4096).all()
        carrier=read(LABELS/e['carrier']);assert sha(LABELS/e['carrier'])==e['carrier_sha256']
        if e['prefix_source']=='policy':
            assert np.array_equal(v['raw_logits_live'].numpy(),carrier['raw_action_scores'][:n])
            totals['policy_trace_logits_checked']+=n
        for r in (r for r in roots if r['episode_index']==e['id']):
            assert np.array_equal(v['raw_logits_live'][r['t']].numpy(),r['base_raw_logits'])
            assert v['base_action_canonical'][r['t']].item()==r['base_action_canonical']
            assert v['prompt_lengths'][r['t']].item()==r['prompt_length']
            assert r['world_model_frame_indices']==[r['t']-2,r['t']-1,r['t']]
            assert r['world_model_actions_canonical']==[LIVE_TO_CANONICAL[a] for a in carrier['actions'][r['t']-2:r['t']]]
            assert r['t']>=3 and r['actual_next_image_ids'][3]==r['current_image_id']
            totals['root_logits_checked']+=1
        totals['prefix_updates']+=n;totals['instruction_tokens']+=len(v['instruction'])
    assert totals['prefix_updates']==read(RUN/'index/manifest.json')['real_prefix_updates']
    for directory in [CACHE/'vision']+[CACHE/'predictions'/trial for trial in TRIALS]:
        m=read(directory/'manifest.json');assert m['status']=='complete' and m['protocol_sha256']==signature
        for name,info in m['files'].items():
            path=directory/name;assert sha(path)==info['sha256']
            files[str(path.relative_to(CACHE))]={**info,'bytes':path.stat().st_size}
    raw=np.load(CACHE/'vision/raw.npy',mmap_mode='r');pooled=np.load(CACHE/'vision/pooled.npy',mmap_mode='r')
    assert raw.shape==(len(images),196,384) and raw.dtype==np.float16
    assert pooled.shape==(len(images),49,384) and pooled.dtype==np.float32
    for start in range(0,len(raw),128):
        values=np.asarray(raw[start:start+128]).astype(np.float32);assert np.isfinite(values).all()
        expected=values.reshape(-1,7,2,7,2,384).mean(axis=(2,4)).reshape(-1,49,384)
        np.testing.assert_allclose(pooled[start:start+len(values)],expected,atol=2e-6,rtol=1e-6)
    predictions={trial:np.load(CACHE/'predictions'/trial/'future.npy',mmap_mode='r') for trial in TRIALS}
    errors={}
    for trial,a in predictions.items():
        assert a.shape==(len(roots),4,49,384) and a.dtype==np.float32 and np.isfinite(a).all()
        per_group={g:[] for g in ('B','C')}
        for r in roots:
            assert np.array_equal(a[r['id'],3],pooled[r['current_image_id']])
            actual=pooled[r['actual_next_image_ids'][:3]]
            mse=((a[r['id'],:3]-actual)**2).mean(axis=(1,2))
            baseline=((pooled[r['current_image_id']][None]-actual)**2).mean(axis=(1,2))
            per_group[r['group']].append((mse,baseline))
        errors[trial]={g:{'movement_pooled_MSE':float(np.mean([x[0] for x in rows])),
                         'copy_current_pooled_MSE':float(np.mean([x[1] for x in rows])),
                         'per_action_MSE':np.mean([x[0] for x in rows],axis=0).tolist()} for g,rows in per_group.items()}
    # Independent fresh forwards on roots from every dataset/source/group stratum.
    selected=[]
    for g in ('B','C'):
        for d in ('r2r','rxr'):
            for m in ('expert','policy'):
                rows=[r for r in roots if (r['group'],r['dataset'],r['prefix_source'])==(g,d,m)]
                selected.extend([rows[0],rows[-1]])
    extractor=FrozenPolicyFeatures();policy_samples=[]
    for e in episodes:
        _,_,info=instruction_batch(extractor.policy.inputs,e['instruction']['text'],'cpu')
        value=torch.load(CACHE/'policy'/f"{e['key']}.pt",map_location='cpu',weights_only=True)
        assert value['instruction_token_ids'].tolist()==info['instruction_token_ids']
        assert value['instruction_span']==info['character_span']
    for r in selected:
        e=episodes[r['episode_index']];online=OnlineInputs(extractor.policy.inputs,extractor.device)
        for number in e['frame_ids'][:r['t']+1]:
            im=images[number];online.append(checked_image(LABELS/im['path'],im['sha256'],im['raw_rgb_sha256']))
        fresh=extractor.context(online(e['instruction']['text'],e['actions_live'][:r['t']]),compare_hidden=True)
        value=torch.load(CACHE/'policy'/f"{e['key']}.pt",map_location='cpu',weights_only=True)
        assert torch.equal(fresh['context'],value['context'][r['t']])
        assert torch.equal(fresh['raw_logits'],value['raw_logits_live'][r['t']])
        fresh_i=extractor.instruction(e['instruction']['text'],compare_hidden=True)
        assert torch.equal(fresh_i['states'],value['instruction'])
        policy_samples.append(r['root_id'])
    assert not any(x.requires_grad for x in extractor.policy.model.parameters())
    del extractor,online;torch.cuda.empty_cache()
    live_checks=[]
    with torch.inference_mode():
        for i,trial in enumerate(TRIALS):
            model=FrozenDinoWorldModel(FROZEN/trial,expected_policy_identity=POLICY_ID,load_encoder=i==0)
            for r in selected:
                e=episodes[r['episode_index']]
                ids=[e['frame_ids'][j] for j in r['world_model_frame_indices']]
                if i==0:
                    for number in dict.fromkeys(ids+r['actual_next_image_ids']):
                        im=images[number];image=checked_image(LABELS/im['path'],im['sha256'],im['raw_rgb_sha256'])
                        encoded=model.encode_images([image])[0].cpu().numpy()
                        assert np.array_equal(encoded,raw[number]),('Encoder replay drift',number)
                history=torch.from_numpy(np.array(raw[ids],copy=True))[None].cuda()
                past=torch.tensor([r['world_model_actions_canonical']],device='cuda')
                for a in range(3):
                    y=model.predict(history,past,torch.tensor([a],device='cuda'),observations_seen=r['t']+1,encoder_identity=ENCODER_ID)
                    f=pool_dino_patches(y.float())[0].cpu().numpy()
                    assert np.array_equal(f,predictions[trial][r['id'],a]),('Frozen prediction replay drift',trial,r['root_id'],a)
                live_checks.append({'trial':trial,'root':r['root_id'],'exact':True})
            del model;torch.cuda.empty_cache()
    # Exercise the real scorer API without training or persisting its random weights.
    torch.manual_seed(7);scorer=NavigationScorer.from_config(ROOT/'Scorer/config.json','smolvlm2').eval()
    replay_checks=[]
    with torch.inference_mode():
        for g in ('B','C'):
            dataset=FrozenScorerDataset(g,require_complete=False)
            for d in ('r2r','rxr'):
                for mode in ('expert','policy'):
                    index=next(i for i,r in enumerate(dataset.roots) if (r['dataset'],r['prefix_source'])==(d,mode))
                    prefix=dataset.prefix(index);memory=None
                    for t in range(len(prefix['policy_context'])):
                        context=scorer.observe(**dataset.observation(prefix,t),memory=memory);memory=context.memory
                    before=memory.clone();targets=dataset.targets(index)
                    for representation in ('actual',)+TRIALS:
                        out=scorer.score(context,dataset.futures(index,representation),targets['actions'])
                        assert all(v.shape==(1,4) and torch.isfinite(v).all() for v in out.values())
                        assert torch.equal(memory,before)
                    replay_checks.append(dataset.roots[index]['root_id'])
    # Recheck upstream identities and controls after all extraction/inference.
    ref=verify_bundle(REFERENCE);assert not ref['selection']['original_stopping_safeguard_passed']
    assert sha(Path(ref['source_checkpoint'])/'adapter_model.safetensors')==ref['identity_fields']['adapter_sha256']
    control=Path(ref['selection']['control_reference'])
    for f,h in read(control/'sha256.json').items():assert sha(control/f)==h
    for f,h in read(FROZEN/'encoder/file_sha256.json').items():assert sha(FROZEN/'encoder'/f)==h
    for trial in TRIALS:
        for f,h in read(FROZEN/trial/'file_sha256.json').items():assert sha(FROZEN/trial/f)==h
    verify_protocol()
    report={'status':'passed','completed_utc':now(),'protocol_sha256':signature,**totals,
        'episodes':len(episodes),'roots':len(roots),'unique_images':len(images),
        'all_array_and_episode_file_hashes_checked':True,'all_shapes_dtypes_finite_and_pooling_checked':True,
        'all_STOP_futures_exact_identity':True,'instruction_preserved_without_truncation':True,
        'native_policy_replay_exact_roots':policy_samples,'native_encoder_and_prediction_replay':live_checks,
        'real_prefix_scorer_API_smoke_roots':replay_checks,'candidate_scoring_preserves_memory':True,
        'policy_world_models_frozen_and_controls_preserved':True,'no_scorer_trained':True,
        'prediction_diagnostics':errors,'diagnostic_scope':'Pooled feature reconstruction on mixed real roots; not navigation performance'}
    write(RUN/'verification.json',report)
    catalogs={}
    for group in ('B','C'):
        entries=[]
        for r in roots:
            if r['group']!=group:continue
            e=episodes[r['episode_index']]
            entries.append({'root_id':r['root_id'],'group':group,'episode_index':e['id'],'t':r['t'],
                'policy_prefix':f"policy/{e['key']}.pt",'visual_pool':'vision/pooled.npy',
                'current_image_id':r['current_image_id'],'actual_next_image_ids':r['actual_next_image_ids'],
                'prediction_row':r['id'],'predictions':{name:f'predictions/{name}/future.npy' for name in TRIALS},
                'actions_canonical':[0,1,2,3],'base_action_canonical':r['base_action_canonical'],
                'labels':r['candidates_canonical'],'source_labels':str(LABELS/r['labels']),'source_labels_sha256':r['labels_sha256']})
        path=CACHE/f'catalog_{group}.jsonl';temp=path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in entries));temp.replace(path)
        catalogs[group]={'file':str(path),'roots':len(entries),'sha256':sha(path)}
        files[path.name]={'sha256':sha(path),'bytes':path.stat().st_size}
    cfg=read(ROOT/'Scorer/config.json')
    cfg['status']='Step7_inputs_verified_Step8_scorer_not_trained'
    cfg['visual_features']['encoder_weights_revision']=ENCODER_ID
    cfg['visual_features']['checkpoint_contract_verified']=True
    cfg['frozen_input_dataset']={'version':p['version'],'protocol_sha256':signature,'policy_identity_sha256':POLICY_ID,
        'world_model_trials':list(TRIALS),'equal_model_sampling_probabilities':[.5,.5],
        'world_model_identity_input':False,'future_grad':False,'memory_recomputed_from_real_prefix':True}
    write(CACHE/'scorer_config.json',cfg);files['scorer_config.json']={'sha256':sha(CACHE/'scorer_config.json'),'bytes':(CACHE/'scorer_config.json').stat().st_size}
    manifest={'status':'complete','format':'frozen_scorer_feature_dataset_v1','completed_utc':now(),
        'protocol_sha256':signature,'protocol':str(RUN/'protocol.json'),'index':str(RUN/'index'),
        'index_manifest_sha256':sha(RUN/'index/manifest.json'),'policy_identity_sha256':POLICY_ID,
        'encoder_identity_sha256':ENCODER_ID,'world_models':p['models'],'scope':p['scope'],
        'catalogs':catalogs,'files':files,'counts':{'episodes':len(episodes),'roots':len(roots),
        'real_prefix_updates':totals['prefix_updates'],'unique_images':len(images),'measured_action_labels':4*len(roots),
        'movement_predictions_per_model':3*len(roots)},'verification':str(RUN/'verification.json'),
        'verification_sha256':sha(RUN/'verification.json'),'trainable_memory_cached':False,'scorer_trained':False}
    write(CACHE/'manifest.json',manifest)
    result={'status':'complete','completed_utc':now(),'scope':'Step 7 frozen features for SmolVLM2 and the DINO pair',
        'policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,'cache':str(CACHE),
        'manifest_sha256':sha(CACHE/'manifest.json'),'catalogs':catalogs,'counts':manifest['counts'],
        'all_checks_passed':True,'verification':str(RUN/'verification.json'),'preserved_selection_caveat':ref['selection'],
        'files_bytes':sum(f['bytes'] for f in files.values()),'prediction_diagnostics':errors,
        'next':'Step 8: train one scorer on the pooled frozen DINO pair, fitting on B and developing on C'}
    write(RUN/'result.json',result)
    lines=['# Step 7 complete — frozen scorer features','',f"Completed: {result['completed_utc']}",'',
        'The cache preserves refined SmolVLM2 checkpoint 512, STOP bias −0.25, its native saved processor and batch-one decoding. Its deliberate selection despite the failed stopping safeguard remains recorded. Step 2 and Step 5 artifacts are unchanged.','',
        f"There are 1,024 B fitting roots and 256 C development roots from {len(episodes)} episodes, with {totals['prefix_updates']:,} real observation updates and {len(images):,} distinct encoded images. All 5,120 measured action labels are retained. Each DINO model supplies 3,840 movement predictions; STOP is exact current-view identity.",'',
        'Instruction token states and each actual prefill context are stored in BF16. Every instruction is complete within the saved context limit. Real DINO patches use the frozen FP32 encoder, FP16 raw storage, and FP32 2×2 pooling to 49×384. Encoding uses one image per native forward: large batches can select TF32 cuDNN patch convolution, documented in [the diagnostic](dino_batch_diagnosis.json). Both world models use the same actual history and canonical commands.','',
        'The cache includes every real prefix needed to recompute trainable scorer memory. It contains no cached scorer memory. Actual next views and both predicted views carry identical measured labels. The initial comparison includes the DINO transfer/fresh pair; NWM/CDiT requires a newly versioned pooled dataset and scorer.','',
        '[Runtime checks](check_runtime.json) and [full verification](verification.json) passed, including exact native policy replay, exact sampled encoder/prediction replay, all cache hashes and STOP identities, and the scorer memory/shape interface. The scorer API check used discarded random weights; no scorer was trained.','',
        '| Model | B movement pooled MSE | C movement pooled MSE |','| --- | ---: | ---: |']
    for trial in TRIALS:lines.append(f"| {trial} | {errors[trial]['B']['movement_pooled_MSE']:.6f} | {errors[trial]['C']['movement_pooled_MSE']:.6f} |")
    lines+=['','These feature errors describe mixed expert/policy roots, not navigation success or scorer gains. Both frozen models are retained.','',
        f'Cache: `{CACHE}`. `catalog_B.jsonl` is for fitting and `catalog_C.jsonl` for development. `scorer_config.json` records the initial pooled pair and equal model weights. Use `scorer_features.dataset.FrozenScorerDataset` to load prefixes, candidates and labels; recompute memory under current scorer weights in Step 8.','',
        'See [machine-readable results](result.json), [locked protocol](protocol.json), and [methods](../../docs/Step7_Run.md).','']
    (RUN/'RESULTS.md').write_text('\n'.join(lines))
    write(RUN/'status.json',{'status':'complete','completed_utc':now(),'result':str(RUN/'result.json')})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
