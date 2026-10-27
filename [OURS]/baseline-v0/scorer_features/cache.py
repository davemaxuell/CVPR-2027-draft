"""Resumable frozen-feature jobs; upstream checkpoints and labels are read-only."""
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from policy.online import OnlineInputs
from Scorer.interfaces import pool_dino_patches
from .common import (CACHE, ENCODER_ID, FROZEN, LABELS, LIVE_TO_CANONICAL, POLICY_ID, RUN,
                     TRIALS, now, read, sha, torch_save, verify_protocol, write)


def checked_image(path, expected, raw_expected):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected, str(path)
    with Image.open(io.BytesIO(raw)) as im:
        assert im.mode=='RGB' and im.size==(640,480)
        image=im.copy()
    assert hashlib.sha256(np.array(image).tobytes()).hexdigest()==raw_expected
    return image


def policy_job(rank=0, workers=2, limit=None):
    from .policy import FrozenPolicyFeatures
    p=verify_protocol();signature=sha(RUN/'protocol.json')
    assert workers==p['runtime']['policy_workers']
    episodes=read(RUN/'index/episodes.json');images=read(RUN/'index/images.json');roots=read(RUN/'index/roots.json')
    selected=episodes[rank::workers]
    if limit is not None:selected=selected[:limit]
    extractor=FrozenPolicyFeatures(); started=time.time();done=0;steps=0
    for e in selected:
        path=CACHE/'policy'/f"{e['key']}.pt";marker=path.with_suffix('.json')
        if marker.exists():
            m=read(marker);assert m['protocol_sha256']==signature and sha(path)==m['sha256']
            done+=1;steps+=e['max_t']+1;continue
        carrier=read(LABELS/e['carrier']);assert sha(LABELS/e['carrier'])==e['carrier_sha256']
        instruction=extractor.instruction(e['instruction']['text'])
        online=OnlineInputs(extractor.policy.inputs,extractor.device)
        q,raw,biased,chosen,lengths=[],[],[],[],[]
        targets={r['t']:r for r in roots if r['episode_index']==e['id']}
        for t,image_id in enumerate(e['frame_ids']):
            im=images[image_id]
            online.append(checked_image(LABELS/im['path'],im['sha256'],im['raw_rgb_sha256']))
            batch=online(e['instruction']['text'],e['actions_live'][:t])
            v=extractor.context(batch)
            if e['prefix_source']=='policy':
                assert np.array_equal(v['raw_logits'].numpy(),carrier['raw_action_scores'][t]),('Carrier native score drift',e['key'],t)
                assert v['base_live']==carrier['actions'][t]
                assert v['prompt_length']==carrier['prompt_lengths'][t]
            if t in targets:
                r=targets[t]
                assert np.array_equal(v['raw_logits'].numpy(),r['base_raw_logits']),('Root native score drift',r['root_id'])
                assert v['prompt_length']==r['prompt_length']
                assert LIVE_TO_CANONICAL[v['base_live']]==r['base_action_canonical']
            q.append(v['context']);raw.append(v['raw_logits']);biased.append(v['biased_logits'])
            chosen.append(LIVE_TO_CANONICAL[v['base_live']]);lengths.append(v['prompt_length'])
            if t%50==0:
                write(RUN/f'policy_progress_{rank}.json',{'status':'running','episodes_done':done,'episodes_total':len(selected),
                    'updates_done':steps+t+1,'episode':e['key'],'episode_t':t,'updated_utc':now()})
        value={'format':'frozen_real_prefix_v1','protocol_sha256':signature,'episode_index':e['id'],
            'episode_key':e['key'],'carrier_sha256':e['carrier_sha256'],'policy_identity_sha256':POLICY_ID,
            'instruction':instruction['states'],'instruction_valid':instruction['valid'],
            'instruction_token_ids':torch.tensor(instruction['instruction_token_ids'],dtype=torch.long),
            'instruction_text':instruction['text'],'language':e['instruction']['language'],
            'instruction_prompt':instruction['prompt'],'instruction_span':instruction['character_span'],
            'context':torch.stack(q),'raw_logits_live':torch.stack(raw),'biased_logits_live':torch.stack(biased),
            'base_action_canonical':torch.tensor(chosen,dtype=torch.long),
            'previous_action_canonical':torch.tensor(e['previous_action_canonical'],dtype=torch.long),
            'elapsed_fraction':torch.arange(e['max_t']+1,dtype=torch.float32)/e['budget'],
            'frame_ids':torch.tensor(e['frame_ids'],dtype=torch.long),'prompt_lengths':torch.tensor(lengths,dtype=torch.long),
            'original_budget':e['budget'],'trainable_memory_cached':False}
        assert value['context'].dtype==torch.bfloat16 and value['instruction'].dtype==torch.bfloat16
        torch_save(path,value)
        write(marker,{'status':'complete','protocol_sha256':signature,'sha256':sha(path),'updates':e['max_t']+1,
                      'instruction_tokens':len(instruction['instruction_token_ids']),'completed_utc':now()})
        done+=1;steps+=e['max_t']+1
        progress={'status':'running','episodes_done':done,'episodes_total':len(selected),'updates_done':steps,
                  'seconds':time.time()-started,'episode':e['key'],'updated_utc':now()}
        write(RUN/f'policy_progress_{rank}.json',progress);print(json.dumps(progress),flush=True)
    write(RUN/f'policy_progress_{rank}.json',{'status':'complete','episodes_done':done,'episodes_total':len(selected),
        'updates_done':steps,'seconds':time.time()-started,'completed_utc':now()})


class Images(Dataset):
    def __init__(self, records, start=0):
        from wm.dino import preprocessing
        self.records=records;self.start=start;self.transform=preprocessing()
    def __len__(self):return len(self.records)-self.start
    def __getitem__(self,i):
        i+=self.start;r=self.records[i]
        return i,self.transform(checked_image(LABELS/r['path'],r['sha256'],r['raw_rgb_sha256']))


def finish_pending(directory):
    pending=directory/'manifest.pending.json'
    if pending.exists():
        m=read(pending)
        for name,info in m['files'].items():
            path=directory/name;part=directory/(name+'.partial')
            current=path if path.exists() else part
            assert sha(current)==info['sha256']
            if current==part:part.replace(path)
        pending.replace(directory/'manifest.json')
    return (directory/'manifest.json').exists()


def open_arrays(directory, specs, signature):
    directory.mkdir(parents=True,exist_ok=True)
    progress=read(directory/'progress.json') if (directory/'progress.json').exists() else {'rows':0,'protocol_sha256':signature}
    assert progress['protocol_sha256']==signature
    arrays={}
    for name,(shape,dtype) in specs.items():
        path=directory/(name+'.partial')
        if path.exists():a=np.load(path,mmap_mode='r+')
        else:
            assert progress['rows']==0
            a=np.lib.format.open_memmap(path,mode='w+',shape=shape,dtype=dtype)
        assert a.shape==tuple(shape) and a.dtype==np.dtype(dtype);arrays[name]=a
        if progress['rows']:
            digest=hashlib.sha256(np.ascontiguousarray(a[progress['last_start']:progress['rows']]).tobytes()).hexdigest()
            assert digest==progress['last_chunk_sha256'][name]
    return arrays,progress['rows']


def checkpoint(directory,arrays,start,end,signature,**extra):
    hashes={}
    for name,a in arrays.items():
        a.flush()
        with (directory/(name+'.partial')).open('rb') as f:os.fsync(f.fileno())
        hashes[name]=hashlib.sha256(np.ascontiguousarray(a[start:end]).tobytes()).hexdigest()
    write(directory/'progress.json',{'status':'running','protocol_sha256':signature,'rows':end,'last_start':start,
        'last_chunk_sha256':hashes,'updated_utc':now(),**extra})


def seal(directory,arrays,signature,**extra):
    files={name:{'sha256':sha(directory/(name+'.partial')),'shape':list(a.shape),'dtype':str(a.dtype)} for name,a in arrays.items()}
    write(directory/'manifest.pending.json',{'status':'complete','protocol_sha256':signature,'files':files,'completed_utc':now(),**extra})
    finish_pending(directory)


@torch.inference_mode()
def vision_job():
    from wm.dino import build_encoder
    p=verify_protocol();signature=sha(RUN/'protocol.json');directory=CACHE/'vision';directory.mkdir(parents=True,exist_ok=True)
    if finish_pending(directory):
        assert read(directory/'manifest.json')['protocol_sha256']==signature;return
    images=read(RUN/'index/images.json');n=len(images)
    arrays,done=open_arrays(directory,{'raw.npy':((n,196,384),'float16'),'pooled.npy':((n,49,384),'float32')},signature)
    encoder=build_encoder(assets=FROZEN/'encoder');assert not any(x.requires_grad for x in encoder.parameters())
    loader=DataLoader(Images(images,done),batch_size=p['visual']['encoder_batch'],num_workers=p['visual']['loader_workers'],
                      pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(7))
    start=done;started=time.time()
    for number,(ids,pixels) in enumerate(loader):
        assert ids.tolist()==list(range(done,done+len(ids)))
        with torch.autocast('cuda',enabled=False):
            raw=encoder.encoder.forward_features(pixels.cuda(non_blocking=True))['x_norm_patchtokens'].to(torch.float16)
        pooled=pool_dino_patches(raw.float())
        assert torch.isfinite(raw).all() and torch.isfinite(pooled).all()
        arrays['raw.npy'][done:done+len(ids)]=raw.cpu().numpy()
        arrays['pooled.npy'][done:done+len(ids)]=pooled.cpu().numpy();done+=len(ids)
        if done-start>=512 or done==n:
            checkpoint(directory,arrays,start,done,signature,total=n,seconds=time.time()-started)
            print(json.dumps({'stage':'vision','rows':done,'total':n}),flush=True);start=done
    seal(directory,arrays,signature,encoder_identity_sha256=ENCODER_ID,source_images_hash_checked=True,
         source_pixels_hash_checked=True,seconds=time.time()-started)


@torch.inference_mode()
def prediction_job():
    from wm.frozen import FrozenDinoWorldModel
    p=verify_protocol();signature=sha(RUN/'protocol.json')
    visual=read(CACHE/'vision/manifest.json');assert visual['protocol_sha256']==signature
    for name,info in visual['files'].items():assert sha(CACHE/'vision'/name)==info['sha256']
    raw=np.load(CACHE/'vision/raw.npy',mmap_mode='r');pooled=np.load(CACHE/'vision/pooled.npy',mmap_mode='r')
    episodes=read(RUN/'index/episodes.json');roots=read(RUN/'index/roots.json')
    for trial in TRIALS:
        directory=CACHE/'predictions'/trial;directory.mkdir(parents=True,exist_ok=True)
        if finish_pending(directory):
            assert read(directory/'manifest.json')['protocol_sha256']==signature;continue
        arrays,done=open_arrays(directory,{'future.npy':((len(roots),4,49,384),'float32')},signature)
        model=FrozenDinoWorldModel(FROZEN/trial,expected_policy_identity=POLICY_ID)
        started=time.time();start=done
        for row in roots[done:]:
            e=episodes[row['episode_index']];t=row['t'];assert row['id']==done and t>=3
            frame_ids=[e['frame_ids'][j] for j in row['world_model_frame_indices']]
            history=torch.from_numpy(np.array(raw[frame_ids],copy=True))[None].cuda()
            past=torch.tensor([row['world_model_actions_canonical']],device='cuda',dtype=torch.long)
            assert history.shape==(1,3,196,384) and past.shape==(1,2)
            for a in range(3):
                predicted=model.predict(history,past,torch.tensor([a],device='cuda'),observations_seen=t+1,encoder_identity=ENCODER_ID)
                value=pool_dino_patches(predicted.float())[0]
                assert torch.isfinite(value).all();arrays['future.npy'][done,a]=value.cpu().numpy()
            arrays['future.npy'][done,3]=pooled[row['current_image_id']]
            done+=1
            if done%32==0 or done==len(roots):
                checkpoint(directory,arrays,start,done,signature,total=len(roots),seconds=time.time()-started)
                print(json.dumps({'stage':'predictions','trial':trial,'roots':done,'total':len(roots)}),flush=True);start=done
        seal(directory,arrays,signature,trial=trial,model_sha256=p['models'][trial]['model_sha256'],
             calls_per_root=3,STOP_model_calls=0,seconds=time.time()-started)
        del model;torch.cuda.empty_cache()
