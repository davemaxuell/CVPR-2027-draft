"""Index all verified roots and real prefixes, then lock feature extraction inputs."""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scorer_features.common import (CACHE, ENCODER_ID, FROZEN, LABELS, LIVE_TO_CANONICAL,
    POLICY_ID, REFERENCE, RUN, TRIALS, UPSTREAM, now, read, sha, verify_index, write)


def build_index():
    directory = RUN / 'index'
    if (directory / 'manifest.json').exists():
        return verify_index()
    result = read(UPSTREAM / 'result.json'); corpus = read(LABELS / 'manifest.json')
    assert result['status'] == 'complete' and result['all_label_checks_passed']
    assert read(UPSTREAM / 'full/verification.json')['status'] == 'passed'
    assert sha(LABELS / 'manifest.json') == result['corpus_manifest_sha256']
    assert sha(UPSTREAM / 'full/verification.json') == corpus['verification_sha256']
    assert result['policy_identity_sha256'] == POLICY_ID and result['STOP_bias'] == -.25
    rows = []
    for group, expected in [('B', 1024), ('C', 256)]:
        assert sha(LABELS / f'catalog_{group}.jsonl') == corpus['catalogs'][group]['sha256']
        part = [json.loads(line) for line in (LABELS / f'catalog_{group}.jsonl').read_text().splitlines()]
        assert len(part) == expected and all(row['group'] == group for row in part)
        rows.extend(part)
    grouped = {}
    for row in rows:
        grouped.setdefault(row['carrier'], []).append(row)
    episodes, images, image_ids, root_rows = [], [], {}, []

    def image_id(path, digest, pixels):
        assert (LABELS / path).is_file()
        if digest not in image_ids:
            image_ids[digest] = len(images)
            images.append({'id': len(images), 'path': path, 'sha256': digest, 'raw_rgb_sha256': pixels})
        number = image_ids[digest]
        assert images[number]['raw_rgb_sha256'] == pixels
        return number

    for carrier, roots in grouped.items():
        assert sha(LABELS / carrier) == roots[0]['carrier_sha256']
        c = read(LABELS / carrier); assert c['eligible'] and c['policy_identity_sha256'] == POLICY_ID
        last = max(r['t'] for r in roots); assert last < c['num_actions']
        assert min(r['t'] for r in roots) >= 3, 'Insufficient real world-model context'
        assert 0 not in c['actions'][:last], 'Prefix cannot continue after STOP'
        prefix = Path(carrier).parent
        frame_ids = [image_id(str(prefix / f'rgb/{t:04d}.png'), c['file_sha256'][f'rgb/{t:04d}.png'],
                              c['rgb_sha256'][t]) for t in range(last + 1)]
        key = str(prefix.parent)
        episode = {'id': len(episodes), 'key': key, 'group': roots[0]['group'], 'dataset': roots[0]['dataset'],
                   'prefix_source': c['mode'], 'episode_id': c['episode_id'], 'scene': c['scene'],
                   'carrier': carrier, 'carrier_sha256': sha(LABELS / carrier), 'max_t': last,
                   'budget': c['budget'], 'instruction': c['instruction'], 'frame_ids': frame_ids,
                   'actions_live': c['actions'][:last],
                   'previous_action_canonical': [4] + [LIVE_TO_CANONICAL[a] for a in c['actions'][:last]]}
        episodes.append(episode)
        for row in roots:
            path = LABELS / row['labels']; assert sha(path) == row['labels_sha256']
            label = read(path); actual_ids = []
            for a in range(4):
                b = row['candidates_canonical'][str(a)]
                branch_file = label['branch_files'][str(b['live_action'])]
                assert sha(path.parent / branch_file) == label['branch_sha256'][str(b['live_action'])]
                branch = read(path.parent / branch_file)
                actual_ids.append(image_id(b['first_next_rgb'], b['first_next_sha256'], branch['rgb_sha256'][1]))
            assert images[actual_ids[3]]['raw_rgb_sha256'] == images[frame_ids[row['t']]]['raw_rgb_sha256']
            root_rows.append({**row, 'id': len(root_rows), 'episode_index': episode['id'],
                'current_image_id': frame_ids[row['t']], 'actual_next_image_ids': actual_ids,
                'world_model_frame_indices': list(range(row['t'] - 2, row['t'] + 1)),
                'world_model_actions_canonical': [LIVE_TO_CANONICAL[a] for a in c['actions'][row['t']-2:row['t']]],
                'base_raw_logits': label['base_raw_action_scores'], 'prompt_length': label['root_prompt_tokens']})
    assert len(root_rows) == 1280 and len(episodes) == 161
    assert not {x['scene'] for x in episodes if x['group']=='B'} & {x['scene'] for x in episodes if x['group']=='C'}
    for name, value in [('episodes.json', episodes), ('images.json', images), ('roots.json', root_rows)]:
        write(directory / name, value)
    manifest = {'format': 'scorer_frozen_input_index_v1', 'prepared_utc': now(),
                'upstream_manifest_sha256': sha(LABELS / 'manifest.json'),
                'upstream_result_sha256': sha(UPSTREAM / 'result.json'),
                'upstream_verification_sha256': sha(UPSTREAM / 'full/verification.json'),
                'upstream_catalog_sha256': {g: sha(LABELS / f'catalog_{g}.jsonl') for g in ('B','C')},
                'episodes': len(episodes), 'roots': len(rows), 'unique_images': len(images),
                'real_prefix_updates': sum(e['max_t']+1 for e in episodes),
                'roots_by_group': dict(Counter(r['group'] for r in rows)), 'excluded_roots': [],
                'files': {name: sha(directory / name) for name in ('episodes.json','images.json','roots.json')}}
    write(directory / 'manifest.json', manifest)
    return manifest


def lock_protocol():
    import importlib.metadata
    import torch
    import transformers.models.smolvlm.modeling_smolvlm as smol
    import transformers.models.llama.modeling_llama as llama
    from policy.experimental_reference import verify_bundle
    from scorer_features.common import native_flags
    index = verify_index(); reference = verify_bundle(REFERENCE)
    checks = read(RUN / 'check_runtime.json'); assert checks['status'] == 'passed'
    assert checks['index_manifest_sha256'] == sha(RUN / 'index/manifest.json')
    source_names = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'scorer_features').glob('*.py'))
    source_names += sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tools').glob('*step7*.py'))
    source_names += ['policy/common.py','policy/online.py','policy/smolvlm2.py','policy/experimental_reference.py',
                     'Scorer/interfaces.py','Scorer/world_models.py','Scorer/scorer.py','Scorer/config.json',
                     'wm/common.py','wm/dino.py','wm/frozen.py']
    sources = {name: sha(ROOT / name) for name in sorted(set(source_names))}
    for name, expected in checks['source_sha256'].items():assert sha(ROOT/name)==expected,name
    models = {}
    for name in TRIALS:
        cfg = read(FROZEN/name/'config.json')
        assert cfg['selected_policy_identity_sha256']==POLICY_ID and cfg['encoder_identity_sha256']==ENCODER_ID
        for f,h in read(FROZEN/name/'file_sha256.json').items():assert sha(FROZEN/name/f)==h
        models[name] = {'bundle': str(FROZEN/name), 'config_sha256':sha(FROZEN/name/'config.json'),
                        'model_sha256':cfg['model_sha256'],'selected_step':cfg['selected_step']}
    protocol = {'format':'scorer_features_protocol_v1','version':'smolvlm2_refined512_dino_pair_v1',
        'policy_identity_sha256':POLICY_ID,'encoder_identity_sha256':ENCODER_ID,
        'policy_reference':str(REFERENCE),'policy_decoder':reference['decoder'],'selection_caveat':reference['selection'],
        'index_manifest_sha256':sha(RUN/'index/manifest.json'),'runtime_checks_sha256':sha(RUN/'check_runtime.json'),
        'cache':str(CACHE),'groups':{'fit':'B','development':'C'},'models':models,
        'instruction':'Native user chat containing the full instruction only; final norm at instruction-token offsets; whitespace strip matches the navigation prompt; no truncation',
        'context':{'hook':checks['policy_hook'],'mask':'native expanded input attention_mask','forward':'one full native prefill, no KV/vision approximation','stored_dtype':'bfloat16'},
        'real_memory':'All actual observation updates from zero through each root; START=4 then actual canonical previous actions; elapsed=t/original_budget; no cached trainable memory',
        'visual':{'raw_shape':[196,384],'raw_dtype':'float16','pooled_shape':[49,384],
                  'pooling':'float32 2x2 mean of raw FP16 features; store float32','encoder_batch':1,'loader_workers':4,
                  'batch_choice':'Native one-image inference; avoids cuDNN TF32 convolution changing with large encoder batches',
                  'batch_diagnosis_sha256':sha(RUN/'dino_batch_diagnosis.json')},
        'future':{'models':list(TRIALS),'samples_per_movement_action_per_model':1,'prediction_batch':1,
                  'history':'last three actual raw DINO frames, two intervening executed canonical commands',
                  'minimum_observations':4,'STOP':'exact current pooled features, no model call',
                  'actual_next':'encode each measured first-next RGB; same labels as both predictions',
                  'pooled_dtype':'float32','model_identity_input':False,'intended_model_sampling':'equal (1/2 each) in Step 8'},
        'scope':'Initial DINO pair only; NWM/CDiT requires a new pooled dataset/scorer version',
        'runtime':{'flags':native_flags(),'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','numpy','Pillow','torchvision')},
                   'device':'physical GPU 0','policy_workers':2},
        'native_source_sha256':{str(Path(m.__file__).resolve()):sha(m.__file__) for m in (smol,llama)},
        'source_sha256':sources}
    if (RUN/'protocol.json').exists():assert read(RUN/'protocol.json')==protocol
    else:
        for name in sources:
            p=RUN/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
        write(RUN/'protocol.json',protocol)
    return protocol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index-only',action='store_true');args=ap.parse_args()
    RUN.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    m=build_index();print(json.dumps(m,indent=2))
    if not args.index_only:
        p=lock_protocol();print('Locked',sha(RUN/'protocol.json'),p['version'])

if __name__=='__main__':main()
