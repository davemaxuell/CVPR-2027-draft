"""Audit the four-model cache and fresh native predictions across all eight strata."""
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from nwm_training.frozen import FrozenNwmWorldModel
from scorer_features.common import native_flags
from scorer_four.common import (ADDITIONS, BASE_CACHE, BASE_RUN, CACHE, ENCODER_ID, FEATURES_RUN,
    POLICY_ID, TRIALS, now, read, sha, verify_features_protocol, write)
from tools.cache_four_features import actual_rgb


def main():
    native_flags(); torch.backends.mha.set_fastpath_enabled(False); verify_features_protocol()
    base = read(BASE_CACHE / 'manifest.json')
    assert base['policy_identity_sha256'] == POLICY_ID and not base['trainable_memory_cached']
    for name, entry in base['files'].items():
        path = BASE_CACHE / name
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    roots = read(FEATURES_RUN / 'roots.json'); originals = read(BASE_RUN / 'index/roots.json')
    images = read(BASE_RUN / 'index/images.json')
    moment_meta = read(CACHE / 'moments.json')
    assert moment_meta['protocol_sha256'] == sha(FEATURES_RUN / 'protocol.json')
    assert sha(CACHE / 'moments.npy') == moment_meta['moments_sha256']
    moments = np.load(CACHE / 'moments.npy', mmap_mode='r')
    slot_of = {i: slot for slot, i in enumerate(moment_meta['image_ids'])}
    pooled = np.load(BASE_CACHE / 'vision/pooled.npy', mmap_mode='r')
    spot = []
    for group in ('B','C'):
        for dataset in ('r2r','rxr'):
            for source in ('expert','policy'):
                part = [r for r in roots if (r['group'],r['dataset'],r['prefix_source']) == (group,dataset,source)]
                part.sort(key=lambda r: (originals[r['id']]['t'], r['id']))
                spot += [part[0], part[-1]]
    assert len({r['id'] for r in spot}) == 16
    checks = []; files = {'moments.npy': sha(CACHE / 'moments.npy'), 'moments.json': sha(CACHE / 'moments.json')}
    for name in ADDITIONS:
        metadata = read(CACHE / name / 'manifest.json')
        assert metadata['protocol_sha256'] == sha(FEATURES_RUN / 'protocol.json')
        path = CACHE / name / 'future.npy'
        assert metadata['status'] == 'complete' and sha(path) == metadata['future_sha256']
        array = np.load(path, mmap_mode='r')
        assert array.shape == (1280,4,49,384) and array.dtype == np.float32 and np.isfinite(array).all()
        assert np.array_equal(array[:,3], pooled[[r['current_image_id'] for r in originals]])
        model = FrozenNwmWorldModel(name)
        for root in spot:
            context = torch.cat([model.encode_actual(actual_rgb(images[i])).cpu() for i in root['four_real_image_ids']])[None]
            cached = np.array(moments[[slot_of[i] for i in root['four_real_image_ids']]], copy=True)
            assert np.array_equal(context[0].numpy(), cached)
            future = model.predict_movements(context, context_seed=root['context_seed'], sampling_seed=root['diffusion_seed'])
            assert np.array_equal(future.cpu().numpy(), array[root['id'],:3])
            checks.append({'model': name, 'root_id': root['root_id'], 'stratum': [root['group'],root['dataset'],root['prefix_source']],
                           'live_moments_exact': True, 'live_predictions_exact': True})
            print('Verified native extension', name, root['root_id'], flush=True)
        files[f'{name}/future.npy'] = metadata['future_sha256']; files[f'{name}/manifest.json'] = sha(CACHE / name / 'manifest.json')
        del model; gc.collect(); torch.cuda.empty_cache()
    verify_features_protocol()
    report = {'status': 'passed', 'completed_utc': now(), 'protocol_sha256': sha(FEATURES_RUN / 'protocol.json'),
        'original_feature_files_rehashed': len(base['files']), 'original_policy_instruction_prefix_and_labels_preserved': True,
        'roots': 1280, 'B_roots': 1024, 'C_roots': 256, 'new_movement_predictions_per_model': 3840,
        'STOP_identity_all_roots_exact': True, 'all_new_features_finite': True, 'fresh_native_cases': checks,
        'trainable_memory_cached': False, 'no_E_used': True, 'files': files}
    write(FEATURES_RUN / 'verification.json', report)
    manifest = {'status': 'complete', 'format': 'four_WM_scorer_feature_extension_v1', 'completed_utc': now(),
        'protocol_sha256': sha(FEATURES_RUN / 'protocol.json'), 'base_cache': str(BASE_CACHE),
        'base_manifest_sha256': sha(BASE_CACHE / 'manifest.json'), 'policy_identity_sha256': POLICY_ID,
        'encoder_identity_sha256': ENCODER_ID, 'trials': list(TRIALS), 'roots': 1280,
        'verification_sha256': sha(FEATURES_RUN / 'verification.json'), 'files_sha256': files,
        'trainable_memory_cached': False, 'scorer_trained': False}
    write(CACHE / 'manifest.json', manifest)
    result = {'status': 'complete', 'completed_utc': now(), 'protocol_sha256': sha(FEATURES_RUN / 'protocol.json'),
        'manifest_sha256': sha(CACHE / 'manifest.json'), 'verification_sha256': sha(FEATURES_RUN / 'verification.json'),
        'cache': str(CACHE), 'models': list(TRIALS), 'roots': 1280, 'policy_identity_sha256': POLICY_ID,
        'next': 'Step 8B: matched pooled four-world-model scorer and null-future control'}
    write(FEATURES_RUN / 'result.json', result)
    write(FEATURES_RUN / 'status.json', {'status': 'complete', 'updated_utc': now()})
    (FEATURES_RUN / 'RESULTS.md').write_text('# Step 7B — four-model scorer features\n\n'
        'The original 1,280 measured roots, full actual prefixes, policy contexts, instruction features, labels and both DINO predictions are preserved exactly. Each added NWM/CDiT model provides 3,840 movement predictions; STOP copies the current view.\n\n'
        'All original cache files were rehashed. All new features are finite, and fresh live VAE encoding and complete prediction agree exactly on 16 roots per model across all B/C, R2R/RxR and expert/policy strata. No trainable memory or E data is cached.\n\n'
        '[Result](result.json), [verification](verification.json), [protocol](protocol.json).\n')
    print('Four-world-model features verified', flush=True)


if __name__ == '__main__':
    main()
