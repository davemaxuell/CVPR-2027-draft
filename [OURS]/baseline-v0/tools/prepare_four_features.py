"""Lock extension inputs after both NWM/CDiT models are frozen and checked."""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scorer_four.common import (ADDITIONS, BASE_CACHE, BASE_RUN, CACHE, ENCODER_ID, FEATURES_RUN,
    NWM_RUN, POLICY_ID, TRIALS, read, sha, verify_base_protocol, write)


def main():
    verify_base_protocol(); assert read(BASE_RUN / 'verification.json')['status'] == 'passed'
    result = read(NWM_RUN / 'result.json'); assert result['status'] == 'complete'
    assert read(NWM_RUN / 'final_verification.json')['status'] == 'passed'
    inputs = [BASE_RUN / 'protocol.json', BASE_RUN / 'result.json', BASE_RUN / 'verification.json',
              BASE_CACHE / 'manifest.json', NWM_RUN / 'protocol.json', NWM_RUN / 'result.json', NWM_RUN / 'final_verification.json']
    for name in ADDITIONS:
        directory = NWM_RUN / 'frozen' / name
        config = read(directory / 'config.json')
        assert config['policy_identity_sha256'] == POLICY_ID and config['encoder_identity_sha256'] == ENCODER_ID
        for file, digest in read(directory / 'file_sha256.json').items():
            assert sha(directory / file) == digest; inputs.append(directory / file)
    sources = [ROOT / 'scorer_four' / name for name in ('__init__.py','common.py','data.py')]
    sources += [ROOT / n for n in ('tools/prepare_four_features.py','tools/cache_four_features.py','tools/verify_four_features.py')]
    sources += list((ROOT / 'nwm_training').glob('*.py'))
    sources += [ROOT / 'scorer_features/dataset.py', ROOT / 'scorer_training/data.py']
    roots = read(BASE_RUN / 'index/roots.json'); episodes = read(BASE_RUN / 'index/episodes.json')
    index = []
    for root in roots:
        episode = episodes[root['episode_index']]; t = root['t']; assert t >= 3
        frames = episode['frame_ids'][t - 3:t + 1]; assert len(frames) == 4 and frames[-1] == root['current_image_id']
        prefix = 'NWM_CE_v1/5/' + root['root_id']
        seeds = {kind + '_seed': int.from_bytes(hashlib.sha256((prefix + '/' + kind).encode()).digest()[:8], 'little') % (2**63)
                 for kind in ('context','diffusion')}
        index.append({'id': root['id'], 'root_id': root['root_id'], 'group': root['group'], 'dataset': root['dataset'],
            'prefix_source': root['prefix_source'], 'four_real_image_ids': frames, **seeds})
    path = FEATURES_RUN / 'roots.json'
    if path.exists(): assert read(path) == index
    else: write(path, index)
    inputs.append(path)
    protocol = {'format': 'four_WM_frozen_scorer_features_v1', 'policy_identity_sha256': POLICY_ID,
        'encoder_identity_sha256': ENCODER_ID, 'STOP_bias': -.25, 'base_cache': str(BASE_CACHE),
        'base_policy_instruction_and_prefix_features_reused_exactly': True, 'labels_reused_exactly': True,
        'roots': 1280, 'B_roots': 1024, 'C_roots': 256, 'trials': list(TRIALS), 'additions': list(ADDITIONS),
        'minimum_real_observations': 4, 'posterior_encoding': 'FP32 native VAE, batch1, CPU shared preprocessing, FP16 stored mean/logvar',
        'sampling_steps': 50, 'candidate_sampling': 'One fixed F/L/R batch, shared sampled real history, one completed sample per candidate',
        'STOP': 'Exact current pooled DINO features, no model call', 'memory': 'No trainable memory cached; replay full actual prefixes at fitting',
        'no_E_used': True, 'input_sha256': {str(p): sha(p) for p in inputs},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    path = FEATURES_RUN / 'protocol.json'
    if path.exists(): assert read(path) == protocol
    else:
        for name in protocol['source_sha256']:
            target = FEATURES_RUN / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write(path, protocol)
    print('Locked four-model feature extension', sha(path), flush=True)


if __name__ == '__main__':
    main()
