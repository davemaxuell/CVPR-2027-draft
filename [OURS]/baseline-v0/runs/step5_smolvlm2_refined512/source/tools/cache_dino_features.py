"""Encode each byte-distinct source image once, resumably, without fitting on C."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.common import ROOT, RUN, now, read, sha, write
from wm.dino import build_encoder
from wm.features import CACHE, IndexedImages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = (CACHE / 'cache.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index = RUN / 'data_index'
    data = read(index / 'manifest.json')
    assert data['status'] == 'complete'
    for name, expected in data['files'].items():
        assert sha(index / name) == expected
    assets = RUN / 'assets'
    native = read(assets / 'dinov2_native_provenance.json')
    assert read(assets / 'encoder_weight_equivalence.json')['status'] == 'passed'
    identity = {'family': 'native_dinov2_vits14', 'weights_sha256': native['weight_sha256'],
                'source_revision': native['revision'], 'source_sha256': native['source_sha256'],
                'HF_weights_equivalence': read(assets / 'encoder_weight_equivalence.json'),
                'preprocessing': 'RGB uint8 /255; tensor Resize shortest side224 antialias; CenterCrop224; normalize mean/std .5; Resize196 antialias, CPU float32',
                'position_interpolation': {'offset': .1, 'antialias': False},
                'feature_shape': [196, 384], 'forward': 'native forward_features.x_norm_patchtokens',
                'parameters': 'frozen float32, eval', 'compute_autocast': 'disabled_float32', 'stored_dtype': 'float16',
                'torch_version': torch.__version__}
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config = {'format': 'frozen_dino_features_v1', 'index_manifest_sha256': sha(index / 'manifest.json'),
              'selected_policy_identity_sha256': data['selected_policy_identity_sha256'],
              'encoder_identity_sha256': identity_hash, 'encoder': identity,
              'rows': data['unique_images'], 'shape': [data['unique_images'], 196, 384],
              'source_sha256': {name: sha(ROOT / name) for name in ['wm/common.py', 'wm/dino.py', 'wm/features.py', 'tools/cache_dino_features.py', 'Scorer/world_models.py', 'Scorer/interfaces.py']}}
    if (CACHE / 'config.json').exists():
        assert read(CACHE / 'config.json') == config
    else:
        write(CACHE / 'config.json', config)
    write(assets / 'encoder_identity.json', {'sha256': identity_hash, **identity})
    if (CACHE / 'features.npy').exists() and (CACHE / 'manifest.pending.json').exists():
        pending = read(CACHE / 'manifest.pending.json')
        assert pending['config_sha256'] == sha(CACHE / 'config.json')
        assert sha(CACHE / 'features.npy') == pending['features_sha256']
        (CACHE / 'manifest.pending.json').replace(CACHE / 'manifest.json')
    if (CACHE / 'manifest.json').exists():
        assert read(CACHE / 'manifest.json')['config_sha256'] == sha(CACHE / 'config.json')
        print((CACHE / 'manifest.json').read_text()); return
    partial = CACHE / 'features.partial.npy'
    progress = read(CACHE / 'progress.json') if (CACHE / 'progress.json').exists() else {'completed_rows': 0}
    done = progress['completed_rows']
    if not partial.exists():
        assert done == 0
        features = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16, shape=tuple(config['shape']))
    else:
        features = np.load(partial, mmap_mode='r+')
        assert list(features.shape) == config['shape'] and features.dtype == np.float16
    if done:
        start = progress['last_chunk_start']
        assert hashlib.sha256(np.ascontiguousarray(features[start:done]).tobytes()).hexdigest() == progress['last_chunk_sha256']
    encoder = build_encoder()
    assert not any(p.requires_grad for p in encoder.parameters()) and not encoder.training
    loader = DataLoader(IndexedImages(start=done), batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(5))
    started = time.time()
    initial = done
    chunk_start = done
    journal = (CACHE / 'chunks.jsonl').open('a')
    with torch.inference_mode():
        for batch_number, (ids, pixels) in enumerate(loader):
            assert ids.tolist() == list(range(done, done + len(ids)))
            values = encoder.encoder.forward_features(pixels.cuda(non_blocking=True))['x_norm_patchtokens']
            assert values.shape[1:] == (196, 384) and torch.isfinite(values).all()
            features[done:done + len(ids)] = values.cpu().numpy().astype(np.float16)
            done += len(ids)
            if (batch_number + 1) % 32 == 0 or done == len(features):
                features.flush()
                with partial.open('rb') as file:
                    os.fsync(file.fileno())
                digest = hashlib.sha256(np.ascontiguousarray(features[chunk_start:done]).tobytes()).hexdigest()
                record = {'start': chunk_start, 'end': done, 'sha256': digest}
                journal.write(json.dumps(record) + '\n'); journal.flush(); os.fsync(journal.fileno())
                elapsed = time.time() - started
                progress = {'status': 'encoding', 'completed_rows': done, 'total_rows': len(features),
                            'elapsed_seconds_this_run': elapsed, 'images_per_second': (done - initial) / elapsed,
                            'last_chunk_start': chunk_start, 'last_chunk_sha256': digest, 'updated_utc': now()}
                write(CACHE / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
                chunk_start = done
    journal.close()
    assert done == len(features)
    # Hash every completed feature byte once before publishing the immutable cache.
    digest = sha(partial)
    del features
    manifest = {'status': 'complete', 'completed_utc': now(), 'config_sha256': sha(CACHE / 'config.json'),
                'encoder_identity_sha256': identity_hash, 'selected_policy_identity_sha256': data['selected_policy_identity_sha256'],
                'rows': done, 'features_sha256': digest, 'bytes': partial.stat().st_size,
                'source_images_hash_checked_on_read': True, 'fitted_on_validation': False,
                'seconds_this_run': time.time() - started}
    write(CACHE / 'manifest.pending.json', manifest)
    partial.rename(CACHE / 'features.npy')
    (CACHE / 'manifest.pending.json').replace(CACHE / 'manifest.json')
    write(CACHE / 'progress.json', {'status': 'complete', 'completed_rows': done, 'total_rows': done, 'updated_utc': now()})
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
