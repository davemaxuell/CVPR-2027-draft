"""Cache frozen VAE mean/log-variance once; sample fresh posteriors during fitting."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

from nwm_training.common import CACHE, INDEX, RUN, now, read, sha, verify_assets, verify_index, write
from nwm_training.data import IndexedVAEImages, posterior_moments
from nwm_training.engine import configure
from nwm_training.model import build_vae


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--workers', type=int, default=12); args = parser.parse_args()
    configure(); data = verify_index(); assets = verify_assets()
    check = read(RUN / 'check_gpu.json'); assert check['status'] == 'passed'
    assert args.batch_size == check['selected_cache_batch_size']
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = (CACHE / 'cache.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = ['nwm_training/common.py', 'nwm_training/data.py', 'nwm_training/model.py',
             'nwm_training/engine.py', 'wm/features.py', 'tools/cache_nwm_posteriors.py']
    config = {'format': 'frozen_VAE_posterior_moments_v1', 'assets_sha256': sha(RUN / 'assets/selection.json'),
        'index_manifest_sha256': sha(INDEX / 'manifest.json'), 'rows': data['unique_images'],
        'shape': [data['unique_images'], 8, 28, 28], 'dtype': 'float16',
        'channels': 'four posterior mean, four clamped log-variance; unscaled',
        'encoder_compute': 'frozen FP32, autocast off, matmul/cuDNN TF32 off',
        'preprocessing': assets['preprocessing'], 'batch_size': args.batch_size,
        'selected_policy_identity_sha256': data['selected_policy_identity_sha256'],
        'source_sha256': {n: sha(ROOT / n) for n in names}}
    for n, digest in check['source_sha256'].items(): assert sha(ROOT / n) == digest, n
    if (CACHE / 'config.json').exists(): assert read(CACHE / 'config.json') == config
    else: write(CACHE / 'config.json', config)
    published = CACHE / 'posteriors.npy'; partial = CACHE / 'posteriors.partial.npy'
    pending = CACHE / 'manifest.pending.json'; manifest = CACHE / 'manifest.json'
    if published.exists() and pending.exists():
        value = read(pending); assert value['config_sha256'] == sha(CACHE / 'config.json')
        assert sha(published) == value['posteriors_sha256']; pending.replace(manifest)
    if manifest.exists():
        value = read(manifest); assert value['config_sha256'] == sha(CACHE / 'config.json')
        assert sha(published) == value['posteriors_sha256']; print('Already complete'); return
    progress = read(CACHE / 'progress.json') if (CACHE / 'progress.json').exists() else {'completed_rows': 0}
    done = initial = progress['completed_rows']
    if partial.exists():
        moments = np.load(partial, mmap_mode='r+')
        assert list(moments.shape) == config['shape'] and moments.dtype == np.float16
    else:
        assert not done
        moments = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16, shape=tuple(config['shape']))
    if done:
        start = progress['last_chunk_start']
        assert hashlib.sha256(np.ascontiguousarray(moments[start:done]).tobytes()).hexdigest() == progress['last_chunk_sha256']
    vae = build_vae('cuda:0')
    loader = DataLoader(IndexedVAEImages(start=done), batch_size=args.batch_size, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=True, generator=torch.Generator().manual_seed(5))
    started = time.monotonic(); chunk_start = done
    with (CACHE / 'chunks.jsonl').open('a') as journal, torch.inference_mode():
        for batch_index, (ids, pixels) in enumerate(loader):
            assert ids.tolist() == list(range(done, done + len(ids)))
            values = posterior_moments(vae, pixels.cuda(non_blocking=True))
            moments[done:done + len(ids)] = values.cpu().numpy().astype(np.float16); done += len(ids)
            if (batch_index + 1) % 32 == 0 or done == len(moments):
                moments.flush()
                with partial.open('rb') as file: os.fsync(file.fileno())
                digest = hashlib.sha256(np.ascontiguousarray(moments[chunk_start:done]).tobytes()).hexdigest()
                journal.write(json.dumps({'start': chunk_start, 'end': done, 'sha256': digest}) + '\n')
                journal.flush(); os.fsync(journal.fileno())
                progress = {'status': 'encoding', 'completed_rows': done, 'total_rows': len(moments),
                    'seconds_this_run': time.monotonic() - started, 'images_per_second': (done - initial) / (time.monotonic() - started),
                    'last_chunk_start': chunk_start, 'last_chunk_sha256': digest, 'updated_utc': now()}
                write(CACHE / 'progress.json', progress); print(json.dumps(progress), flush=True); chunk_start = done
    assert done == len(moments); del moments
    value = {'status': 'complete', 'completed_utc': now(), 'config_sha256': sha(CACHE / 'config.json'),
        'posteriors_sha256': sha(partial), 'rows': done, 'bytes': partial.stat().st_size,
        'source_images_hash_checked': True, 'VAE_frozen': True, 'fitted_on_C': False,
        'seconds_this_run': time.monotonic() - started}
    write(pending, value); partial.replace(published); pending.replace(manifest)
    write(CACHE / 'progress.json', {'status': 'complete', 'completed_rows': done, 'total_rows': done, 'updated_utc': now()})
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
