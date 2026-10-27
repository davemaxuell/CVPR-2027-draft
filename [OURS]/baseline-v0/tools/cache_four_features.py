"""Add NWM/CDiT predictions on the existing measured roots, without rewriting inputs."""
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from nwm_training.frozen import FrozenNwmWorldModel
from scorer_features.common import native_flags
from scorer_four.common import (ADDITIONS, BASE_CACHE, BASE_RUN, CACHE, FEATURES_RUN, LABELS,
    now, read, sha, verify_features_protocol, write)


def actual_rgb(entry):
    path = LABELS / entry['path']; assert sha(path) == entry['sha256']
    with Image.open(path) as image:
        image = image.convert('RGB'); array = np.asarray(image)
        assert hashlib.sha256(array.tobytes()).hexdigest() == entry['raw_rgb_sha256']
        return ToTensor()(image)[None]


def main():
    native_flags(); torch.backends.mha.set_fastpath_enabled(False); verify_features_protocol()
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = (CACHE / 'cache.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    roots = read(FEATURES_RUN / 'roots.json'); old_roots = read(BASE_RUN / 'index/roots.json')
    images = read(BASE_RUN / 'index/images.json')
    needed = sorted({i for root in roots for i in root['four_real_image_ids']}); slot_of = {i: slot for slot, i in enumerate(needed)}
    identity = sha(FEATURES_RUN / 'protocol.json'); model = None
    if (CACHE / 'moments.npy').exists() and (CACHE / 'moments.pending.json').exists():
        pending = read(CACHE / 'moments.pending.json')
        assert pending['protocol_sha256'] == identity and sha(CACHE / 'moments.npy') == pending['moments_sha256']
        (CACHE / 'moments.pending.json').replace(CACHE / 'moments.json')
    if not (CACHE / 'moments.json').exists():
        model = FrozenNwmWorldModel(ADDITIONS[0])
        partial = CACHE / 'moments.partial.npy'
        progress = read(CACHE / 'moments_progress.json') if (CACHE / 'moments_progress.json').exists() else {'completed': 0}
        done = progress['completed']
        if done: assert partial.exists() and progress['protocol_sha256'] == identity
        moments = np.load(partial, mmap_mode='r+') if partial.exists() else np.lib.format.open_memmap(partial,
            mode='w+', dtype=np.float16, shape=(len(needed), 8, 28, 28))
        assert moments.shape == (len(needed), 8, 28, 28) and moments.dtype == np.float16
        if done:
            chunks = progress['chunks']; cursor = 0
            for chunk in chunks:
                assert chunk['start'] == cursor and chunk['end'] > cursor
                assert hashlib.sha256(np.ascontiguousarray(moments[cursor:chunk['end']]).tobytes()).hexdigest() == chunk['sha256']
                cursor = chunk['end']
            assert cursor == done
        else: chunks = []
        start = done
        for slot in range(done, len(needed)):
            value = model.encode_actual(actual_rgb(images[needed[slot]]))
            moments[slot] = value[0].cpu().numpy()
            if (slot + 1) % 32 == 0 or slot + 1 == len(needed):
                moments.flush()
                with partial.open('rb') as file: os.fsync(file.fileno())
                digest = hashlib.sha256(np.ascontiguousarray(moments[start:slot + 1]).tobytes()).hexdigest()
                chunks.append({'start':start, 'end':slot + 1, 'sha256':digest})
                write(CACHE / 'moments_progress.json', {'completed': slot + 1, 'total': len(needed),
                    'chunks':chunks, 'protocol_sha256': identity, 'updated_utc': now()})
                start = slot + 1
        del moments
        record = {'status': 'complete', 'protocol_sha256': identity, 'image_ids': needed,
            'moments_sha256': sha(partial), 'dtype': 'float16', 'shape': [len(needed),8,28,28]}
        write(CACHE / 'moments.pending.json', record); partial.replace(CACHE / 'moments.npy')
        (CACHE / 'moments.pending.json').replace(CACHE / 'moments.json')
    record = read(CACHE / 'moments.json')
    assert record['protocol_sha256'] == identity and record['image_ids'] == needed
    assert sha(CACHE / 'moments.npy') == record['moments_sha256']
    moments = np.load(CACHE / 'moments.npy', mmap_mode='r')
    pooled = np.load(BASE_CACHE / 'vision/pooled.npy', mmap_mode='r')
    for name in ADDITIONS:
        directory = CACHE / name; directory.mkdir(exist_ok=True)
        if (directory / 'future.npy').exists() and (directory / 'manifest.pending.json').exists():
            pending = read(directory / 'manifest.pending.json')
            assert pending['protocol_sha256'] == identity and sha(directory / 'future.npy') == pending['future_sha256']
            (directory / 'manifest.pending.json').replace(directory / 'manifest.json')
        if (directory / 'manifest.json').exists():
            assert read(directory / 'manifest.json')['protocol_sha256'] == identity; continue
        if name != ADDITIONS[0] or model is None:
            if model is not None: del model; torch.cuda.empty_cache()
            model = FrozenNwmWorldModel(name)
        partial = directory / 'future.partial.npy'
        progress = read(directory / 'progress.json') if (directory / 'progress.json').exists() else {'completed': 0}
        done = progress['completed']
        if done: assert partial.exists() and progress['protocol_sha256'] == identity
        values = np.load(partial, mmap_mode='r+') if partial.exists() else np.lib.format.open_memmap(partial,
            mode='w+', dtype=np.float32, shape=(len(roots),4,49,384))
        assert values.shape == (1280,4,49,384) and values.dtype == np.float32
        if done:
            chunks = progress['chunks']; cursor = 0
            for chunk in chunks:
                assert chunk['start'] == cursor and chunk['end'] > cursor
                assert hashlib.sha256(np.ascontiguousarray(values[cursor:chunk['end']]).tobytes()).hexdigest() == chunk['sha256']
                cursor = chunk['end']
            assert cursor == done
        else: chunks = []
        start = done; started = time.monotonic()
        for index in range(done, len(roots)):
            root = roots[index]; assert root['id'] == index
            context = torch.from_numpy(np.array(moments[[slot_of[i] for i in root['four_real_image_ids']]], copy=True))[None]
            prediction = model.predict_movements(context, context_seed=root['context_seed'], sampling_seed=root['diffusion_seed'])
            values[index,:3] = prediction.cpu().numpy()
            values[index,3] = pooled[old_roots[index]['current_image_id']]
            if (index + 1) % 8 == 0 or index + 1 == len(roots):
                values.flush()
                with partial.open('rb') as file: os.fsync(file.fileno())
                digest = hashlib.sha256(np.ascontiguousarray(values[start:index + 1]).tobytes()).hexdigest()
                chunks.append({'start':start, 'end':index + 1, 'sha256':digest})
                progress = {'status': 'predicting', 'model': name, 'completed': index + 1, 'total': len(roots),
                    'chunks':chunks, 'protocol_sha256': identity,
                    'seconds_this_run': time.monotonic() - started, 'updated_utc': now()}
                write(directory / 'progress.json', progress); write(FEATURES_RUN / 'status.json', progress)
                print(json.dumps(progress), flush=True); start = index + 1
        del values
        record = {'status': 'complete', 'protocol_sha256': identity, 'model': name, 'roots': len(roots),
            'shape': [1280,4,49,384], 'future_sha256': sha(partial), 'movement_predictions': 3840, 'STOP_model_calls': 0}
        write(directory / 'manifest.pending.json', record); partial.replace(directory / 'future.npy')
        (directory / 'manifest.pending.json').replace(directory / 'manifest.json')
    write(FEATURES_RUN / 'status.json', {'status': 'cached_pending_verification', 'updated_utc': now()})


if __name__ == '__main__':
    main()
