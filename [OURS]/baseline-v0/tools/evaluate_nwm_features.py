"""Matched C feature diagnostics after checkpoint selection; no navigation claim."""
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from Scorer.interfaces import pool_dino_patches
from nwm_training.common import CACHE, INDEX, POLICY_ID, RUN, now, read, sha, write
from nwm_training.engine import configure
from nwm_training.frozen import FrozenNwmWorldModel
from nwm_training.protocol import verify_protocol
from wm.features import CACHE as DINO_CACHE
from wm.frozen import FrozenDinoWorldModel


def seeds(identity):
    prefix = f"NWM_CE_v1/5/C/{identity['dataset']}/{identity['episode_id']}/t{identity['t']}"
    return [int.from_bytes(hashlib.sha256((prefix + '/' + kind).encode()).digest()[:8], 'little') % (2**63)
            for kind in ('context', 'diffusion')]


def main():
    configure(); plan = verify_protocol(); assert read(RUN / 'freeze_verification.json')['status'] == 'passed'
    old = ROOT / 'runs/step5_smolvlm2_refined512'
    seal = read(old / 'cache_seal.json'); stat = (DINO_CACHE / 'features.npy').stat()
    assert [stat.st_size, stat.st_mtime_ns] == seal['features_size_mtime_ns']
    assert sha(DINO_CACHE / 'manifest.json') == seal['cache_manifest_sha256']
    raw = np.load(DINO_CACHE / 'features.npy', mmap_mode='r')
    moments = np.load(CACHE / 'posteriors.npy', mmap_mode='r')
    samples = np.load(INDEX / 'validation_samples.npy', mmap_mode='r')
    identities = read(INDEX / 'validation_identities.json')
    assert len(samples) == len(identities) == 3072
    models = ['nwm_transfer', 'cdit_fresh_dynamics', 'dinowm_transfer', 'dinov2_fresh_dynamics']
    reports = {}
    for name in models:
        nwm = name in models[:2]
        folder = RUN / 'frozen' / name if nwm else old / 'frozen' / name
        model_sha = sha(folder / 'model.pt'); directory = RUN / 'quality' / name
        if (directory / 'summary.json').exists():
            report = read(directory / 'summary.json')
            assert report['model_sha256'] == model_sha
            for path, digest in report['row_sha256'].items(): assert sha(directory / 'rows' / path) == digest
            reports[name] = report; continue
        model = FrozenNwmWorldModel(name) if nwm else FrozenDinoWorldModel(folder, expected_policy_identity=POLICY_ID)
        records = []; started = time.monotonic()
        with torch.inference_mode():
            for slot, (row, identity) in enumerate(zip(samples, identities)):
                context_seed, sampling_seed = seeds(identity)
                path = directory / 'rows' / f'{slot:06d}.json'
                marker = {'slot': slot, 'model_sha256': model_sha, 'protocol_sha256': sha(RUN / 'protocol.json'),
                    'index_manifest_sha256': sha(INDEX / 'manifest.json'), 'identity': identity,
                    'context_seed': context_seed if nwm else None, 'sampling_seed': sampling_seed if nwm else None}
                if path.exists():
                    record = read(path)
                    assert all(record[k] == value for k, value in marker.items())
                else:
                    target = pool_dino_patches(torch.from_numpy(np.array(raw[row[4]], copy=True)).float()[None]).cuda()
                    current = pool_dino_patches(torch.from_numpy(np.array(raw[row[3]], copy=True)).float()[None]).cuda()
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    if nwm:
                        context = torch.from_numpy(np.array(moments[row[:4]], copy=True))[None]
                        prediction = model.predict_movements(context, context_seed=context_seed, sampling_seed=sampling_seed)
                    else:
                        context = torch.from_numpy(np.array(raw[row[1:4]], copy=True))[None].cuda()
                        actions = torch.tensor(row[6:8], dtype=torch.long, device='cuda')[None]
                        values = []
                        for action in range(3):
                            value = model.predict(context, actions, torch.tensor([action], device='cuda'),
                                observations_seen=4, encoder_identity=seal['encoder_identity_sha256'])
                            values.append(pool_dino_patches(value.float()))
                        prediction = torch.cat(values)
                    torch.cuda.synchronize(); seconds = time.perf_counter() - tick
                    errors = (prediction.float() - target).square().mean((1, 2)).cpu().tolist()
                    assert all(np.isfinite(x) for x in errors)
                    record = {**marker, 'dataset': int(row[9]), 'action': int(row[8]),
                        'matched_action_pooled_mse': errors[int(row[8])],
                        'all_commands_against_single_observed_target': errors,
                        'copy_current_pooled_mse': (current - target).square().mean().item(),
                        'prediction_seconds_three_candidates': seconds}
                    write(path, record)
                records.append(record)
                if (slot + 1) % 32 == 0:
                    progress = {'status': 'evaluating', 'model': name, 'completed': slot + 1, 'total': len(samples),
                        'seconds_this_run': time.monotonic() - started, 'updated_utc': now()}
                    write(RUN / 'quality/progress.json', progress); print(json.dumps(progress), flush=True)
        groups = {}
        for dataset, label in enumerate(('r2r','rxr')):
            for action, command in enumerate(('FORWARD','LEFT','RIGHT')):
                group = [r for r in records if r['dataset'] == dataset and r['action'] == action]
                assert len(group) == 512
                groups[f'{label}/{command}'] = {'samples': len(group),
                    'pooled_mse': float(np.mean([r['matched_action_pooled_mse'] for r in group])),
                    'copy_current_pooled_mse': float(np.mean([r['copy_current_pooled_mse'] for r in group]))}
        report = {'status': 'complete', 'model': name, 'model_sha256': model_sha, 'samples': len(records),
            'macro_pooled_mse': float(np.mean([v['pooled_mse'] for v in groups.values()])),
            'copy_current_macro_pooled_mse': float(np.mean([v['copy_current_pooled_mse'] for v in groups.values()])),
            'by_dataset_action': groups, 'group': 'C', 'metric_shape': [49,384],
            'prediction_mean_seconds_three_candidates': float(np.mean([r['prediction_seconds_three_candidates'] for r in records])),
            'latency_scope': 'Prediction from cached context only, excluding context encoding and policy; three candidates together',
            'selection_effect': 'None; selected checkpoints and sampling budget remain fixed',
            'diagnostic_limit': 'Expert-path validation without collisions; alternate-command errors use one observed target and are not counterfactual labels',
            'row_sha256': {p.name: sha(p) for p in sorted((directory / 'rows').glob('*.json'))}}
        write(directory / 'summary.json', report); reports[name] = report
        del model; gc.collect(); torch.cuda.empty_cache()
    result = {'status': 'complete', 'completed_utc': now(), 'protocol_sha256': sha(RUN / 'protocol.json'),
        'models': reports, 'shared_samples': 3072, 'shared_visual_shape': [49,384], 'no_E_used': True,
        'scope': 'Common feature prediction diagnostic; no navigation success claim or post-selection retuning'}
    write(RUN / 'quality/result.json', result)
    write(RUN / 'quality/progress.json', {'status': 'complete', 'updated_utc': now()})
    print('Common C feature evaluation complete', flush=True)


if __name__ == '__main__':
    main()
