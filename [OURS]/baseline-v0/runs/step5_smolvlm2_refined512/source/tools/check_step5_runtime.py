"""Real-weight encoder, native/SDPA gradients, CE commands and training profile."""
import argparse
import copy
import gc
import hashlib
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.common import ROOT, RUN, now, read, sha, write
from wm.dino import build_dynamics, build_encoder, preprocessing
from wm.features import IndexedImages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-updates', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(5)
    encoder = build_encoder()
    db = sqlite3.connect(f"file:{RUN / 'data_index/images.sqlite'}?mode=ro&immutable=1", uri=True)
    raw_images = []
    for number in (0, 31, 80, 200):
        path, expected = db.execute('SELECT path,sha256 FROM images WHERE id=?', (number,)).fetchone()
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        raw_images.append(Image.open(io.BytesIO(raw)).copy())
    cpu_prepared = torch.stack([preprocessing()(image) for image in raw_images])
    rgb = torch.stack([to_tensor(image) for image in raw_images]).cuda()
    with torch.inference_mode():
        native_prepared = encoder.preprocess(rgb)
        delta = (native_prepared.cpu() - cpu_prepared).abs()
        assert delta.max() < 2e-5
        exact = encoder.encoder.forward_features(cpu_prepared.cuda())['x_norm_patchtokens']
        with torch.autocast('cuda', dtype=torch.float16):
            reduced = encoder.encoder.forward_features(cpu_prepared.cuda())['x_norm_patchtokens']
        bridge = encoder(rgb, pooled=False)
        precision_mse = (exact - reduced).square().mean().item()
        preprocessing_feature_mse = (bridge - exact).square().mean().item()
        storage_mse = (exact.to(torch.float16).float() - exact).square().mean().item()
        print(json.dumps({'CPU_GPU_preprocessing_max_abs_difference': delta.max().item(),
                          'unused_float16_encoder_mse': precision_mse, 'float32_bridge_mse': preprocessing_feature_mse,
                          'storage_float16_mse': storage_mse}), flush=True)
        assert preprocessing_feature_mse < 1e-7 and storage_mse < 5e-7
    frozen_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    rows = np.load(RUN / 'data_index/train_samples.npy', mmap_mode='r')
    slots = np.random.default_rng(5).choice(len(rows), 32, replace=False)
    samples = rows[slots]
    frame_ids = samples[:, 1:5].reshape(-1)
    unique, inverse = np.unique(frame_ids, return_inverse=True)
    images = IndexedImages()
    pixels = torch.stack([images[int(i)][1] for i in unique]).cuda()
    with torch.inference_mode():
        encoded = encoder.encoder.forward_features(pixels)['x_norm_patchtokens'].to(torch.float16)
    features = encoded[torch.tensor(inverse, device='cuda')].reshape(32, 4, 196, 384).detach().clone()
    actions = torch.tensor(samples[:, 6:9], device='cuda', dtype=torch.long)
    model = build_dynamics('transfer', sdpa=False).eval()
    fast = build_dynamics('transfer', sdpa=True).eval()
    fresh = build_dynamics('fresh', sdpa=True).eval()
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), fast.state_dict().values()))
    assert torch.equal(model.command_embedding.weight, fresh.command_embedding.weight)
    assert not torch.equal(model.predictor.pos_embedding, fresh.predictor.pos_embedding)
    parity = []
    for history in (1, 2, 3):
        x = features[:2, 3 - history:3].float()
        past = actions[:2, 3 - history:2]
        native = model(x, past, actions[:2, 2])
        optimized = fast(x, past, actions[:2, 2])
        torch.testing.assert_close(native, optimized, atol=3e-5, rtol=3e-4)
        parity.append({'history': history, 'max_abs_error': (native - optimized).abs().max().item()})
    native.square().mean().backward()
    optimized.square().mean().backward()
    gradient_error = 0.0
    for (name, p), (_, q) in zip(model.named_parameters(), fast.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-3)
        gradient_error = max(gradient_error, (p.grad - q.grad).abs().max().item())
    assert fast.command_embedding.weight.grad.abs().sum() > 0
    # Check the block-causal frame mask directly, including an adversarial future view.
    with torch.no_grad():
        packed = torch.randn(1, 588, 404, device='cuda')
        original = fast.predictor(packed)
        perturbed = packed.clone(); perturbed[:, 392:] += 10 * torch.randn_like(perturbed[:, 392:])
        changed = fast.predictor(perturbed)
        torch.testing.assert_close(original[:, :392], changed[:, :392], atol=0, rtol=0)
    del model, fast, fresh, native, optimized, packed, original, changed
    gc.collect(); torch.cuda.empty_cache()
    profile = []
    for batch_size in (16, 32, 64):
        model = build_dynamics('transfer').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
        repeat = max(1, batch_size // len(features))
        x = features.repeat(repeat, 1, 1, 1)[:batch_size]
        a = actions.repeat(repeat, 1)[:batch_size]
        torch.cuda.reset_peak_memory_stats()
        losses = []
        started = None
        for step in range(args.profile_updates + 5):
            if step == 5:
                torch.cuda.synchronize(); started = time.time()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x[:, :3], a[:, :2], a[:, 2])
                loss = (pred.float() - x[:, 3].float()).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        torch.cuda.synchronize()
        seconds = time.time() - started
        profile.append({'batch_size': batch_size, 'timed_updates': args.profile_updates, 'seconds': seconds,
                        'samples_per_second': batch_size * args.profile_updates / seconds,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'first_loss': losses[0], 'last_loss': losses[-1]})
        assert losses[-1] < losses[0]
        print(json.dumps(profile[-1]), flush=True)
        del model, optimizer, x, a, pred, loss
        gc.collect(); torch.cuda.empty_cache()
    assert all(torch.equal(frozen_state[k], v.detach().cpu()) for k, v in encoder.state_dict().items())
    assert not any(p.requires_grad for p in encoder.parameters())
    report = {'status': 'passed', 'completed_utc': now(), 'encoder_frozen_and_unchanged': True,
              'native_encoder_weights_equal_pinned_HF': read(RUN / 'assets/encoder_weight_equivalence.json'),
              'CPU_GPU_preprocessing_max_abs_difference': delta.max().item(),
              'unused_float16_encoder_mse_vs_float32': precision_mse,
              'encoder_compute': 'float32', 'float16_storage_mse_vs_float32': storage_mse,
              'preprocessed_encoder_mse_vs_existing_bridge': preprocessing_feature_mse,
              'native_SDPA_output_parity': parity, 'native_SDPA_gradient_max_abs_error': gradient_error,
              'block_causal_future_view_test': 'passed', 'same_new_CE_command_embedding': True,
              'training_profile': profile, 'profile_is_discarded': True,
              'policy_identity_sha256': read(RUN / 'data_index/manifest.json')['selected_policy_identity_sha256'],
              'source_sha256': {name: sha(ROOT / name) for name in ['wm/dino.py', 'wm/features.py', 'tools/check_step5_runtime.py', 'Scorer/world_models.py', 'Scorer/interfaces.py']}}
    write(RUN / 'check_runtime.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
