"""Profile the next repeat after Step 10; validate native precision and exact resume."""
import copy
import gc
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from Scorer.world_models import build_nwm_diffusion, sample_nwm_latents
from nwm_training.common import INDEX, POLICY_ID, RUN, VAE_DIRECTORY, now, read, sha, verify_assets, verify_index, write
from nwm_training.data import IndexedVAEImages, posterior_moments, sample_posterior
from nwm_training.engine import AutocastDenoiser, configure, diffusion, losses, update
from nwm_training.model import LATENT_SCALE, build_model, build_vae, command_table, conditioning
from wm.training import restore_rng, save_checkpoint


def checkpoint(model, ema, optimizer):
    return {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}


def main():
    predecessor = ROOT / 'runs/step10_smolvlm2_refined512_dino_pair_v1/result.json'
    assert read(predecessor)['status'] == 'complete', 'Keep GPU profiling after the matched DINO evaluation'
    configure(); verify_assets(); verify_index()
    cpu = read(RUN / 'check_cpu.json'); assert cpu['status'] == 'passed'
    for name, digest in cpu['source_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert torch.cuda.device_count() == 1
    rows = np.load(INDEX / 'train_samples.npy', mmap_mode='r')
    order = np.load(INDEX.parent / 'sample_order.npy', mmap_mode='r')
    selected = np.array(rows[order[:64]], copy=True)
    ids = sorted(set(selected[:, :5].flatten().tolist()))
    images = IndexedVAEImages()
    loader = DataLoader(Subset(images, ids), batch_size=32, num_workers=8, pin_memory=True)
    pixels = {}; moments_by_id = {}
    for image_ids, batch in loader:
        pixels.update({int(i): value for i, value in zip(image_ids, batch)})
    sample_pixels = torch.stack([pixels[i] for i in ids[:64]]).cuda()
    assert len(sample_pixels) == 64
    vae = build_vae('cuda:0'); profile = []
    with torch.inference_mode():
        serial = torch.cat([posterior_moments(vae, x[None]) for x in sample_pixels[:4]])
        for size in (16, 32, 64):
            posterior_moments(vae, sample_pixels[:size]); torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
            for _ in range(3): values = posterior_moments(vae, sample_pixels[:size])
            torch.cuda.synchronize(); seconds = (time.perf_counter() - started) / 3
            drift = (values[:4] - serial).square().mean().item()
            assert drift < 1e-7, (size, drift)
            profile.append({'batch_size': size, 'seconds_per_batch': seconds, 'images_per_second': size / seconds,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'batch1_moments_mse': drift})
        cache_size = max(profile, key=lambda x: x['images_per_second'])['batch_size']
        for start in range(0, len(ids), cache_size):
            chunk = ids[start:start + cache_size]
            value = posterior_moments(vae, torch.stack([pixels[i] for i in chunk]).cuda()).half().cpu()
            moments_by_id.update(dict(zip(chunk, value)))
    vae.cpu(); del pixels, sample_pixels, serial, values; gc.collect(); torch.cuda.empty_cache()
    batch = {'moments': torch.stack([torch.stack([moments_by_id[int(i)] for i in row[:5]]) for row in selected]),
             'action': torch.tensor(selected[:, 8], dtype=torch.long)}
    model = build_model('transfer', 'cuda:0').train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    process = diffusion(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
    # Independently follow the pinned train.py data flow using the VAE library's
    # actual posterior class, then compare the loss to our cache-based adapter.
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    small = {k: v[:2] for k, v in batch.items()}
    torch.manual_seed(515)
    ours = losses(model, process, small, torch.device('cuda:0'))
    torch.manual_seed(515)
    packed = small['moments'].cuda().float().flatten(0, 1)
    latent = DiagonalGaussianDistribution(packed).sample().mul_(LATENT_SCALE).unflatten(0, (2, 5))
    target = latent[:, 4:5].flatten(0, 1)
    kwargs = conditioning(latent[:, :4], small['action'].cuda())
    timestep = torch.randint(0, process.num_timesteps, (len(target),), device='cuda:0')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        native = process.training_losses(model, target, timestep, model_kwargs=kwargs)
    for key in ours: assert torch.equal(ours[key], native[key]), key
    del ours, native, latent, target, packed, kwargs; gc.collect(); torch.cuda.empty_cache()

    torch.manual_seed(516)
    update(model, ema, optimizer, process, batch)
    path = RUN / 'diagnostics/restart.pt'; save_checkpoint(path, checkpoint(model, ema, optimizer))
    expected_loss, _ = update(model, ema, optimizer, process, batch)
    expected_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    expected_ema = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True); ema.load_state_dict(saved['ema'], strict=True)
    optimizer.load_state_dict(saved['optimizer']); restore_rng(saved)
    resumed_loss, _ = update(model, ema, optimizer, process, batch)
    assert resumed_loss == expected_loss
    for name, value in model.state_dict().items(): assert torch.equal(value.cpu(), expected_model[name]), name
    for name, value in ema.state_dict().items(): assert torch.equal(value.cpu(), expected_ema[name]), name
    del saved, expected_model, expected_ema; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(); times = []; train_losses = []
    for _ in range(6):
        torch.cuda.synchronize(); started = time.perf_counter()
        values, norm = update(model, ema, optimizer, process, batch)
        torch.cuda.synchronize(); times.append(time.perf_counter() - started); train_losses.append(values)
    train_profile = {'batch_size': 64, 'steady_seconds_per_update': float(np.mean(times[1:])),
        'samples_per_second': 64 / float(np.mean(times[1:])), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'profile_losses': train_losses, 'full_pass_updates': int(np.ceil(len(rows) / 64)),
        'estimated_full_pass_compute_seconds': int(np.ceil(len(rows) / 64)) * float(np.mean(times[1:]))}
    del optimizer, ema; model.eval().requires_grad_(False); gc.collect(); torch.cuda.empty_cache()
    context = sample_posterior(batch['moments'][:1, :4].cuda(), generator=torch.Generator(device='cuda').manual_seed(517))
    kwargs = conditioning(context, torch.tensor([0], device='cuda'))
    denoiser = AutocastDenoiser(model).eval(); noise = torch.randn(1, 4, 28, 28, device='cuda')
    timestep = torch.tensor([200], device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(noise, timestep, **kwargs)
    with torch.inference_mode(): observed = denoiser(noise, timestep, **kwargs)
    assert torch.equal(expected, observed)
    sampling = []; predictions = {}
    for steps in (50, 250):
        sampler = build_nwm_diffusion(ROOT / 'third_party/nwm/diffusion', sampling_steps=steps)
        torch.cuda.synchronize(); started = time.perf_counter()
        value = sample_nwm_latents(denoiser, sampler, kwargs, seed=518)
        torch.cuda.synchronize()
        sampling.append({'steps': steps, 'seconds_one_candidate': time.perf_counter() - started})
        predictions[steps] = value
    sampler = build_nwm_diffusion(ROOT / 'third_party/nwm/diffusion', sampling_steps=50)
    repeated = sample_nwm_latents(denoiser, sampler, kwargs, seed=518)
    assert torch.equal(repeated, predictions[50])
    vae.cuda()
    with torch.inference_mode():
        rgb = (vae.decode(repeated.float() / LATENT_SCALE).sample.clamp(-1, 1) + 1) / 2
    from wm.dino import build_encoder
    encoder = build_encoder()
    with torch.inference_mode(): visual = encoder(rgb)
    assert visual.shape == (1, 49, 384) and torch.isfinite(visual).all()
    assert not any(p.requires_grad for p in vae.parameters())
    # Exercise the actual packaged runtime before committing to a long fit.
    from nwm_training.frozen import FrozenNwmWorldModel
    asset = read(RUN / 'assets/selection.json'); bundle = RUN / 'diagnostics/preflight_bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset['extracted_ema'], bundle / 'model.pt')
    encoder_folder = ROOT / 'runs/step5_smolvlm2_refined512/frozen/encoder'
    config = {'policy_identity_sha256': POLICY_ID, 'STOP_bias': -.25, 'minimum_real_observations': 4,
        'latent_scale': LATENT_SCALE, 'candidate_order': ['FORWARD','LEFT','RIGHT'], 'sampling_steps': 50,
        'command_table': command_table().tolist(), 'relative_time': 1 / 128,
        'VAE_directory': str(VAE_DIRECTORY), 'DINO_encoder_directory': str(encoder_folder),
        'shared_files_sha256': {str(VAE_DIRECTORY / n): sha(VAE_DIRECTORY / n) for n in ('config.json','diffusion_pytorch_model.safetensors')},
        'runtime_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'nwm_training').glob('*.py'))},
        'scope': 'Unadapted pretrained runtime preflight only; not selected deployment weights'}
    write(bundle / 'config.json', config)
    write(bundle / 'file_sha256.json', {n: sha(bundle / n) for n in ('model.pt','config.json')})
    probe = FrozenNwmWorldModel('nwm_transfer', encoder=encoder, bundle=bundle)
    from PIL import Image
    from torchvision.transforms import ToTensor
    image_id = int(selected[0, 0]); images[image_id]  # Hash-check before opening the real RGB.
    image_path = images.connection.execute('SELECT path FROM images WHERE id=?', (image_id,)).fetchone()[0]
    with Image.open(image_path) as image: actual_rgb = ToTensor()(image.convert('RGB'))[None]
    actual_moments = probe.encode_actual(actual_rgb)
    actual_cache_mse = (actual_moments.cpu().float() - moments_by_id[image_id][None].float()).square().mean().item()
    assert actual_cache_mse < 1e-6
    assert torch.equal(actual_moments, probe.encode_actual(actual_rgb))
    before = torch.get_rng_state().clone(); context_moments = batch['moments'][:1, :4].clone()
    original = context_moments.clone(); torch.cuda.synchronize(); tick = time.perf_counter()
    three = probe.predict_movements(context_moments, context_seed=520, sampling_seed=521)
    torch.cuda.synchronize(); three_seconds = time.perf_counter() - tick
    again = probe.predict_movements(context_moments, context_seed=520, sampling_seed=521)
    assert torch.equal(three, again) and torch.equal(original, context_moments)
    assert torch.equal(before, torch.get_rng_state())
    try: probe.predict_movements(context_moments[:, 1:], context_seed=520, sampling_seed=521)
    except AssertionError: pass
    else: raise AssertionError('Runtime accepted fewer than four real observations')
    record = {'status': 'passed', 'completed_utc': now(), 'predecessor_result_sha256': sha(predecessor),
        'assets_sha256': sha(RUN / 'assets/selection.json'), 'CPU_check_sha256': sha(RUN / 'check_cpu.json'),
        'GPU': torch.cuda.get_device_name(0), 'no_E_used_for_fitting_or_choices': True,
        'native_posterior_and_training_loss_exact': True, 'optimizer_and_EMA_resume_bit_exact': True,
        'native_FP32_residual_BF16_denoiser_exact': True, 'sampling_repeat_exact': True,
        'complete_4channel_sample_VAE_DINO_bridge_finite': True, 'bridge_shape': list(visual.shape),
        'packaged_three_candidate_runtime_repeat_exact': True, 'real_context_immutable': True,
        'packaged_three_candidate_seconds': three_seconds, 'warmup_guard_checked': True,
        'live_VAE_vs_training_cache_moments_MSE': actual_cache_mse, 'live_VAE_batch1_repeat_exact': True,
        'VAE_cache_profiles': profile, 'selected_cache_batch_size': cache_size, 'training_profile': train_profile,
        'sampling_profiles': sampling, 'selected_sampling_steps': 50,
        'sampling_budget_rationale': 'Declared one-action pilot budget: 50 complete ancestral steps, 5x fewer denoiser calls than source 250. Matched across transfer/fresh; no accuracy selection from E.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'nwm_training').glob('*.py')) + [Path(__file__).resolve()]}}
    write(RUN / 'check_gpu.json', record); print(record, flush=True)


if __name__ == '__main__':
    main()
