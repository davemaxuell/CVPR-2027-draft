"""Verify source command semantics, real VAE posteriors and native denoising on CPU."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

from Scorer.world_models import (CheckedCDiTDenoiser, NWMVisualBridge,
                                 build_nwm_diffusion, sample_nwm_latents)
from nwm_training.common import RUN, now, sha, verify_assets, verify_index, write
from nwm_training.data import IndexedVAEImages, posterior_moments, sample_posterior
from nwm_training.model import (LATENT_SCALE, NOMINAL_COMMANDS, build_model, build_vae,
                                command_table, conditioning)


def main():
    torch.set_num_threads(4)
    assets = verify_assets(); verify_index()
    # Execute the exact inspected upstream normalization function independently.
    tree = ast.parse((ROOT / 'third_party/nwm/misc.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'normalize_data')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}; exec(compile(module, '<pinned native normalize_data>', 'exec'), namespace)
    config = yaml.safe_load((ROOT / 'third_party/nwm/config/data_config.yaml').read_text())
    stats = {k: np.asarray(v) for k, v in config['action_stats'].items()}
    expected = np.array(NOMINAL_COMMANDS)
    expected[:, :2] = namespace['normalize_data'](expected[:, :2] / .25, stats)
    np.testing.assert_allclose(command_table().numpy(), expected, rtol=0, atol=3e-8)

    vae = build_vae(); indexed = IndexedVAEImages(end=1); number, pixels = indexed[0]
    image_path = indexed.connection.execute('SELECT path FROM images WHERE id=?', (number,)).fetchone()[0]
    with Image.open(image_path) as image:
        rgb = ToTensor()(image.convert('RGB'))[None]
    bridge = NWMVisualBridge(vae, torch.nn.Identity())
    assert torch.equal(bridge.preprocess(rgb), pixels[None])
    with torch.inference_mode():
        distribution = vae.encode(pixels[None]).latent_dist
        moments = posterior_moments(vae, pixels[None])
        native = distribution.sample(generator=torch.Generator().manual_seed(25)) * LATENT_SCALE
        sampled = sample_posterior(moments, generator=torch.Generator().manual_seed(25))
        assert torch.equal(native, sampled)
        rounded = sample_posterior(moments.half(), generator=torch.Generator().manual_seed(25))
        quantization_mse = (native - rounded).square().mean().item()
        assert quantization_mse < 1e-5
        reconstructed = vae.decode(native / LATENT_SCALE).sample
        assert reconstructed.shape == pixels[None].shape and torch.isfinite(reconstructed).all()
    assert not vae.training and not any(p.requires_grad for p in vae.parameters())

    model = build_model('transfer').eval()
    assert all(p.requires_grad for p in model.parameters())
    context = sampled[:, None].expand(-1, 4, -1, -1, -1).clone()
    inputs = conditioning(context, torch.tensor([0]))
    assert set(inputs) == {'x_cond', 'y', 'rel_t'}
    wrapper = CheckedCDiTDenoiser(model).eval()
    noise = torch.randn(1, 4, 28, 28, generator=torch.Generator().manual_seed(26))
    t = torch.tensor([512])
    with torch.inference_mode():
        direct = model(noise, t, **inputs)
        checked = wrapper(noise, t, **inputs)
    assert direct.shape == (1, 8, 28, 28) and torch.isfinite(direct).all()
    assert torch.equal(direct, checked)
    diffusion = build_nwm_diffusion(ROOT / 'third_party/nwm/diffusion', sampling_steps=2)
    before = torch.get_rng_state().clone()
    result = sample_nwm_latents(wrapper, diffusion, inputs, seed=27)
    assert torch.equal(before, torch.get_rng_state())
    second = sample_nwm_latents(wrapper, diffusion, inputs, seed=27)
    assert torch.equal(result, second) and result.shape == noise.shape
    full = build_nwm_diffusion(ROOT / 'third_party/nwm/diffusion', sampling_steps=1000)
    losses = full.training_losses(model, sampled, t, model_kwargs=inputs, noise=noise)
    assert set(losses) == {'vb', 'mse', 'loss'}
    assert torch.isfinite(losses['loss']).all()
    assert torch.equal(losses['loss'], losses['vb'] + losses['mse'])
    losses['loss'].mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in vae.parameters())
    record = {'status': 'passed', 'completed_utc': now(), 'assets_sha256': sha(RUN / 'assets/selection.json'),
        'no_GPU_used': True, 'no_E_data_used': True, 'pinned_native_command_normalization_matches': True,
        'CE_preprocessing_matches_existing_bridge': True,
        'VAE_native_posterior_sampling_exact': True, 'posterior_fp16_sample_mse': quantization_mse,
        'pretrained_denoiser_native_output_exact': True, 'native_epsilon_plus_variance_objective_and_gradients_finite': True,
        'seeded_sampler_repeatable_and_rng_preserving': True, 'sampler_test_steps': 2,
        'sampler_test_scope': 'Interface test only; production sampling budget remains pending GPU profile',
        'VAE_frozen': True, 'source_sha256': {str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / 'nwm_training').glob('*.py')) + [Path(__file__).resolve()]}}
    write(RUN / 'check_cpu.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
