"""Frozen NWM/CDiT predictions with shared real context and one sample per action."""
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torchvision import transforms

from Scorer.world_models import build_nwm_diffusion, sample_nwm_latents
from nwm_training.common import POLICY_ID, ROOT, RUN, read, sha
from nwm_training.data import posterior_moments, sample_posterior
from nwm_training.engine import AutocastDenoiser
from nwm_training.model import LATENT_SCALE, command_table, conditioning, native_model
from wm.dino import build_encoder


@contextmanager
def precision_for_vae():
    with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=False), \
            torch.backends.cudnn.flags(enabled=True, benchmark=False,
                deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
        yield


@contextmanager
def precision_for_common_encoder():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(False)
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.use_deterministic_algorithms(old_deterministic)


class FrozenNwmWorldModel:
    def __init__(self, trial, device='cuda:0', *, encoder=None, bundle=None):
        assert trial in ('nwm_transfer', 'cdit_fresh_dynamics')
        folder = Path(bundle) if bundle is not None else RUN / 'frozen' / trial
        for name, digest in read(folder / 'file_sha256.json').items(): assert sha(folder / name) == digest
        self.config = read(folder / 'config.json')
        for name, digest in self.config['runtime_source_sha256'].items(): assert sha(ROOT / name) == digest, name
        assert self.config['policy_identity_sha256'] == POLICY_ID and self.config['STOP_bias'] == -.25
        assert self.config['minimum_real_observations'] == 4 and self.config['latent_scale'] == LATENT_SCALE
        assert self.config['candidate_order'] == ['FORWARD','LEFT','RIGHT']
        assert self.config['command_table'] == command_table().tolist() and self.config['relative_time'] == 1 / 128
        assert self.config['sampling_steps'] == 50
        for path, digest in self.config['shared_files_sha256'].items(): assert sha(path) == digest, path
        self.device = torch.device(device)
        self.model = native_model().to(self.device).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        self.denoiser = AutocastDenoiser(self.model).eval().requires_grad_(False)
        self.diffusion = build_nwm_diffusion(ROOT / 'third_party/nwm/diffusion', sampling_steps=50)
        dependencies = str(ROOT / 'env/nwm_dependencies')
        if dependencies not in sys.path: sys.path.insert(0, dependencies)
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(self.config['VAE_directory'], local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32).to(self.device).eval().requires_grad_(False)
        self.encoder = encoder if encoder is not None else build_encoder(device, assets=Path(self.config['DINO_encoder_directory']))
        assert not any(p.requires_grad for p in self.encoder.parameters())
        self.preprocess = transforms.Compose([transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224), transforms.Normalize([.5] * 3, [.5] * 3)])

    @torch.inference_mode()
    def encode_actual(self, rgb):
        """One real RGB[1,3,H,W] in [0,1]; return rounded posterior moments."""
        assert rgb.ndim == 4 and rgb.shape[:2] == (1, 3)
        assert torch.isfinite(rgb).all() and torch.all((rgb >= 0) & (rgb <= 1))
        with precision_for_vae():
            pixels = self.preprocess(rgb.detach().to('cpu', dtype=torch.float32)).to(self.device)
            return posterior_moments(self.vae, pixels).half()

    @torch.inference_mode()
    def predict_movements(self, moments, *, context_seed, sampling_seed):
        """Three candidate rows; one shared sampled real context, no STOP call.

        Both seeds are explicit and common to the paired dynamics variants.
        Candidate sampling is batched F/L/R in that fixed order. DINO encoding
        stays batch one to match the existing frozen visual representation.
        """
        assert moments.shape == (1, 4, 8, 28, 28)
        assert moments.dtype == torch.float16
        assert type(context_seed) is int and 0 <= context_seed < 2**63
        original = moments.clone()
        generator = torch.Generator(device=self.device).manual_seed(context_seed)
        context = sample_posterior(moments.to(self.device), generator=generator).expand(3, -1, -1, -1, -1)
        kwargs = conditioning(context, torch.arange(3, device=self.device, dtype=torch.long))
        completed = sample_nwm_latents(self.denoiser, self.diffusion, kwargs, seed=sampling_seed)
        with precision_for_vae():
            rgb = (self.vae.decode(completed.float() / LATENT_SCALE).sample.clamp(-1, 1) + 1) / 2
        with precision_for_common_encoder():
            features = torch.cat([self.encoder(image[None]) for image in rgb], dim=0)
        assert features.shape == (3, 49, 384) and torch.isfinite(features).all()
        assert torch.equal(moments, original)
        return features
