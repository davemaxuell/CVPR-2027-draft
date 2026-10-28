"""Checked boundaries for DINO-WM and NWM/CDiT; no automatic weight downloads."""
import hashlib
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

if __package__:
    from .interfaces import VisualContract, action_ids, pool_dino_patches, require_finite, require_float
else:
    from interfaces import VisualContract, action_ids, pool_dino_patches, require_finite, require_float

DINO_VIT_SHA256 = '60c77bdb03f3b2d565bfdb1695c9c193a802236763d33a03e79ed382b51f0670'
NWM_MODELS_SHA256 = '3758b07bbe7327ae48df600232375b33b480fbb18aa7f28957f1152ba6bd2a6c'
NWM_DIFFUSION_SHA256 = {
    '__init__.py': 'ece965de63cced15cf438f8cfdba27720c675b44217f16a61d43bd09c2d6abba',
    'respace.py': '43e437fb4428b8162759886b333ed9f416605d824c0f48de1a61e41ec14eebff',
    'gaussian_diffusion.py': 'b526564245bfdb06ace88cd3c8fc6be759ec5a6b2c2500327f73157d6d9d7721',
    'diffusion_utils.py': '62e73ff4c5e74429ab8589fe8e0fac81a8a9235a02cf2ab97c80797d0819bba7',
}

def build_nwm_diffusion(source_dir, *, sampling_steps):
    """Pinned official epsilon/learned-variance sampler. Steps must be explicit."""
    if type(sampling_steps) is not int or not 2 <= sampling_steps <= 1000:
        raise ValueError('Declare 2..1000 diffusion steps; no hidden deployment budget')
    directory = Path(source_dir).resolve()
    for name, expected in NWM_DIFFUSION_SHA256.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Upstream diffusion source changed: {name}')
    package = '_checked_nwm_diffusion'
    if package in sys.modules:
        if Path(sys.modules[package].__file__).resolve() != directory / '__init__.py':
            raise ValueError('A different diffusion source is already loaded')
        module = sys.modules[package]
    else:
        spec = importlib.util.spec_from_file_location(package, directory / '__init__.py',
                                                     submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    return module.create_diffusion(str(sampling_steps), learn_sigma=True)

def load_pinned_source(path, sha256, *, portable_dino=False):
    """Load inspected local source. The only DINO edit replaces its CUDA mask
    assignment with a nonpersistent buffer; original file and state keys stay intact.
    """
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f'Upstream source hash changed: review {path} before loading')
    source = data.decode('utf-8')
    if portable_dino:
        old = "self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')"
        new = "self.register_buffer('bias', generate_mask_matrix(NUM_PATCHES, NUM_FRAMES), persistent=False)"
        if source.count(old) != 1:
            raise ValueError('Pinned DINO mask patch no longer matches')
        source = source.replace(old, new)
    module = types.ModuleType('checked_' + path.stem)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module

def build_dino_predictor(source_path, *, contract=VisualContract(), history=3,
                         action_width=10, proprio_width=10, **overrides):
    module = load_pinned_source(source_path, DINO_VIT_SHA256, portable_dino=True)
    settings = dict(num_patches=contract.raw_patches, num_frames=history,
                    dim=contract.feature_width + action_width + proprio_width,
                    depth=6, heads=16, mlp_dim=2048, dropout=.1, emb_dropout=0., pool='mean')
    settings.update(overrides)
    return module.ViTPredictor(**settings)

class DinoCommandDynamics(nn.Module):
    """CE command adapter for the feature-concatenation DINO predictor.

    Replace source-task command/proprio encoders: learn a CE command embedding,
    reserve zero-valued proprio channels, and transfer only compatible predictor
    weights. CE retraining is required; this is not zero-shot navigation.
    """
    def __init__(self, predictor, contract=VisualContract(), history=3, action_width=10, proprio_width=10):
        super().__init__()
        self.predictor, self.contract, self.history = predictor, contract, history
        self.action_width, self.proprio_width = action_width, proprio_width
        expected = (1, history * contract.raw_patches, contract.feature_width + action_width + proprio_width)
        if tuple(predictor.pos_embedding.shape) != expected:
            raise ValueError(f'Predictor positions/width incompatible: expected {expected}')
        self.command_embedding = nn.Embedding(3, action_width).to(
            device=predictor.pos_embedding.device, dtype=predictor.pos_embedding.dtype)

    def forward(self, real_patches, executed_actions, candidate):
        require_float(real_patches, 'real DINO history')
        if real_patches.ndim != 4:
            raise ValueError('DINO history must be [B,T,P,Dvisual]')
        b, t, p, d = real_patches.shape
        if not 1 <= t <= self.history or (p, d) != (self.contract.raw_patches, self.contract.feature_width):
            raise ValueError('Real DINO history does not match predictor contract')
        if executed_actions.shape != (b, t-1) or candidate.shape != (b,):
            raise ValueError('T real views require T-1 executed actions and one candidate')
        parameter = self.predictor.pos_embedding
        past = action_ids(executed_actions, maximum=2, device=parameter.device)
        candidate = action_ids(candidate, maximum=2, device=parameter.device)
        commands = torch.cat((past, candidate[:, None]), dim=1)
        real = real_patches.detach().to(parameter.device, dtype=parameter.dtype)
        require_finite(real, 'real DINO history')
        command = self.command_embedding(commands).unsqueeze(2).expand(b,t,p,self.action_width)
        proprio = real.new_zeros(b,t,p,self.proprio_width)
        packed = torch.cat((real, proprio, command), dim=-1).reshape(b,t*p,-1)
        prediction = self.predictor(packed).reshape(b,t,p,-1)
        return prediction[:, -1, :, :d]  # visual channels only, raw native patches

    def load_transferred_predictor(self, state_dict):
        # Strict loading rejects mismatched grid, history, width or layer count.
        # Source action/proprio encoders are intentionally NOT loaded here.
        return self.predictor.load_state_dict(state_dict, strict=True)


class DinoRGBEncoder(nn.Module):
    """Shared real/decoded-RGB encoding using the inspected DINO-WM convention.

    Input is float RGB [B,3,H,W] in [0,1]. The base DINO module must expose
    forward_features(), patch_size, and num_features; no torch.hub download occurs.
    """
    def __init__(self, encoder, contract=VisualContract(), source_image_size=224):
        super().__init__()
        self.encoder, self.contract = encoder, contract
        patch_size = encoder.patch_size
        if isinstance(patch_size, (tuple, list)):
            if len(set(patch_size)) != 1:
                raise ValueError('Only square DINO patches are supported')
            patch_size = patch_size[0]
        if patch_size != contract.patch_size or encoder.num_features != contract.feature_width:
            raise ValueError('DINO encoder does not match feature width/patch size')
        self.preprocess = transforms.Compose([
            transforms.Resize(source_image_size, antialias=True),
            transforms.CenterCrop(source_image_size),
            transforms.Normalize([.5,.5,.5], [.5,.5,.5]),
            transforms.Resize(contract.encoder_resolution, antialias=True),
        ])
        self.encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, rgb, *, input_grad=False, pooled=True):
        require_float(rgb, 'RGB')
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('RGB must be [B,3,H,W]')
        require_finite(rgb, 'RGB')
        if ((rgb < 0) | (rgb > 1)).any():
            raise ValueError('Pass RGB in [0,1], not VAE or ImageNet-normalized pixels')
        parameter = next(self.encoder.parameters())
        with torch.set_grad_enabled(input_grad):
            value = rgb if input_grad else rgb.detach()
            value = self.preprocess(value.to(parameter.device, dtype=parameter.dtype))
            features = self.encoder.forward_features(value)['x_norm_patchtokens']
            if features.shape[-2:] != (self.contract.raw_patches, self.contract.feature_width):
                raise ValueError('Actual DINO output violates the declared contract')
            return pool_dino_patches(features, self.contract) if pooled else features


class NWMVisualBridge(nn.Module):
    """VAE representation boundary. Supply completed FOUR-channel diffusion
    samples, never the CDiT denoiser's epsilon/variance (usually 8-channel) output.
    """
    def __init__(self, vae, dino: DinoRGBEncoder, image_size=224, latent_scale=.18215):
        super().__init__()
        if image_size % 16 or latent_scale <= 0:
            raise ValueError('NWM image size must support VAE /8 and CDiT /2 patches')
        if getattr(vae.config, 'latent_channels', None) != 4:
            raise ValueError('NWM requires a four-channel VAE')
        self.vae, self.dino = vae, dino
        self.image_size, self.latent_size, self.latent_scale = image_size, image_size // 8, latent_scale
        self.vae.requires_grad_(False).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size, antialias=True), transforms.CenterCrop(image_size),
            transforms.Normalize([.5,.5,.5], [.5,.5,.5]),
        ])

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode_context(self, rgb, *, posterior='sample', generator=None):
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError('NWM real context must be [B,T,3,H,W]')
        require_float(rgb, 'RGB context'); require_finite(rgb, 'RGB context')
        if ((rgb < 0) | (rgb > 1)).any():
            raise ValueError('Real context must be RGB in [0,1]')
        b, t = rgb.shape[:2]
        parameter = next(self.vae.parameters())
        pixels = self.preprocess(rgb.flatten(0,1).to(parameter.device, dtype=parameter.dtype))
        posterior_dist = self.vae.encode(pixels).latent_dist
        if posterior == 'sample':
            latent = posterior_dist.sample(generator=generator)
        elif posterior == 'mode':
            latent = posterior_dist.mode()
        else:
            raise ValueError('Declare sample or mode for VAE posterior encoding')
        expected = (b*t,4,self.latent_size,self.latent_size)
        if latent.shape != expected:
            raise ValueError('VAE latent grid differs from NWM configuration')
        return (latent * self.latent_scale).reshape(b,t,4,self.latent_size,self.latent_size)

    def forward(self, sampled_latents, *, input_grad=False):
        if sampled_latents.ndim != 4 or sampled_latents.shape[1:] != (4,self.latent_size,self.latent_size):
            raise ValueError('Decode completed diffusion samples [B,4,H/8,W/8], not denoiser outputs')
        require_float(sampled_latents, 'sampled latents'); require_finite(sampled_latents, 'sampled latents')
        parameter = next(self.vae.parameters())
        with torch.set_grad_enabled(input_grad):
            latent = sampled_latents if input_grad else sampled_latents.detach()
            decoded = self.vae.decode(latent.to(parameter.device, dtype=parameter.dtype) / self.latent_scale).sample
            rgb = (decoded.clamp(-1,1) + 1) / 2
            return self.dino(rgb, input_grad=input_grad)


def cdit_conditioning(command_ids, command_table, *, context_latents, model, steps=1, time_denominator=128.):
    """Caller declares CE command x/y/yaw normalization. Never use realized poses.
    One temporal unit is one CE command, not an assumed duration in seconds.
    """
    ids = action_ids(command_ids, maximum=2, device=context_latents.device)
    if ids.ndim != 1 or command_table.shape != (3,3):
        raise ValueError('Declare [3,3] normalized F/L/R command table and [B] IDs')
    require_float(command_table, 'command table'); require_finite(command_table, 'command table')
    if context_latents.ndim != 5 or context_latents.shape[0] != len(ids) or context_latents.shape[1] != model.context_size or context_latents.shape[2] != 4:
        raise ValueError('NWM context must match checkpoint frame count and four latent channels')
    grid = tuple(model.x_embedder.img_size)
    if tuple(context_latents.shape[-2:]) != grid:
        raise ValueError('Context latent grid does not match CDiT positions')
    if not math.isfinite(steps) or not math.isfinite(time_denominator) or steps <= 0 or time_denominator <= 0:
        raise ValueError('Positive time index and denominator required')
    parameter = next(model.parameters())
    context = context_latents.to(parameter.device, dtype=parameter.dtype)
    table = command_table.to(parameter.device, dtype=parameter.dtype)
    return dict(y=table[ids.to(parameter.device)], x_cond=context,
                rel_t=context.new_full((len(ids),), steps/time_denominator))


class CheckedCDiTDenoiser(nn.Module):
    """Preserve NWM epsilon/variance output; adapt dtype and validate all axes.
    Pass this module to a compatible diffusion sampler, not directly to a VAE.
    """
    def __init__(self, model):
        super().__init__()
        if model.in_channels != 4 or model.out_channels not in (4,8):
            raise ValueError('Expected NWM CDiT latent/noise dimensions')
        self.model = model

    def forward(self, x, t, *, y, x_cond, rel_t):
        parameter = next(self.model.parameters())
        b = x.shape[0]
        grid = tuple(self.model.x_embedder.img_size)
        if x.shape != (b,4,*grid) or t.shape != (b,) or rel_t.shape != (b,) or y.shape != (b,3):
            raise ValueError('CDiT expects x[B,4,H,W], t[B], y[B,3], rel_t[B]')
        if x_cond.shape != (b,self.model.context_size,4,*grid):
            raise ValueError('Wrong CDiT context frame count/grid')
        values = [v.to(parameter.device, dtype=parameter.dtype) for v in (x,t,y,x_cond,rel_t)]
        for value in values:
            require_finite(value, 'CDiT input')
        # Its sinusoidal time embedding explicitly produces FP32; autocast is
        # required when the learned weights use FP16/BF16.
        use_amp = parameter.dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(parameter.device.type, dtype=parameter.dtype if use_amp else torch.bfloat16, enabled=use_amp):
            output = self.model(values[0], values[1], values[2], values[3], values[4])
        if output.shape != (b,self.model.out_channels,*grid):
            raise ValueError('Unexpected denoiser output shape')
        return output


@torch.no_grad()
def sample_nwm_latents(denoiser, diffusion, conditioning, *, seed):
    """Complete upstream ancestral sampling before decoding. Frozen inference only.
    Separate processes are required for concurrent callers (upstream uses global RNG).
    """
    if denoiser.training:
        raise ValueError('Set the denoiser to eval mode for frozen prediction')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Record an integer sampling seed in [0,2**63)')
    if diffusion.model_mean_type.name != 'EPSILON' or diffusion.model_var_type.name != 'LEARNED_RANGE' or denoiser.model.out_channels != 8:
        raise ValueError('Sampler must match the inspected epsilon plus learned-variance checkpoint')
    context = conditioning['x_cond']
    parameter = next(denoiser.parameters())
    device = parameter.device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Seeded sampler currently supports CPU/CUDA only')
    shape = (context.shape[0], 4, *denoiser.model.x_embedder.img_size)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        for index in cuda_devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        noise = torch.randn(shape, device=device, dtype=torch.float32)
        samples = diffusion.p_sample_loop(denoiser, shape, noise, clip_denoised=False,
                                         model_kwargs=conditioning, progress=False, device=device)
    require_finite(samples, 'completed NWM samples')
    return samples
