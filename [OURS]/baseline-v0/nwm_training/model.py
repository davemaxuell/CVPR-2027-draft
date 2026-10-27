"""Pinned native CDiT, frozen VAE, and nominal CE command conditioning."""
import math
import sys

import torch
from torchvision import transforms

from Scorer.world_models import NWM_MODELS_SHA256, load_pinned_source
from nwm_training.common import ROOT, RUN, VAE_DIRECTORY, verify_assets

LATENT_SCALE = .18215
TIME_DENOMINATOR = 128.
METRIC_WAYPOINT_SPACING = .25
NOMINAL_COMMANDS = ((.25, 0., 0.), (0., 0., math.pi / 12), (0., 0., -math.pi / 12))


def command_table(device='cpu'):
    """Robot-local forward x, left y, positive left yaw; no realized motion.

    Apply the pinned source's waypoint scaling and asymmetric x/y min/max
    normalization. A zero displacement therefore has normalized x = -1/3.
    A time unit is one CE command, not a claimed number of physical seconds.
    """
    commands = torch.tensor(NOMINAL_COMMANDS, dtype=torch.float32, device=device)
    limits_min = commands.new_tensor([-2.5, -4.])
    limits_max = commands.new_tensor([5., 4.])
    commands[:, :2] = ((commands[:, :2] / METRIC_WAYPOINT_SPACING - limits_min)
                       / (limits_max - limits_min)) * 2 - 1
    return commands


def conditioning(context, actions):
    assert context.ndim == 5 and tuple(context.shape[1:]) == (4, 4, 28, 28)
    assert actions.shape == (len(context),) and actions.dtype == torch.long
    assert torch.all((actions >= 0) & (actions < 3))
    assert torch.isfinite(context).all()
    return {'x_cond': context, 'y': command_table(context.device)[actions],
            'rel_t': context.new_full((len(context),), 1 / TIME_DENOMINATOR)}


def preprocessing():
    # The agreed CE square crop, deliberately different from native NWM's
    # 4:3-to-square rescaling. Matches the existing NWMVisualBridge interface.
    return transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias=True),
                               transforms.CenterCrop(224), transforms.Normalize([.5] * 3, [.5] * 3)])


def native_model():
    source = load_pinned_source(ROOT / 'third_party/nwm/models.py', NWM_MODELS_SHA256)
    return source.CDiT_models['CDiT-S/2'](context_size=4, input_size=28, in_channels=4, learn_sigma=True)


def build_model(initialization, device='cpu', *, seed=5):
    assert initialization in ('transfer', 'fresh')
    torch.manual_seed(seed)
    model = native_model()
    if initialization == 'transfer':
        record = verify_assets()
        state = torch.load(record['extracted_ema'], map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    assert model.out_channels == 8 and model.context_size == 4
    assert sum(p.numel() for p in model.parameters()) == 49227680
    return model.to(device)


def build_vae(device='cpu'):
    dependencies = str(ROOT / 'env/nwm_dependencies')
    if dependencies not in sys.path:
        sys.path.insert(0, dependencies)
    from diffusers import AutoencoderKL
    model = AutoencoderKL.from_pretrained(str(VAE_DIRECTORY), local_files_only=True,
                                         use_safetensors=True, torch_dtype=torch.float32)
    assert model.config.latent_channels == 4 and len(model.config.block_out_channels) == 4
    return model.to(device).eval().requires_grad_(False)
