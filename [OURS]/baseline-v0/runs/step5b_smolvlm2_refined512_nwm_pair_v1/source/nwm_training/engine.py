"""Native epsilon/learned-variance fitting with a matched EMA and reproducible RNG."""
import os
import random

import numpy as np
import torch

from Scorer.world_models import build_nwm_diffusion
from nwm_training.common import ROOT
from nwm_training.data import sample_posterior
from nwm_training.model import conditioning


def configure(seed=5):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def diffusion():
    return build_nwm_diffusion(ROOT / 'third_party/nwm/diffusion', sampling_steps=1000)


@torch.no_grad()
def update_ema(ema, model, decay=.9999):
    live = dict(model.named_parameters())
    for name, parameter in ema.named_parameters():
        parameter.mul_(decay).add_(live[name].detach(), alpha=1 - decay)
    for name, buffer in ema.named_buffers():
        buffer.copy_(dict(model.named_buffers())[name])


def losses(model, process, batch, device):
    moments = batch['moments'].to(device, non_blocking=True)
    actions = batch['action'].to(device, dtype=torch.long, non_blocking=True)
    latents = sample_posterior(moments)
    target = latents[:, 4]
    kwargs = conditioning(latents[:, :4], actions)
    t = torch.randint(0, process.num_timesteps, (len(target),), device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        result = process.training_losses(model, target, t, model_kwargs=kwargs)
    assert all(torch.isfinite(v).all() for v in result.values())
    return result


def update(model, ema, optimizer, process, batch):
    assert model.training and not ema.training
    optimizer.zero_grad(set_to_none=True)
    result = losses(model, process, batch, next(model.parameters()).device)
    loss = result['loss'].mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step(); update_ema(ema, model)
    return {k: v.mean().item() for k, v in result.items()}, norm.item()


@torch.no_grad()
def evaluate(model, process, loader):
    assert not model.training
    device = next(model.parameters()).device
    rows = []
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.manual_seed(5005)
        for batch in loader:
            result = losses(model, process, batch, device)
            for i in range(len(batch['action'])):
                rows.append({'slot': int(batch['slot'][i]), 'dataset': int(batch['dataset'][i]),
                             'action': int(batch['action'][i]), **{k: v[i].item() for k, v in result.items()}})
    assert len(rows) == len(loader.dataset)
    groups = {}
    for dataset, label in enumerate(('r2r', 'rxr')):
        for action, name in enumerate(('FORWARD', 'LEFT', 'RIGHT')):
            selected = [r for r in rows if r['dataset'] == dataset and r['action'] == action]
            assert selected
            groups[f'{label}/{name}'] = {'samples': len(selected),
                **{k: float(np.mean([r[k] for r in selected])) for k in ('loss', 'mse', 'vb')}}
    return {'samples': len(rows), 'group': 'C', 'by_dataset_action': groups,
            **{'macro_' + k: float(np.mean([v[k] for v in groups.values()])) for k in ('loss', 'mse', 'vb')},
            'fixed_randomness_seed': 5005, 'posterior': 'sample_from_frozen_float16_moments',
            'objective': 'native 1000-step linear epsilon MSE plus learned-range variance variational term',
            'scope': 'Expert-path validation; prediction objective, not navigation success'}, rows


class AutocastDenoiser(torch.nn.Module):
    """Keep source FP32 weights/residuals and native BF16 autocast on CUDA."""
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, x, t, *, y, x_cond, rel_t):
        assert x.shape == (len(x), 4, 28, 28) and t.shape == rel_t.shape == (len(x),)
        assert y.shape == (len(x), 3) and x_cond.shape == (len(x), 4, 4, 28, 28)
        assert next(self.model.parameters()).dtype == torch.float32
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            result = self.model(x, t, y, x_cond, rel_t)
        assert result.shape == (len(x), 8, 28, 28)
        return result
