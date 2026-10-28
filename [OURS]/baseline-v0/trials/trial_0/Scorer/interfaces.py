"""Explicit model interfaces. No checkpoint downloads or simulator dependencies."""
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F
from contextlib import contextmanager

@dataclass(frozen=True)
class VisualContract:
    raw_grid: int = 14
    pool_factor: int = 2
    feature_width: int = 384
    patch_size: int = 14

    def __post_init__(self):
        values = (self.raw_grid, self.pool_factor, self.feature_width, self.patch_size)
        if any(type(x) is not int or x < 1 for x in values) or self.raw_grid % self.pool_factor:
            raise ValueError("Positive dimensions and a divisible pooling grid are required")

    @property
    def pooled_grid(self): return self.raw_grid // self.pool_factor
    @property
    def raw_patches(self): return self.raw_grid ** 2
    @property
    def pooled_patches(self): return self.pooled_grid ** 2
    @property
    def encoder_resolution(self): return self.raw_grid * self.patch_size

    @classmethod
    def from_config(cls, cfg):
        result = cls(cfg['raw_grid'], cfg['pool_factor'], cfg['feature_width'], cfg['patch_size'])
        for name, value in [('raw_patches', result.raw_patches), ('pooled_patches', result.pooled_patches),
                            ('input_resolution', result.encoder_resolution)]:
            if name in cfg and cfg[name] != value:
                raise ValueError(f'{name} contradicts the configured grid')
        return result

def require_finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'Nonfinite {name}')

def require_float(value, name):
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f'{name} must be floating point')

def action_ids(value, maximum=3, device=None):
    if value.dtype not in (torch.int32, torch.int64):
        raise TypeError('Action/index tensors must use integers, not floats or bools')
    value = value.to(device=device, dtype=torch.long)
    if ((value < 0) | (value > maximum)).any():
        raise ValueError('Action/index outside its declared range')
    return value

def pool_dino_patches(features, contract=VisualContract()):
    require_float(features, 'DINO patches')
    if features.ndim < 2 or features.shape[-2:] != (contract.raw_patches, contract.feature_width):
        raise ValueError(f'Expected [...,{contract.raw_patches},{contract.feature_width}] raw patches')
    prefix = features.shape[:-2]
    grid = features.reshape(-1, contract.raw_grid, contract.raw_grid, contract.feature_width).permute(0,3,1,2)
    return F.avg_pool2d(grid, contract.pool_factor).permute(0,2,3,1).reshape(*prefix, contract.pooled_patches, contract.feature_width)

def extract_dino_prediction(prediction, contract, *, layout, condition_width=0):
    """Take last predicted frame and remove declared action/proprio channels/tokens."""
    require_float(prediction, 'DINO prediction')
    if prediction.ndim != 4 or prediction.shape[1] < 1:
        raise ValueError('Expected [B,T,P,D] DINO-WM output')
    if layout == 'visual_only':
        expected = (contract.raw_patches, contract.feature_width)
        visual = prediction[:, -1]
    elif layout == 'feature_concat':
        if type(condition_width) is not int or condition_width < 1:
            raise ValueError('Declare the total action/proprio feature width')
        expected = (contract.raw_patches, contract.feature_width + condition_width)
        visual = prediction[:, -1, :, :contract.feature_width]
    elif layout == 'token_concat':
        expected = (contract.raw_patches + 2, contract.feature_width)
        visual = prediction[:, -1, :-2]
    else:
        raise ValueError('Unknown DINO layout; do not infer it from shape')
    if prediction.shape[-2:] != expected:
        raise ValueError(f'DINO output does not match {layout}: expected {expected}')
    return pool_dino_patches(visual, contract)

def last_valid_state(hidden, attention_valid):
    """Last prefill state, with correct left/right padding and expanded-token mask."""
    if hidden.ndim != 3 or attention_valid.shape != hidden.shape[:2] or attention_valid.dtype != torch.bool:
        raise ValueError('Mask must match the actual expanded prefill sequence')
    mask = attention_valid.to(hidden.device)
    if not mask.any(dim=1).all():
        raise ValueError('Every prompt needs a valid token')
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(mask)
    index = positions.masked_fill(~mask, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]

def instruction_states(hidden, instruction_mask):
    """Pack complete instruction tokens from an instruction-only prefill."""
    if hidden.ndim != 3 or instruction_mask.shape != hidden.shape[:2] or instruction_mask.dtype != torch.bool:
        raise ValueError('Instruction mask must match prefill hidden states')
    mask = instruction_mask.to(hidden.device)
    lengths = mask.sum(dim=1)
    if (lengths == 0).any():
        raise ValueError('Each instruction needs at least one token')
    length = int(lengths.max())
    output = hidden.new_zeros(hidden.shape[0], length, hidden.shape[2])
    valid = torch.arange(length, device=hidden.device)[None] < lengths[:, None]
    for row in range(hidden.shape[0]):
        output[row, :lengths[row]] = hidden[row, mask[row]]
    return output.detach(), valid

def validate_policy_config(backbone, policy_config, expected_width):
    """Validate the TEXT hidden width, not the vision-tower dimension."""
    expected_types = {'smolvlm2':'smolvlm', 'onevision':'llava', 'qwen3vl':'qwen3_vl'}
    if policy_config.get('model_type') != expected_types[backbone]:
        raise ValueError('Wrong VLM family; OneVision requires its native LLaVA integration')
    text = policy_config.get('text_config', policy_config)
    if text.get('hidden_size') != expected_width:
        raise ValueError('Navigation VLM text width differs from scorer projection')

class PrefillFeatureTap:
    """Attach to the actual final decoder norm ONLY during one prefill forward.
    Caller selects the module for its installed model implementation. Do not wrap
    generate(): multiple calls are rejected instead of using candidate/decode states.
    """
    def __init__(self, final_norm, expected_width):
        self.module, self.expected_width = final_norm, expected_width
        self.hidden = None

    @contextmanager
    def capture(self):
        if self.module.training:
            raise ValueError('Set the frozen policy to eval mode before extracting features')
        self.hidden = None
        calls = []
        def hook(_module, _args, output):
            if not isinstance(output, Tensor) or output.ndim != 3 or output.shape[-1] != self.expected_width:
                raise ValueError('Hook must target final decoder hidden states, not vocabulary logits')
            calls.append(1)
            self.hidden = output.detach()
        handle = self.module.register_forward_hook(hook)
        try:
            yield self
            if len(calls) != 1:
                self.hidden = None
                raise ValueError('Expected one prefill call; generation/cached decode states are not allowed')
        except BaseException:
            self.hidden = None
            raise
        finally:
            handle.remove()

@dataclass(frozen=True)
class ActionMap:
    """Read these IDs from the live task; internal order is F,L,R,STOP."""
    simulator_ids: tuple[int, int, int, int]

    def __post_init__(self):
        if len(self.simulator_ids) != 4 or len(set(self.simulator_ids)) != 4 or any(type(v) is not int or v < 0 for v in self.simulator_ids):
            raise ValueError('Four distinct nonnegative simulator IDs required')

    def to_simulator(self, canonical):
        canonical = action_ids(canonical)
        return torch.tensor(self.simulator_ids, device=canonical.device)[canonical]

    def from_simulator(self, simulator):
        if simulator.dtype not in (torch.int32, torch.int64):
            raise TypeError('Simulator action IDs must be integers')
        matches = simulator[..., None] == torch.tensor(self.simulator_ids, device=simulator.device)
        if not matches.any(dim=-1).all():
            raise ValueError('Unsupported simulator action')
        return matches.long().argmax(dim=-1)
