"""CPU-testable scorer reference. No policy, world model, or Habitat integration.

Inputs are cached frozen features. Only observe() updates actual-history memory;
score() is read-only. Canonical action IDs require an explicit Habitat adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if __package__:
    from .interfaces import VisualContract, pool_dino_patches, action_ids, require_float, require_finite
else:
    from interfaces import VisualContract, pool_dino_patches, action_ids, require_float, require_finite

FORWARD, LEFT, RIGHT, STOP, START = range(5)


def spatial_positions(width: int, grid: int) -> Tensor:
    if width % 4:
        raise ValueError("Width must be divisible by four")
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    freq = 10000.0 ** (-torch.arange(width // 4).float() / (width // 4))
    xf, yf = x.flatten()[:, None] * freq, y.flatten()[:, None] * freq
    return torch.cat((xf.sin(), xf.cos(), yf.sin(), yf.cos()), dim=-1)


class CandidateBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.norm_cross = nn.LayerNorm(width)
        self.norm_ff = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))

    def forward(self, tokens: Tensor, context: Tensor, padding: Tensor) -> Tensor:
        query = self.norm_self(tokens)
        tokens = tokens + self.self_attention(query, query, query, need_weights=False)[0]
        tokens = tokens + self.cross_attention(
            self.norm_cross(tokens), context, context,
            key_padding_mask=padding, need_weights=False,
        )[0]
        return tokens + self.ff(self.norm_ff(tokens))


@dataclass
class RealContext:
    memory: Tensor
    tokens: Tensor
    padding: Tensor
    current_patches: Tensor

    def detached(self):
        return RealContext(self.memory.detach(), self.tokens.detach(), self.padding, self.current_patches.detach())

    def select(self, index):
        return RealContext(*(x[index] for x in (self.memory, self.tokens, self.padding, self.current_patches)))


class NavigationScorer(nn.Module):
    def __init__(self, policy_width: int, width: int = 256, slots: int = 8, *,
                 visual=None, heads=4, blocks=2, ff_width=512, readout_width=128):
        super().__init__()
        sizes = (policy_width, width, slots, heads, blocks, ff_width, readout_width)
        if any(type(v) is not int or v < 1 for v in sizes) or width % heads or width % 4:
            raise ValueError('Invalid architecture dimensions/head divisibility')
        self.width, self.slots, self.policy_width = width, slots, policy_width
        self.visual = visual or VisualContract()
        self.patch_count = self.visual.pooled_patches
        self.settings = None
        self.policy_projection = nn.Sequential(nn.LayerNorm(policy_width), nn.Linear(policy_width, width))
        self.visual_projection = nn.Sequential(nn.LayerNorm(self.visual.feature_width), nn.Linear(self.visual.feature_width, width))
        self.action_embedding = nn.Embedding(5, width)
        self.type_embedding = nn.Embedding(6, width)  # text, real, state, memory, future, candidate
        self.time_projection = nn.Linear(1, width)
        self.slot_queries = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.candidate_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.register_buffer("positions", spatial_positions(width, self.visual.pooled_grid))
        self.memory_norm = nn.LayerNorm(width)
        self.memory_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.memory_ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, ff_width), nn.GELU(), nn.Linear(ff_width, width))
        self.memory_gru = nn.GRUCell(width, width)
        self.blocks = nn.ModuleList([CandidateBlock(width, heads, ff_width) for _ in range(blocks)])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, readout_width), nn.GELU())
        self.utility_head = nn.Linear(readout_width, 1)
        self.success_head = nn.Linear(readout_width, 1)
        self.collision_head = nn.Linear(readout_width, 1)

    @classmethod
    def from_config(cls, config, backbone):
        cfg = json.loads(Path(config).read_text(encoding='utf-8-sig')) if isinstance(config, (str, Path)) else copy.deepcopy(config)
        arch = cfg['architecture']
        if cfg['prediction_horizon_actions'] != 1 or cfg['candidate_actions'] != ['FORWARD','LEFT','RIGHT','STOP']:
            raise ValueError('This runtime implements H=1 and F,L,R,STOP')
        if cfg['canonical_action_ids'] != dict(zip(['FORWARD','LEFT','RIGHT','STOP','START'], range(5))):
            raise ValueError('Canonical IDs cannot be changed')
        if arch['heads_out'] != ['utility','success_logit','collision_logit'] or cfg['loss'].get('ranking_loss', 0) != 0:
            raise ValueError('Unsupported heads/loss; disable auxiliary supervision with zero loss weights')
        model = cls(cfg['text_and_context']['hidden_widths'][backbone], arch['width'], arch['memory_slots'],
                    visual=VisualContract.from_config(cfg['visual_features']), heads=arch['heads'],
                    blocks=arch['candidate_blocks'], ff_width=arch['feedforward_width'], readout_width=arch['readout_width'])
        validate_margin(cfg['selection']['override_margin'])
        validate_loss_weights(cfg['loss'])
        model.settings, model.backbone = cfg, backbone
        return model

    def _feature(self, value, name, *, frozen=True):
        require_float(value, name)
        value = value.detach() if frozen else value
        return value.to(device=self.slot_queries.device, dtype=self.slot_queries.dtype)

    def loss(self, outputs, utility, success, immediate_collision, actions):
        return scorer_loss(outputs, utility, success, immediate_collision, actions,
                           weights=self.settings['loss'] if self.settings else None)

    def observe(
        self, instruction: Tensor, instruction_valid: Tensor, current_patches: Tensor,
        policy_context: Tensor, previous_action: Tensor, elapsed_fraction: Tensor,
        memory: Tensor | None = None, reset_mask: Tensor | None = None,
    ) -> RealContext:
        """Call exactly once per real observation. Reset with memory=None per episode.

        instruction [B,L,policy_width], instruction_valid [B,L] bool,
        current_patches [B,P,Dvisual] from the configured grid, policy_context [B,policy_width],
        previous_action [B] (START on first observation), elapsed_fraction [B,1].
        Never call this on an imagined observation.
        """
        if instruction.ndim != 3:
            raise ValueError('Instruction features must be [B,L,Dpolicy]')
        batch, length, policy_width = instruction.shape
        if policy_width != self.policy_width or length < 1:
            raise ValueError("Invalid instruction feature dimensions")
        if instruction_valid.shape != (batch, length) or instruction_valid.dtype != torch.bool:
            raise ValueError("instruction_valid must be a [B,L] boolean mask")
        if not instruction_valid.any(dim=1).all():
            raise ValueError("Use a valid null token for an instruction ablation")
        if current_patches.shape != (batch, self.patch_count, self.visual.feature_width):
            raise ValueError('Current patches do not match the configured grid')
        if policy_context.shape != (batch, self.policy_width):
            raise ValueError("Invalid policy_context shape")
        if previous_action.shape != (batch,) or elapsed_fraction.shape != (batch, 1):
            raise ValueError("Invalid action/time shapes")
        if ((previous_action < 0) | (previous_action > START)).any():
            raise ValueError("Invalid canonical previous action")
        if ((elapsed_fraction < 0) | (elapsed_fraction > 1)).any():
            raise ValueError("elapsed_fraction must be between zero and one")

        instruction_valid = instruction_valid.to(self.slot_queries.device)
        previous_action = action_ids(previous_action, maximum=START, device=self.slot_queries.device)
        instruction = self._feature(instruction, 'instruction').masked_fill(~instruction_valid[..., None], 0)
        current_patches = self._feature(current_patches, 'current patches')
        policy_context = self._feature(policy_context, 'policy context')
        elapsed_fraction = self._feature(elapsed_fraction, 'elapsed fraction')
        for name, value in [('instruction',instruction),('current patches',current_patches),
                            ('policy context',policy_context),('elapsed fraction',elapsed_fraction)]:
            require_finite(value, name)

        # Frozen inputs cannot receive gradients; projections still train.
        text = self.policy_projection(instruction.detach()) + self.type_embedding.weight[0]
        current = self.visual_projection(current_patches.detach()) + self.positions + self.type_embedding.weight[1]
        state = self.policy_projection(policy_context.detach()) + self.type_embedding.weight[2]
        action_time = self.action_embedding(previous_action) + self.time_projection(elapsed_fraction.detach())
        source = torch.cat((text, current, state[:, None], action_time[:, None]), dim=1)
        padding = torch.cat((~instruction_valid, torch.zeros(batch, self.patch_count + 2, dtype=torch.bool, device=text.device)), dim=1)

        if memory is None:
            memory = text.new_zeros(batch, self.slots, self.width)
        else:
            memory = self._feature(memory, 'memory', frozen=False)
        if memory.shape != (batch, self.slots, self.width):
            raise ValueError("Invalid real-memory dimensions")
        if reset_mask is not None:
            if reset_mask.shape != (batch,) or reset_mask.dtype != torch.bool:
                raise ValueError('reset_mask must be [B] bool')
            memory = memory.masked_fill(reset_mask.to(memory.device)[:,None,None], 0)
        require_finite(memory, 'real memory')
        query = self.memory_norm(memory + self.slot_queries)
        update = self.memory_attention(query, source, source, key_padding_mask=padding, need_weights=False)[0]
        update = update + self.memory_ff(update)
        # Do not detach here: caller controls truncated BPTT between real steps.
        new_memory = self.memory_gru(update.reshape(-1, self.width), memory.reshape(-1, self.width))
        new_memory = new_memory.reshape(batch, self.slots, self.width)
        context = torch.cat((source, new_memory + self.type_embedding.weight[3]), dim=1)
        context_padding = torch.cat((padding, torch.zeros(batch, self.slots, dtype=torch.bool, device=text.device)), dim=1)
        return RealContext(new_memory, context, context_padding, current_patches.detach())

    def score(self, context: RealContext, future_patches: Tensor, actions: Tensor, *, future_grad=False) -> dict[str, Tensor]:
        """future_patches [B,K,P,Dvisual], actions [B,K]. No history mutation.

        STOP uses the current view, overriding its supplied placeholder. The
        world-model caller should never generate a STOP prediction.
        """
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError("Expected at least one candidate per root")
        batch, candidates = actions.shape
        if context.current_patches.shape != (batch, self.patch_count, self.visual.feature_width) or future_patches.shape != (batch, candidates, self.patch_count, self.visual.feature_width):
            raise ValueError("Invalid candidate dimensions")
        if context.tokens.device != self.slot_queries.device or context.tokens.dtype != self.slot_queries.dtype:
            raise ValueError('Rebuild real context after moving/changing scorer dtype')
        if future_grad:
            if self.training or any(p.requires_grad for p in self.parameters()):
                raise ValueError('future_grad requires scorer.eval().requires_grad_(False)')
            context = context.detached()
        actions = action_ids(actions, device=self.slot_queries.device)
        future_patches = self._feature(future_patches, 'future patches', frozen=not future_grad)
        if ((actions < FORWARD) | (actions > STOP)).any():
            raise ValueError("Candidate action must be FORWARD, LEFT, RIGHT, or STOP")
        future = torch.where(
            (actions == STOP)[..., None, None],
            context.current_patches[:, None], future_patches,
        )
        require_finite(future, 'movement prediction')
        visual = self.visual_projection(future) + self.positions + self.type_embedding.weight[4]
        candidate = self.action_embedding(actions) + self.candidate_token + self.type_embedding.weight[5]
        tokens = torch.cat((candidate[:, :, None], visual), dim=2).reshape(batch * candidates, self.patch_count + 1, self.width)
        source = context.tokens.repeat_interleave(candidates, dim=0)
        padding = context.padding.repeat_interleave(candidates, dim=0)
        for block in self.blocks:
            tokens = block(tokens, source, padding)
        summary = self.readout(tokens[:, 0])
        return {
            "utility": self.utility_head(summary).reshape(batch, candidates),
            "success_logit": self.success_head(summary).reshape(batch, candidates),
            "collision_logit": self.collision_head(summary).reshape(batch, candidates),
        }


def validate_loss_weights(weights):
    for name in ('utility_mse','terminal_success_bce','immediate_forward_collision_bce'):
        value = weights[name]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError('Loss weights must be finite and nonnegative')
    if weights['utility_mse'] <= 0:
        raise ValueError('Utility supervision must be positive')


def scorer_loss(outputs: dict[str, Tensor], utility: Tensor, success: Tensor,
                immediate_collision: Tensor, actions: Tensor, *, weights=None) -> dict[str, Tensor]:
    weights = weights or {'utility_mse':1., 'terminal_success_bce':.1, 'immediate_forward_collision_bce':.1}
    validate_loss_weights(weights)
    prediction = outputs['utility']
    labels = []
    for name, label in [('utility',utility),('success',success),('collision',immediate_collision)]:
        if label.shape != prediction.shape:
            raise ValueError(f'{name} shape differs from scores')
        label = label.detach().to(prediction.device, dtype=prediction.dtype)
        require_finite(label, name)
        if name != 'utility' and ((label < 0) | (label > 1)).any():
            raise ValueError('Binary targets must be in [0,1]')
        labels.append(label)
    utility, success, immediate_collision = labels
    if actions.shape != prediction.shape:
        raise ValueError('Action shape differs from scores')
    actions = action_ids(actions, device=prediction.device)
    value = F.mse_loss(outputs["utility"], utility.detach())
    completion = F.binary_cross_entropy_with_logits(outputs["success_logit"], success.detach())
    collision_raw = F.binary_cross_entropy_with_logits(
        outputs["collision_logit"], immediate_collision.detach(), reduction="none",
    )
    forward_mask = (actions == FORWARD).to(collision_raw.dtype)
    collision = (collision_raw * forward_mask).sum() / forward_mask.sum().clamp_min(1)
    return {"total": weights['utility_mse'] * value + weights['terminal_success_bce'] * completion + weights['immediate_forward_collision_bce'] * collision,
            "value": value, "success": completion, "collision": collision}


def validate_margin(margin):
    if margin is not None and (isinstance(margin, bool) or not math.isfinite(margin) or margin < 0):
        raise ValueError('Margin must be null (disabled) or finite and nonnegative')


@dataclass
class Selection:
    actions: Tensor
    overridden: Tensor
    reasons: list[str]


def choose_action(utility, actions, base_index, margin) -> Selection:
    validate_margin(margin)
    actions = action_ids(actions)
    if actions.ndim != 2 or actions.shape[1] < 1 or base_index.shape != (actions.shape[0],):
        raise ValueError('Incompatible action-selection shapes')
    base_index = action_ids(base_index, maximum=actions.shape[1]-1, device=actions.device)
    ordered = actions.sort(dim=1).values
    if (ordered[:,1:] == ordered[:,:-1]).any():
        raise ValueError('Candidate actions must be unique')
    base = actions.gather(1, base_index[:,None]).squeeze(1)
    no_override = torch.zeros_like(base, dtype=torch.bool)
    if margin is None:
        return Selection(base, no_override, ['disabled'] * len(base))
    if utility is None:
        return Selection(base, no_override, ['prediction_unavailable'] * len(base))
    require_float(utility, 'utility')
    if utility.shape != actions.shape:
        raise ValueError('Utility shape differs from candidates')
    utility = utility.detach().to(actions.device)
    valid = torch.isfinite(utility).all(dim=1)
    safe = torch.where(valid[:,None], utility, 0.)
    best_value = safe.max(dim=1).values
    best_action = torch.where(safe == best_value[:,None], actions, STOP+1).min(dim=1).values
    base_value = safe.gather(1, base_index[:,None]).squeeze(1)
    override = valid & (best_value - base_value > margin)
    chosen = torch.where(override, best_action, base)
    reasons = ['nonfinite_scores' if not good else 'override' if changed else 'base_margin_or_tie'
               for good, changed in zip(valid.tolist(), override.tolist())]
    return Selection(chosen, override, reasons)


def select_action(utility: Tensor, actions: Tensor, base_index: Tensor, margin: float | None) -> Tensor:
    """Conservative override; margin must be selected on held-out training scenes."""
    return choose_action(utility, actions, base_index, margin).actions
