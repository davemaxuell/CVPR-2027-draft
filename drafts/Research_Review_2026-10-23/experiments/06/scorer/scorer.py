"""CPU-testable scorer reference. No policy, world model, or Habitat integration.

Inputs are cached frozen features. Only observe() updates actual-history memory;
score() is read-only. Canonical action IDs require an explicit Habitat adapter.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

FORWARD, LEFT, RIGHT, STOP, START = range(5)


def pool_dino_patches(features: Tensor) -> Tensor:
    """Row-major 16x16 patches -> 8x8 patches by 2x2 average pooling."""
    if features.shape[-2:] != (256, 384):
        raise ValueError("Expected [..., 256, 384] DINOv2-S/14 patch features")
    prefix = features.shape[:-2]
    grid = features.reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
    return F.avg_pool2d(grid, 2).permute(0, 2, 3, 1).reshape(*prefix, 64, 384)


def spatial_positions(width: int) -> Tensor:
    if width % 4:
        raise ValueError("Width must be divisible by four")
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
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


class NavigationScorer(nn.Module):
    def __init__(self, policy_width: int, width: int = 256, slots: int = 8):
        super().__init__()
        self.width, self.slots, self.policy_width = width, slots, policy_width
        self.policy_projection = nn.Sequential(nn.LayerNorm(policy_width), nn.Linear(policy_width, width))
        self.visual_projection = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, width))
        self.action_embedding = nn.Embedding(5, width)
        self.type_embedding = nn.Embedding(6, width)  # text, real, state, memory, future, candidate
        self.time_projection = nn.Linear(1, width)
        self.slot_queries = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.candidate_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.register_buffer("positions", spatial_positions(width))
        self.memory_norm = nn.LayerNorm(width)
        self.memory_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.memory_ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 512), nn.GELU(), nn.Linear(512, width))
        self.memory_gru = nn.GRUCell(width, width)
        self.blocks = nn.ModuleList([CandidateBlock(width, 4, 512) for _ in range(2)])
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.GELU())
        self.utility_head = nn.Linear(128, 1)
        self.success_head = nn.Linear(128, 1)
        self.collision_head = nn.Linear(128, 1)

    def observe(
        self, instruction: Tensor, instruction_valid: Tensor, current_patches: Tensor,
        policy_context: Tensor, previous_action: Tensor, elapsed_fraction: Tensor,
        memory: Tensor | None = None,
    ) -> RealContext:
        """Call exactly once per real observation. Reset with memory=None per episode.

        instruction [B,L,policy_width], instruction_valid [B,L] bool,
        current_patches [B,64,384], policy_context [B,policy_width],
        previous_action [B] (START on first observation), elapsed_fraction [B,1].
        Never call this on an imagined observation.
        """
        batch, length, policy_width = instruction.shape
        if policy_width != self.policy_width or length < 1:
            raise ValueError("Invalid instruction feature dimensions")
        if instruction_valid.shape != (batch, length) or instruction_valid.dtype != torch.bool:
            raise ValueError("instruction_valid must be a [B,L] boolean mask")
        if not instruction_valid.any(dim=1).all():
            raise ValueError("Use a valid null token for an instruction ablation")
        if current_patches.shape != (batch, 64, 384):
            raise ValueError("Expected current_patches [B,64,384]")
        if policy_context.shape != (batch, self.policy_width):
            raise ValueError("Invalid policy_context shape")
        if previous_action.shape != (batch,) or elapsed_fraction.shape != (batch, 1):
            raise ValueError("Invalid action/time shapes")
        if ((previous_action < 0) | (previous_action > START)).any():
            raise ValueError("Invalid canonical previous action")
        if ((elapsed_fraction < 0) | (elapsed_fraction > 1)).any():
            raise ValueError("elapsed_fraction must be between zero and one")

        # Frozen inputs cannot receive gradients; projections still train.
        text = self.policy_projection(instruction.detach()) + self.type_embedding.weight[0]
        current = self.visual_projection(current_patches.detach()) + self.positions + self.type_embedding.weight[1]
        state = self.policy_projection(policy_context.detach()) + self.type_embedding.weight[2]
        action_time = self.action_embedding(previous_action) + self.time_projection(elapsed_fraction.detach())
        source = torch.cat((text, current, state[:, None], action_time[:, None]), dim=1)
        padding = torch.cat((~instruction_valid, torch.zeros(batch, 66, dtype=torch.bool, device=text.device)), dim=1)

        if memory is None:
            memory = text.new_zeros(batch, self.slots, self.width)
        if memory.shape != (batch, self.slots, self.width):
            raise ValueError("Invalid real-memory dimensions")
        query = self.memory_norm(memory + self.slot_queries)
        update = self.memory_attention(query, source, source, key_padding_mask=padding, need_weights=False)[0]
        update = update + self.memory_ff(update)
        # Do not detach here: caller controls truncated BPTT between real steps.
        new_memory = self.memory_gru(update.reshape(-1, self.width), memory.reshape(-1, self.width))
        new_memory = new_memory.reshape(batch, self.slots, self.width)
        context = torch.cat((source, new_memory + self.type_embedding.weight[3]), dim=1)
        context_padding = torch.cat((padding, torch.zeros(batch, self.slots, dtype=torch.bool, device=text.device)), dim=1)
        return RealContext(new_memory, context, context_padding, current_patches.detach())

    def score(self, context: RealContext, future_patches: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """future_patches [B,K,64,384], actions [B,K]. No history mutation.

        STOP uses the current view, overriding its supplied placeholder. The
        world-model caller should never generate a STOP prediction.
        """
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError("Expected at least one candidate per root")
        batch, candidates = actions.shape
        if context.memory.shape[0] != batch or future_patches.shape != (batch, candidates, 64, 384):
            raise ValueError("Invalid candidate dimensions")
        if ((actions < FORWARD) | (actions > STOP)).any():
            raise ValueError("Candidate action must be FORWARD, LEFT, RIGHT, or STOP")
        future = torch.where(
            (actions == STOP)[..., None, None],
            context.current_patches[:, None], future_patches.detach(),
        )
        visual = self.visual_projection(future) + self.positions + self.type_embedding.weight[4]
        candidate = self.action_embedding(actions) + self.candidate_token + self.type_embedding.weight[5]
        tokens = torch.cat((candidate[:, :, None], visual), dim=2).reshape(batch * candidates, 65, self.width)
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


def scorer_loss(outputs: dict[str, Tensor], utility: Tensor, success: Tensor,
                immediate_collision: Tensor, actions: Tensor) -> dict[str, Tensor]:
    value = F.mse_loss(outputs["utility"], utility.detach())
    completion = F.binary_cross_entropy_with_logits(outputs["success_logit"], success.detach())
    collision_raw = F.binary_cross_entropy_with_logits(
        outputs["collision_logit"], immediate_collision.detach(), reduction="none",
    )
    forward_mask = (actions == FORWARD).to(collision_raw.dtype)
    collision = (collision_raw * forward_mask).sum() / forward_mask.sum().clamp_min(1)
    return {"total": value + 0.1 * completion + 0.1 * collision,
            "value": value, "success": completion, "collision": collision}


def select_action(utility: Tensor, actions: Tensor, base_index: Tensor, margin: float) -> Tensor:
    """Conservative override; margin must be selected on held-out training scenes."""
    if margin < 0 or not torch.isfinite(torch.tensor(margin)):
        raise ValueError("Margin must be finite and nonnegative")
    if utility.shape != actions.shape or base_index.shape != (actions.shape[0],):
        raise ValueError("Incompatible selection shapes")
    if not torch.isfinite(utility).all():
        raise ValueError("Nonfinite utility: deployment adapter must fall back to base policy")
    best_value = utility.max(dim=1).values
    tied_actions = torch.where(utility == best_value[:, None], actions, STOP + 1)
    best_action = tied_actions.min(dim=1).values
    base_value = utility.gather(1, base_index[:, None]).squeeze(1)
    base_action = actions.gather(1, base_index[:, None]).squeeze(1)
    return torch.where(best_value - base_value > margin, best_action, base_action)
