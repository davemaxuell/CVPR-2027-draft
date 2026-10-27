import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoRAExpert(nn.Module):
    """Single LoRA expert: down-projection A followed by up-projection B."""

    def __init__(self, in_features: int, out_features: int, r: int):
        super().__init__()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming for A, zero for B (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x))


class MoELoRALayer(nn.Module):
    """
    Replaces a single nn.Linear with MoE-LoRA.

    Keeps the original weight frozen, adds N LoRA experts + a softmax router.
    Supports optional global routing weights from GNN Global Router.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expert_num: int = 4,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        original_linear: nn.Linear = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.expert_num = expert_num
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Original frozen weight and bias
        self.weight = nn.Parameter(
            original_linear.weight.data.clone() if original_linear is not None
            else torch.empty(out_features, in_features),
            requires_grad=False,
        )
        if original_linear is not None and original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.data.clone(), requires_grad=False
            )
        else:
            self.bias = None

        # LoRA experts
        self.experts = nn.ModuleList([
            MoELoRAExpert(in_features, out_features, r)
            for _ in range(expert_num)
        ])

        # Instance Router: per-token softmax over experts
        self.router = nn.Linear(in_features, expert_num, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # External global weights (set before forward, cleared after)
        self._global_weights = None  # [B, expert_num]
        self._beta = 0.0

    def set_global_weights(self, w_ma: torch.Tensor, beta: float = 0.3):
        """Set external global routing weights from GNN Router."""
        self._global_weights = w_ma
        self._beta = beta

    def clear_global_weights(self):
        """Clear external global routing weights after forward pass."""
        self._global_weights = None
        self._beta = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear output (frozen)
        result = F.linear(x, self.weight, self.bias)

        # Instance Router weights: [batch, seq_len, expert_num]
        instance_weights = F.softmax(self.router(x), dim=-1)

        # Blend with GNN Global Router weights if available
        if self._global_weights is not None and self._beta > 0.0:
            # w_ma: [B, expert_num] → [B, 1, expert_num] to broadcast across seq_len
            global_w = self._global_weights.unsqueeze(1)
            router_weights = self._beta * global_w + (1.0 - self._beta) * instance_weights
        else:
            router_weights = instance_weights

        # Expert outputs weighted by blended router
        x_drop = self.lora_dropout(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x_drop)  # [batch, seq_len, out_features]
            result = result + expert_out * self.scaling * router_weights[:, :, i:i+1]

        return result


def inject_moe_lora(
    lang_model: nn.Module,
    target_modules: list = None,
    expert_num: int = 4,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    """
    Inject MoE-LoRA layers into LLaMA's Transformer layers.

    Replaces specified Linear modules (e.g., q_proj, v_proj) in each
    decoder layer with MoELoRALayer.

    Args:
        lang_model: The ModifiedLlamaForCausalLM instance.
        target_modules: List of module names to replace (e.g., ["q_proj", "v_proj"]).
        expert_num: Number of LoRA experts per layer.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout rate for LoRA.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    count = 0
    for layer_idx, layer in enumerate(lang_model.model.layers):
        for module_name in target_modules:
            # Navigate to the correct sub-module
            if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parent = layer.self_attn
            elif module_name in ("gate_proj", "up_proj", "down_proj"):
                parent = layer.mlp
            else:
                raise ValueError(f"Unknown target module: {module_name}")

            original_linear = getattr(parent, module_name)
            if not isinstance(original_linear, nn.Linear):
                continue

            # Create MoE-LoRA replacement
            moe_lora_layer = MoELoRALayer(
                in_features=original_linear.in_features,
                out_features=original_linear.out_features,
                expert_num=expert_num,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                original_linear=original_linear,
            ).to(original_linear.weight.device).to(original_linear.weight.dtype)

            # Replace the original module
            setattr(parent, module_name, moe_lora_layer)
            count += 1

    return count


def set_all_global_weights(model: nn.Module, w_ma: torch.Tensor, beta: float = 0.3):
    """Set GNN Router weights on ALL MoELoRALayer instances in the model."""
    for module in model.modules():
        if isinstance(module, MoELoRALayer):
            module.set_global_weights(w_ma, beta)


def clear_all_global_weights(model: nn.Module):
    """Clear GNN Router weights from ALL MoELoRALayer instances."""
    for module in model.modules():
        if isinstance(module, MoELoRALayer):
            module.clear_global_weights()


def freeze_all_except_moe_lora(model: nn.Module):
    """
    Freeze all parameters, then unfreeze only MoE-LoRA parameters
    (experts + routers) and GNN Router parameters.
    """
    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: unfreeze MoE-LoRA experts, routers, and GNN router
    for name, param in model.named_parameters():
        if 'experts.' in name or 'router.' in name or 'gnn_router.' in name:
            param.requires_grad = True


def get_moe_lora_state_dict(model: nn.Module):
    """Extract only MoE-LoRA + GNN Router parameters for lightweight saving."""
    state_dict = {}
    for name, param in model.named_parameters():
        if 'experts.' in name or 'router.' in name or 'gnn_router.' in name:
            state_dict[name] = param.data
    return state_dict

