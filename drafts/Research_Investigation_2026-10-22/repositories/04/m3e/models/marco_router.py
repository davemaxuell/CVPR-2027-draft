import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):

    def __init__(self, d_h: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_h, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_h)
        self.ffn = nn.Sequential(
            nn.Linear(d_h, d_h * 4),
            nn.GELU(),
            nn.Linear(d_h * 4, d_h),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_h)

    def forward(self, H: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = (~adj_mask).to(H.dtype) * (-1e4)

        n_heads = self.attn.num_heads
        B, N = H.shape
        attn_mask = attn_mask.unsqueeze(1).expand(B, n_heads, N, N).reshape(B * n_heads, N, N)

        attn_out, _ = self.attn(H, H, H, attn_mask=attn_mask)
        H = self.norm(H + attn_out)

        H = self.ffn_norm(H + self.ffn(H))
        return H


class GNNGlobalRouter(nn.Module):

    def __init__(
        self,
        in_features: int = 4096,
        d_h: int = 256,
        expert_num: int = 6,
        n_layers: int = 2,
        n_heads: int = 4,
        dist_threshold: float = 5.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_h = d_h
        self.dist_threshold = dist_threshold

        self.node_proj = nn.Linear(in_features, d_h)
        self.nav_state_embedding = nn.Embedding(2, d_h)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(d_h, n_heads, dropout) for _ in range(n_layers)
        ])

        self.ins_proj = nn.Linear(in_features, d_h)

        self.route_mlp = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, expert_num),
        )

    def forward(
        self,
        gmap_embeds: torch.Tensor,
        gmap_visited_masks: torch.Tensor,
        gmap_pair_dists: torch.Tensor,
        gmap_masks: torch.Tensor,
        instruction_embeds: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.node_proj.weight.dtype
        gmap_embeds = gmap_embeds.to(target_dtype)
        instruction_embeds = instruction_embeds.to(target_dtype)

        H = self.node_proj(gmap_embeds)
        H = H + self.nav_state_embedding(gmap_visited_masks.long())

        adj_mask = (gmap_pair_dists > 0) & (gmap_pair_dists < self.dist_threshold)
        B, N = gmap_masks.shape
        self_loop = torch.eye(N, dtype=torch.bool, device=gmap_masks.device).unsqueeze(0).expand(B, -1, -1)
        adj_mask = (adj_mask | self_loop)
        adj_mask = adj_mask & gmap_masks.unsqueeze(1) & gmap_masks.unsqueeze(2)

        for gnn_layer in self.gnn_layers:
            H = gnn_layer(H, adj_mask)

        ins_emb = self.ins_proj(instruction_embeds)
        attn_scores = (H * ins_emb.unsqueeze(1)).sum(dim=-1)
        attn_scores = attn_scores.masked_fill(~gmap_masks, float('-inf'))
        alpha = F.softmax(attn_scores, dim=-1)
        s_t = (alpha.unsqueeze(-1) * H).sum(dim=1)

        w_ma = F.softmax(self.route_mlp(s_t), dim=-1)

        return w_ma
