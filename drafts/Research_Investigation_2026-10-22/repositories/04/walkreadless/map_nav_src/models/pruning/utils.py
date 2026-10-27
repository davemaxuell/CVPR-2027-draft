import torch

def get_txt_attn_scores_batch_sorted_indexes(txt_attn_scores):
    """
    Used for VPP.
    Computes token importance ranking from attention scores by averaging across heads and queries,
    and sorts token indices in descending order of importance for each sample in the batch.
    """

    # Clone and reshape: [B, H, N, N] -> [B, N, N, H] -> sum over heads and queries
    attn = txt_attn_scores.clone().permute(0, 3, 2, 1)  # [B, N, N, H]
    attn_sum = attn.sum(dim=-1).sum(dim=-1)            # [B, N]
    attn_avg = attn_sum / (attn.shape[-1] * attn.shape[-2])  # mean over heads and queries

    # Sort tokens by descending attention importance
    sorted_indices = torch.argsort(attn_avg, dim=-1, descending=True)

    # Ensure [CLS] token (index 0) is always at the front if it exists but not at front
    for i in range(sorted_indices.size(0)):
        if 0 in sorted_indices[i] and sorted_indices[i][0] != 0:
            idx_list = sorted_indices[i].tolist()
            idx_list.remove(0)
            idx_list.insert(0, 0)
            sorted_indices[i] = torch.tensor(idx_list, device=txt_attn_scores.device)

    return sorted_indices