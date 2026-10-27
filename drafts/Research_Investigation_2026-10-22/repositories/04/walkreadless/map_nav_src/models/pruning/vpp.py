import torch
from ..ops import pad_tensors_wgrad, extend_neg_masks

def vocabulary_priority_pruning(
        tok, txt_embeds, txt_ids, txt_masks, 
        sorted_indices, n_tokens_to_prune_batch, txt_prune_strat, seq_lengths, 
        irrelevant_words_id=None, return_pruned_text=False
    ):
    """
    Prunes token embeddings and masks based on a sorted importance score and an optional vocabulary-aware strategy.
    Optionally returns the resulting text string after pruning for visualization/debugging.

    Args:
        tok: tokenizer used to decode pruned text.
        txt_embeds: [B, L, D] token embeddings for each sentence.
        txt_ids: [B, L] token ids.
        txt_masks: [B, 1, 1, L] attention masks.
        sorted_indices: [B, L] token indices sorted by descending importance.
        n_tokens_to_prune_batch: [B] number of tokens to prune per sample.
        txt_prune_strat: pruning strategy name, checks for 'vocabulary' keyword.
        seq_lengths: [B] original sequence lengths.
        irrelevant_words_id: set of token ids considered irrelevant (used if 'vocabulary' is in strategy).
        return_pruned_text: if True, decodes and returns the pruned instruction as text.

    Returns:
        Pruned versions of txt_embeds, txt_masks, txt_ids, and seq_lengths.
        Optionally, the decoded pruned instruction string.
    """
    indexes_to_be_selected_batch = []
    txt_embeds_c = txt_embeds.clone()
    batch_size = txt_embeds.size(0)
    pruned_texts = []  # Store decoded strings if enabled

    for i in range(batch_size):
        keep_len = seq_lengths[i] - n_tokens_to_prune_batch[i]  # Number of tokens to keep
        idxs_sorted = sorted_indices[i]

        # If using vpp, reorder indices by relevant then irrelevant
        if "vocabulary" in txt_prune_strat:
            relevant, irrelevant = [], []
            for idx in idxs_sorted:
                token_id = int(txt_ids[i][idx])
                if (token_id != 0 and token_id not in irrelevant_words_id) or idx == 0:
                    relevant.append(idx)
                else:
                    irrelevant.append(idx)
            idxs_sorted = relevant + irrelevant  # Relevant first
        else:
            idxs_sorted = [int(idx) for idx in idxs_sorted]  # convert tensor to list

        # Select top-K kept tokens and sort them to maintain original order
        selected = sorted(idxs_sorted[:keep_len])
        indexes_to_be_selected_batch.append(torch.LongTensor(selected).to(txt_embeds.device))
        
        # Decode and store the pruned text if required
        if return_pruned_text:
            selected_tensor = torch.tensor(selected, dtype=torch.long, device=txt_ids[i].device)
            pruned_token_ids = torch.index_select(txt_ids[i], 0, selected_tensor)
            pruned_text = tok.decode(pruned_token_ids, skip_special_tokens=True).strip()
            pruned_texts.append(pruned_text)

    # Apply pruning to all sequences
    txt_embeds_p, txt_masks_p, txt_ids_p, seq_lengths_p = [], [], [], []

    for i in range(batch_size):
        selected_idxs = indexes_to_be_selected_batch[i]
        
        # Select tokens
        txt_embeds_i = torch.index_select(txt_embeds_c[i], 0, selected_idxs)
        txt_ids_i = torch.index_select(txt_ids[i], 0, selected_idxs)

        # Prune attention masks
        mask_i = txt_masks[i].view(-1)
        txt_masks_i = torch.index_select(mask_i, 0, selected_idxs).logical_not()  # Invert for mask

        # Append outputs
        txt_embeds_p.append(txt_embeds_i)
        txt_masks_p.append(txt_masks_i)
        txt_ids_p.append(txt_ids_i)
        seq_lengths_p.append(len(selected_idxs))

    # Pad all sequences to match the longest one
    txt_embeds_p = pad_tensors_wgrad(txt_embeds_p)
    txt_masks_p = pad_tensors_wgrad(txt_masks_p)
    txt_masks_p = extend_neg_masks(txt_masks_p)  # Final formatting
    txt_ids_p = pad_tensors_wgrad(txt_ids_p)

    if return_pruned_text:
        return txt_embeds_p, txt_masks_p, txt_ids_p, seq_lengths_p, pruned_texts
    else:
        return txt_embeds_p, txt_masks_p, txt_ids_p, seq_lengths_p