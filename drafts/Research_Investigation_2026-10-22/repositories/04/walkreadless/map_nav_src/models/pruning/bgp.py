import torch
from typing import Callable, Tuple
import random
from torch.nn.utils.rnn import pad_sequence

def extend_neg_masks(masks, dtype=None):
    """
    mask from (N, L) into (N, 1(H), 1(L), L) and make it negative
    """
    if dtype is None:
        dtype = torch.float
    extended_masks = masks.unsqueeze(1).unsqueeze(2)
    extended_masks = extended_masks.to(dtype=dtype)
    extended_masks = (1.0 - extended_masks) * -10000.0
    return extended_masks

def pad_tensors_wgrad(tensors, lens=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    batch_size = len(tensors)
    hid = list(tensors[0].size()[1:])

    device = tensors[0].device
    dtype = tensors[0].dtype

    output = []
    for i in range(batch_size):
        if lens[i] < max_len:
            tmp = torch.cat(
                [tensors[i], torch.zeros([max_len-lens[i]]+hid, dtype=dtype).to(device)],
                dim=0
            )
        else:
            tmp = tensors[i]
        output.append(tmp)
    output = torch.stack(output, 0)
    return output

def merge_wavg(
     x: torch.Tensor, unm_idx, src_idx, dst_idx, r,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """

    # size = torch.ones_like(x[0, None])
    size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, unm_idx, src_idx, dst_idx, r, mode="sum")
    # raise Exception

    size = merge(size, unm_idx, src_idx, dst_idx, r, mode="sum")

    x = x / size

    return x, size

def merge(x: torch.Tensor, unm_idx, src_idx, dst_idx, r, mode="mean") -> torch.Tensor:

    
    temp = unm_idx.size(0) + src_idx.size(0)
    candidate_n = x.size(0) - temp

    src = x[...,::2,:]
    dst = x[...,1::2,:]


    t1, c = src.shape

    unm = src.gather(dim=-2, index=unm_idx.expand(t1 - r, c))
    src = src.gather(dim=-2, index=src_idx.expand(r, c))
    dst = dst.scatter_reduce(-2, dst_idx.expand(r, c), src, reduce=mode)


    result = torch.cat([unm, dst], dim=0)
    return result

def background_pruning(attn_scores, cand_vpids, vp_img_ids,
                       src_key_padding_mask, pano_img_feat,
                       r, method='tome', feat_key=None, efficiency_count=False):
    """
    Prunes background (non-candidate) image tokens based on attention scores or token merging.

    Args:
        attn_scores: [B, N, N] attention matrix.
        cand_vpids: List[List[str]]; candidate viewpoint ids for each batch element.
        vp_img_ids: List[Tensor]; image ids associated with each token in each batch.
        src_key_padding_mask: [B, N] mask where True indicates padding.
        pano_img_feat: [N, B, D] image features (transposed later to [B, N, D]).
        r: Number of tokens to prune.
        method: Strategy name ("attn_scores", "random", "evit", "FastV", "tome", "action_included").
        feat_key: Additional key for merging in ToMe mode.
        efficiency_count: Shortcut mode to simulate pruning without ranking or merging logic.

    Returns:
        Tuple of padded tensors:
            - pano_img_feat_batch: [B, max_len, D]
            - nav_types_batch: [B, max_len]
            - view_lens_batch: [B]
            - mask_batch: [B, max_len]
            - vp_img_ids_batch: [B, max_len]
    """

    pano_img_feat = pano_img_feat.permute(1, 0, 2)  # Transpose to [B, N, D]
    batch_size = pano_img_feat.size(0)

    # Accumulators for batch outputs
    pano_img_feat_batch, nav_types_batch = [], []
    view_lens_batch, vp_img_ids_batch, mask_batch = [], [], []

    for i in range(batch_size):
        mask_list = src_key_padding_mask[i].view(-1).tolist()
        valid_end_idx = mask_list.index(True) if True in mask_list else len(mask_list)

        num_cand = len(cand_vpids[i])  # Number of candidate viewpoints
        cand_feats = pano_img_feat[i][:num_cand]  # Candidate tokens (always kept)
        bg_feats = pano_img_feat[i][num_cand:valid_end_idx]  # Background tokens to prune
        vp_ids_all = vp_img_ids[i][:valid_end_idx]  # All image ids up to valid region

        # Efficiency count mode: skip scoring and just trim fixed tokens
        if efficiency_count:
            if method != "evit":
                trimmed_bg = pano_img_feat[i][num_cand:-r] if r > 0 else pano_img_feat[i][num_cand:]
            else:
                trimmed_bg = pano_img_feat[i][num_cand:-(r-1)] if r > 1 else pano_img_feat[i][num_cand:]

            merged_feats = torch.cat([cand_feats, trimmed_bg], dim=0)
            merged_ids = vp_ids_all[:len(merged_feats)]

        # "action_included" method: ranks and prunes across all tokens, not just background
        elif method == "action_included":
            scores = attn_scores[i][:valid_end_idx, :valid_end_idx].sum(dim=0)
            ranked = torch.argsort(scores, descending=True)
            keep_idx = ranked[:-r] if r > 0 else ranked
            keep_idx = torch.tensor(sorted(keep_idx.tolist()), device=pano_img_feat.device)

            merged_feats = pano_img_feat[i].index_select(0, keep_idx)
            merged_ids = vp_ids_all.index_select(0, keep_idx)

        # ToMe merging strategy: merge background tokens based on key similarity
        elif method == "tome":
            assert feat_key is not None, "feat_key is required for ToMe"
            bg_key = feat_key[i][num_cand:valid_end_idx]
            normed_key = bg_key / bg_key.norm(dim=-1, keepdim=True)
            a, b = normed_key[::2], normed_key[1::2]
            sim_scores = a @ b.transpose(-1, -2)
            node_max, node_idx = sim_scores.max(dim=-1)
            edge_idx = node_max.argsort(descending=True)[..., None]

            # Split to be merged/unmerged
            r_prune = min(r, len(edge_idx))
            unm_idx = edge_idx[..., r_prune:, :]
            src_idx = edge_idx[..., :r_prune, :]
            dst_idx = node_idx[..., None].gather(-2, index=src_idx)

            merged_bg, _ = merge_wavg(bg_feats, unm_idx, src_idx, dst_idx, r=r_prune)
            merged_feats = torch.cat([cand_feats, merged_bg], dim=0)
            merged_ids = vp_ids_all[:num_cand + len(merged_bg)]

        # All other methods: pruning based on attention scores or random
        else:
            attn_i = attn_scores[i][:valid_end_idx, :valid_end_idx]
            scores = attn_i.sum(dim=0)[num_cand:]  # Scores of background tokens only
            ranked = torch.randperm(len(scores)) if method == "random" else torch.argsort(scores, descending=True)
            if r > 0:
                if method == "evit":
                    pruned = ranked[-r:]
                    keep = [idx for idx in ranked if idx not in pruned]
                    fused = (bg_feats[pruned] * scores[pruned].unsqueeze(-1)).sum(dim=0, keepdim=True)

                    bg_kept = bg_feats[keep]
                    ids_kept = vp_ids_all[num_cand + torch.tensor(keep)]

                    # Add fused token to the end
                    merged_feats = torch.cat([cand_feats, bg_kept, fused], dim=0)
                    merged_ids = torch.cat([vp_ids_all[:num_cand], ids_kept,
                                            torch.tensor([ids_kept.max() + 1], device=ids_kept.device)])
                else:
                    keep = sorted(ranked[:-r].tolist())
                    bg_kept = bg_feats[keep]
                    ids_kept = vp_ids_all[num_cand + torch.tensor(keep)]
                    merged_feats = torch.cat([cand_feats, bg_kept], dim=0)
                    merged_ids = torch.cat([vp_ids_all[:num_cand], ids_kept])
            else:
                merged_feats = torch.cat([cand_feats, bg_feats], dim=0)
                merged_ids = vp_ids_all

        # Create navigation types: 1 for candidate, 0 for background
        nav_types = torch.tensor([1] * num_cand + [0] * (len(merged_feats) - num_cand),
                                 device=merged_feats.device)
        # Create new src key padding mask (False = keep, True = pad)
        mask = torch.tensor(mask_list[:len(merged_feats)],
                            dtype=torch.bool, device=merged_feats.device).logical_not()

        # Append pruned sample outputs
        pano_img_feat_batch.append(merged_feats)
        nav_types_batch.append(nav_types)
        view_lens_batch.append(len(merged_feats))
        vp_img_ids_batch.append(merged_ids)
        mask_batch.append(mask)

    # Pad all batches to uniform length
    pano_img_feat_batch = pad_tensors_wgrad(pano_img_feat_batch)
    nav_types_batch = pad_tensors_wgrad(nav_types_batch).cuda()
    view_lens_batch = torch.LongTensor(view_lens_batch)
    mask_batch = pad_tensors_wgrad(mask_batch).logical_not().cuda()
    vp_img_ids_batch = pad_sequence(vp_img_ids_batch, batch_first=True, padding_value=-1).cuda()

    return pano_img_feat_batch, nav_types_batch, view_lens_batch, mask_batch, vp_img_ids_batch



def background_pruning_reverie(attn_scores, cand_vpids, vp_img_ids, \
                      src_key_padding_mask, pano_img_feat, nav_types, view_lens,\
                      r, method = 'tome', feat_key = None, efficiency_count = False):
    # print("key.shape", key.shape)
    # print("method",method)
    # print("r",r)
    # r = 5

    batch_size = attn_scores.size(0)

    # print("nav_types.shape",nav_types.shape)
    # print("pano_img_feat.shape", pano_img_feat.shape)
    # print("pano_img_feat.shape", pano_img_feat.shape)

    pano_img_feat = pano_img_feat.permute(1,0,2)

    valid_token_end_idx_batch = []

    pano_img_feat_batch = []
    hidden_states_batch = []
    merge_src_key_padding_mask_batch = []
    merge_nav_types_batch = []
    merge_view_lens_batch = []
    merge_vp_img_ids_i_batch = []


    for i in range(batch_size):
        if efficiency_count:
            # print("pano_img_feat.shape",pano_img_feat.shape)
            # print("cand_vpids",cand_vpids)
            candidate_img_feat_i = pano_img_feat[i][:len(cand_vpids[i]),:]
            if method != "evit" and method != "tome":
                if r > 0:
                    merge_pano_img_feat_i = pano_img_feat[i][len(cand_vpids[i]):-r,:]
                else:
                    merge_pano_img_feat_i = pano_img_feat[i][len(cand_vpids[i]):,:]
            elif method == "tome":
                if r > 0:
                    merge_pano_img_feat_i = pano_img_feat[i][len(cand_vpids[i]):-r,:]
                else:
                    merge_pano_img_feat_i = pano_img_feat[i][len(cand_vpids[i]):,:]
            elif method == "evit":
                if r > 1:
                    merge_pano_img_feat_i = pano_img_feat[i][len(cand_vpids[i]):-(r-1),:]
                else:
                    merge_pano_img_feat_i = pano_img_feat[i][len(cand_vpids[i]):,:]
            else:
                raise NotImplementedError
            # print("candidate_img_feat_i.shape",candidate_img_feat_i.shape)
            # print("merge_pano_img_feat_i.shape",merge_pano_img_feat_i.shape)    

            merge_pano_img_feat_i = torch.cat([candidate_img_feat_i, merge_pano_img_feat_i],dim=0)
            merge_view_lens_i = len(merge_pano_img_feat_i)
            merge_nav_types_i = torch.LongTensor([1]*len(cand_vpids[i]) + [0] * (len(merge_pano_img_feat_i) - len(cand_vpids[i])))
            merge_src_key_padding_mask_i = src_key_padding_mask[i][:len(merge_pano_img_feat_i)]
            merge_vp_img_ids_i = vp_img_ids[i][:len(merge_pano_img_feat_i)]

            pano_img_feat_batch.append(merge_pano_img_feat_i)
            merge_view_lens_batch.append(merge_view_lens_i)
            merge_nav_types_batch.append(merge_nav_types_i)
            merge_src_key_padding_mask_batch.append(merge_src_key_padding_mask_i)
            merge_vp_img_ids_i_batch.append(merge_vp_img_ids_i)

        elif method == "action_included":
            src_key_padding_mask_i_list = src_key_padding_mask[i].tolist()
            
            if True in src_key_padding_mask_i_list:
                valid_token_end_idx  = src_key_padding_mask_i_list.index(True)
            else:
                valid_token_end_idx = len(src_key_padding_mask_i_list)

            nav_types_i_list = nav_types[i][:valid_token_end_idx]

            obj_index = -1
            for j,type in enumerate(nav_types_i_list):
                if type==2:
                    obj_index = j
                    break
            
            if obj_index == -1:
                attn_scores_i = attn_scores[i][:valid_token_end_idx, :valid_token_end_idx]
                obj_img_feat_i = pano_img_feat[i][valid_token_end_idx:valid_token_end_idx,:]
            else:
                attn_scores_i = attn_scores[i][:obj_index, :obj_index]
                obj_img_feat_i = pano_img_feat[i][obj_index:valid_token_end_idx,:]
            

            attn_scores_ranking = torch.sum(attn_scores_i,dim=0)
            attentive_scores = attn_scores_ranking # [len(cand_vpids[i]):] including actions


            if obj_index == -1:
               non_candidate_vp_img_ids_i = vp_img_ids[i][:valid_token_end_idx]
            else:
               non_candidate_vp_img_ids_i = vp_img_ids[i][:obj_index]

            # candidate_img_feat_i = pano_img_feat[i][:len(cand_vpids[i]),:]

            if obj_index == -1:
                pano_img_feat_i_to_merge = pano_img_feat[i][:valid_token_end_idx,:]
            else:
                pano_img_feat_i_to_merge = pano_img_feat[i][:obj_index,:]

            attentive_ranking = torch.argsort(attentive_scores,descending=True)

            assert len(non_candidate_vp_img_ids_i) == len(attentive_ranking), f"{len(non_candidate_vp_img_ids_i)}!={len(attentive_ranking)}"

            if r>0:
                attentive_token_idxes = attentive_ranking[:-r]
            else:
                attentive_token_idxes = attentive_ranking
            
            # print("attentive_ranking",attentive_ranking)
            # print("attentive_token_idxes",attentive_token_idxes)

            attentive_token_idxes = torch.LongTensor(sorted(attentive_token_idxes.tolist())).cuda()

            attentive_non_candidate_vp_img_ids_i = non_candidate_vp_img_ids_i[attentive_token_idxes]

            attentive_img_feat_i = torch.index_select(pano_img_feat_i_to_merge,0,attentive_token_idxes)

            candidate_vp_img_ids_i = vp_img_ids[i][:len(cand_vpids[i])]
            merge_pano_img_feat_i = torch.cat([attentive_img_feat_i, obj_img_feat_i],dim=0)

            merge_nav_types_i = []
            for vp_imd_id in attentive_non_candidate_vp_img_ids_i:
                if vp_imd_id in candidate_vp_img_ids_i:
                    merge_nav_types_i.append(1)
                else:
                    merge_nav_types_i.append(0)
            merge_nav_types_i += [2] * len(obj_img_feat_i)
            merge_nav_types_i = torch.LongTensor(merge_nav_types_i)
            # print("merge_nav_types_i",merge_nav_types_i)
            # merge_nav_types_i = torch.LongTensor([1]*(len(cand_vpids[i])) + [0] * (len(merge_pano_img_feat_i) \
            #                                             - len(cand_vpids[i]) - len(obj_img_feat_i)) + [2] * len(obj_img_feat_i))
            merge_view_lens_i = len(merge_pano_img_feat_i)

            merge_vp_img_ids_i = attentive_non_candidate_vp_img_ids_i # torch.cat([vp_img_ids[i][:len(cand_vpids[i])], attentive_non_candidate_vp_img_ids_i],dim=0)
            merge_src_key_padding_mask_i = torch.LongTensor(src_key_padding_mask_i_list[:len(merge_pano_img_feat_i)]).logical_not()

            pano_img_feat_batch.append(merge_pano_img_feat_i)
            merge_view_lens_batch.append(merge_view_lens_i)
            merge_nav_types_batch.append(merge_nav_types_i)
            merge_src_key_padding_mask_batch.append(merge_src_key_padding_mask_i)
            merge_vp_img_ids_i_batch.append(merge_vp_img_ids_i)


        elif method == "tome":
            src_key_padding_mask_i_list = src_key_padding_mask[i].tolist()
            
            if True in src_key_padding_mask_i_list:
                valid_token_end_idx  = src_key_padding_mask_i_list.index(True)
            else:
                valid_token_end_idx = len(src_key_padding_mask_i_list)

            nav_types_i_list = nav_types[i][:valid_token_end_idx]
    
            obj_index = -1
            for j,type in enumerate(nav_types_i_list):
                if type==2:
                    obj_index = j
                    break

            if obj_index == -1:
                attn_scores_i = attn_scores[i][:valid_token_end_idx, :valid_token_end_idx]
                obj_img_feat_i = pano_img_feat[i][valid_token_end_idx:valid_token_end_idx,:]
                key_to_merge = feat_key[i][len(cand_vpids[i]):valid_token_end_idx,:]
            else:
                attn_scores_i = attn_scores[i][:obj_index, :obj_index]
                obj_img_feat_i = pano_img_feat[i][obj_index:valid_token_end_idx,:]
                key_to_merge = feat_key[i][len(cand_vpids[i]):obj_index,:]

            attn_scores_ranking = torch.sum(attn_scores_i,dim=0)
            attentive_scores = attn_scores_ranking[len(cand_vpids[i]):]

            if obj_index == -1:
               non_candidate_vp_img_ids_i = vp_img_ids[i][len(cand_vpids[i]):valid_token_end_idx]
            else:
               non_candidate_vp_img_ids_i = vp_img_ids[i][len(cand_vpids[i]):obj_index]

            candidate_img_feat_i = pano_img_feat[i][:len(cand_vpids[i]),:]

            if obj_index == -1:
                pano_img_feat_i_to_merge = pano_img_feat[i][len(cand_vpids[i]):valid_token_end_idx,:]
            else:
                pano_img_feat_i_to_merge = pano_img_feat[i][len(cand_vpids[i]):obj_index,:]

            
            metric = key_to_merge
            assert len(metric) == len(pano_img_feat_i_to_merge)

            metric = metric / metric.norm(dim=-1, keepdim=True)

            a, b = metric[::2, :], metric[1::2, :]
            scores = a @ b.transpose(-1, -2)


            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            r_prune = min(r, len(edge_idx))
            unm_idx = edge_idx[..., r_prune:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r_prune, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
            
            attentive_img_feat_i, merge_size = merge_wavg(pano_img_feat_i_to_merge, unm_idx, src_idx, dst_idx, r = r_prune)
            merge_pano_img_feat_i = torch.cat([candidate_img_feat_i, attentive_img_feat_i, obj_img_feat_i],dim=0)
            merge_nav_types_i = torch.LongTensor([1]*(len(cand_vpids[i])) + [0] * (len(merge_pano_img_feat_i) \
                                                        - len(cand_vpids[i]) - len(obj_img_feat_i)) + [2] * len(obj_img_feat_i))
            merge_view_lens_i = len(merge_pano_img_feat_i)
            merge_vp_img_ids_i = vp_img_ids[i][:len(cand_vpids[i])+len(attentive_img_feat_i)]
            merge_src_key_padding_mask_i = torch.LongTensor(src_key_padding_mask_i_list[:len(merge_pano_img_feat_i)]).logical_not()
            

            pano_img_feat_batch.append(merge_pano_img_feat_i)
            merge_view_lens_batch.append(merge_view_lens_i)
            merge_nav_types_batch.append(merge_nav_types_i)
            merge_src_key_padding_mask_batch.append(merge_src_key_padding_mask_i)
            merge_vp_img_ids_i_batch.append(merge_vp_img_ids_i)

        elif method == "attn_scores" or method == "random" or method == "evit" or method == "FastV":
            
            src_key_padding_mask_i_list = src_key_padding_mask[i].tolist()
            
            if True in src_key_padding_mask_i_list:
                valid_token_end_idx  = src_key_padding_mask_i_list.index(True)
            else:
                valid_token_end_idx = len(src_key_padding_mask_i_list)

            nav_types_i_list = nav_types[i][:valid_token_end_idx]

            obj_index = -1
            for j,type in enumerate(nav_types_i_list):
                if type==2:
                    obj_index = j
                    break
            
            if obj_index == -1:
                attn_scores_i = attn_scores[i][:valid_token_end_idx, :valid_token_end_idx]
                obj_img_feat_i = pano_img_feat[i][valid_token_end_idx:valid_token_end_idx,:]
            else:
                attn_scores_i = attn_scores[i][:obj_index, :obj_index]
                obj_img_feat_i = pano_img_feat[i][obj_index:valid_token_end_idx,:]
            

            attn_scores_ranking = torch.sum(attn_scores_i,dim=0)
            

            attentive_scores = attn_scores_ranking[len(cand_vpids[i]):]

            if obj_index == -1:
               non_candidate_vp_img_ids_i = vp_img_ids[i][len(cand_vpids[i]):valid_token_end_idx]
            else:
               non_candidate_vp_img_ids_i = vp_img_ids[i][len(cand_vpids[i]):obj_index]

            candidate_img_feat_i = pano_img_feat[i][:len(cand_vpids[i]),:]

            if obj_index == -1:
                pano_img_feat_i_to_merge = pano_img_feat[i][len(cand_vpids[i]):valid_token_end_idx,:]
            else:
                pano_img_feat_i_to_merge = pano_img_feat[i][len(cand_vpids[i]):obj_index,:]

            attentive_ranking = torch.argsort(attentive_scores,descending=True)

            if method == "random":
                attentive_ranking = torch.randperm(len(attentive_ranking)).cuda()

            # print("obj_index",obj_index)
            # print("nav_types_i_list",nav_types_i_list)
            # print("non_candidate_vp_img_ids_i",non_candidate_vp_img_ids_i)
            # print("attentive_ranking.shape",attentive_ranking.shape)
            # print("")
            assert len(non_candidate_vp_img_ids_i) == len(attentive_ranking), f"{len(non_candidate_vp_img_ids_i)}!={len(attentive_ranking)}"

            if r>0:
                attentive_token_idxes = attentive_ranking[:-r]
            else:
                attentive_token_idxes = attentive_ranking
            
            if method == "evit":
                if r>0:
                    unattentive_token_idxes = attentive_ranking[-r:]
                else:
                    unattentive_token_idxes = torch.LongTensor([]).cuda()

                # print("unattentive_token_idxes",unattentive_token_idxes)
                # print("len(unattentive_token_idxes)",len(unattentive_token_idxes))

                attentive_token_idxes = [int(idx) for idx in attentive_ranking if idx not in unattentive_token_idxes]

                # print("attentive_token_idxes",attentive_token_idxes)
                # print("len(attentive_token_idxes)",len(attentive_token_idxes))

                attentive_non_candidate_vp_img_ids_i = non_candidate_vp_img_ids_i[attentive_token_idxes]
                attentive_non_candidate_vp_img_ids_i = torch.sort(attentive_non_candidate_vp_img_ids_i)[0]

                attentive_img_feat_i = torch.index_select(pano_img_feat_i_to_merge,0,torch.LongTensor(attentive_token_idxes).cuda())
                unattentive_img_feat_i = torch.index_select(pano_img_feat_i_to_merge,0,unattentive_token_idxes)

                unattentive_attn_scores = torch.index_select(attentive_scores,0,unattentive_token_idxes)
                unattentive_attn_scores_img_feat_i = unattentive_attn_scores.unsqueeze(-1).expand_as(unattentive_img_feat_i)    


                fused_img_feat_i = unattentive_img_feat_i * unattentive_attn_scores_img_feat_i
                fused_img_feat_i = torch.sum(fused_img_feat_i,dim=0).unsqueeze(0)

                # print("unattentive_img_feat_i.shape",unattentive_img_feat_i.shape)
                # print("attentive_img_feat_i.shape",attentive_img_feat_i.shape)
                # print("fused_img_feat_i.shape",fused_img_feat_i.shape)

            if method != "evit":
                # print("attentive_token_idxes 11",attentive_token_idxes)
                attentive_token_idxes = torch.LongTensor(sorted(attentive_token_idxes.tolist())).cuda()
                attentive_non_candidate_vp_img_ids_i = non_candidate_vp_img_ids_i[attentive_token_idxes]

                attentive_img_feat_i = torch.index_select(pano_img_feat_i_to_merge,0,attentive_token_idxes)

                merge_pano_img_feat_i = torch.cat([candidate_img_feat_i, attentive_img_feat_i, obj_img_feat_i],dim=0)
                merge_nav_types_i = torch.LongTensor([1]*(len(cand_vpids[i])) + [0] * (len(merge_pano_img_feat_i) \
                                                            - len(cand_vpids[i]) - len(obj_img_feat_i)) + [2] * len(obj_img_feat_i))
                merge_view_lens_i = len(merge_pano_img_feat_i)

                merge_vp_img_ids_i = torch.cat([vp_img_ids[i][:len(cand_vpids[i])], attentive_non_candidate_vp_img_ids_i],dim=0)
                merge_src_key_padding_mask_i = torch.LongTensor(src_key_padding_mask_i_list[:len(merge_pano_img_feat_i)]).logical_not()

                # print("merge_pano_img_feat_i.shape",merge_pano_img_feat_i.shape)
                # print("merge_view_lens_i",merge_view_lens_i)
                # print("merge_nav_types_i",merge_nav_types_i)
                # print("merge_nav_types_i.shape",merge_nav_types_i.shape)
                # print("merge_src_key_padding_mask_i.shape",merge_src_key_padding_mask_i)
                # raise Exception
                # print("len(merge_vp_img_ids_i)",len(merge_vp_img_ids_i))
                # raise Exception

                pano_img_feat_batch.append(merge_pano_img_feat_i)
                merge_view_lens_batch.append(merge_view_lens_i)
                merge_nav_types_batch.append(merge_nav_types_i)
                merge_src_key_padding_mask_batch.append(merge_src_key_padding_mask_i)
                merge_vp_img_ids_i_batch.append(merge_vp_img_ids_i)
            else:
                # merging
                if r>0:
                    merge_pano_img_feat_i = torch.cat([candidate_img_feat_i, attentive_img_feat_i, fused_img_feat_i, obj_img_feat_i],dim=0)
                    
                else:
                    merge_pano_img_feat_i = torch.cat([candidate_img_feat_i, attentive_img_feat_i, obj_img_feat_i],dim=0)
                
                merge_nav_types_i = torch.LongTensor([1]*(len(cand_vpids[i])) + [0] * (len(merge_pano_img_feat_i) \
                                                            - len(cand_vpids[i]) - len(obj_img_feat_i)) + [2] * len(obj_img_feat_i))
                merge_view_lens_i = len(merge_pano_img_feat_i)
                merge_vp_img_ids_i = torch.cat([vp_img_ids[i][:len(cand_vpids[i])], attentive_non_candidate_vp_img_ids_i],dim=0)
                if r>0:
                    merge_vp_img_ids_i = torch.LongTensor(merge_vp_img_ids_i.tolist() + [max(merge_vp_img_ids_i.tolist())+1]).cuda()
                # print("merge_vp_img_ids_i",merge_vp_img_ids_i)
                # raise Exception
                merge_src_key_padding_mask_i = torch.LongTensor(src_key_padding_mask_i_list[:len(merge_pano_img_feat_i)]).logical_not()

                # print("merge_pano_img_feat_i.shape",merge_pano_img_feat_i.shape)
                # print("merge_nav_types_i.shape",merge_nav_types_i.shape)
                # print("merge_view_lens_i.shape",merge_view_lens_i)
                # print("merge_vp_img_ids_i",merge_vp_img_ids_i)
                # print("merge_vp_img_ids_i.shape",merge_vp_img_ids_i.shape)
                # raise Exception
                pano_img_feat_batch.append(merge_pano_img_feat_i)
                merge_view_lens_batch.append(merge_view_lens_i)
                merge_nav_types_batch.append(merge_nav_types_i)
                merge_src_key_padding_mask_batch.append(merge_src_key_padding_mask_i)
                merge_vp_img_ids_i_batch.append(merge_vp_img_ids_i)


    pano_img_feat_batch = pad_tensors_wgrad(pano_img_feat_batch)
    # pano_img_feat_batch = pano_img_feat
    # pano_img_feat_last = pano_img_feat[-1]
    # pano_img_feat_batch_last = pad_tensors_wgrad(pano_img_feat_batch)[-1]
    # print("torch.sum(pano_img_feat_last)")
    # print("pano_img_feat_last.shape",pano_img_feat_last.shape)
    # print("torch.sum(pano_img_feat_last)",torch.sum(pano_img_feat_last))
    # print("pano_img_feat_batch_last.shape",pano_img_feat_batch_last.shape)
    # print("torch.sum(pano_img_feat_batch_last)",torch.sum(pano_img_feat_batch_last))
    merge_nav_types_batch = pad_tensors_wgrad(merge_nav_types_batch).cuda()
    merge_view_lens_batch = torch.LongTensor(merge_view_lens_batch).cuda()
    merge_vp_img_ids_i_batch = pad_sequence(merge_vp_img_ids_i_batch, batch_first=True, padding_value=-1).cuda()
    merge_src_key_padding_mask_batch = pad_tensors_wgrad(merge_src_key_padding_mask_batch).logical_not().cuda()
    
    # print("pano_img_feat_batch",pano_img_feat_batch)
    # raise Exception

    # print("pano_img_feat_batch",pano_img_feat_batch.shape)
    # print("merge_nav_types_batch.shape",merge_nav_types_batch.shape)
    # print("merge_nav_types_batch",merge_nav_types_batch)
    # print("merge_view_lens_batch",merge_view_lens_batch)
    # print("merge_vp_img_ids_i_batch",merge_vp_img_ids_i_batch)
    # print("merge_src_key_padding_mask_batch.shape",merge_src_key_padding_mask_batch.shape)
    # print("merge_src_key_padding_mask_batch",merge_src_key_padding_mask_batch)    
    # raise Exception

    return pano_img_feat_batch, merge_nav_types_batch, merge_view_lens_batch, merge_src_key_padding_mask_batch, merge_vp_img_ids_i_batch


