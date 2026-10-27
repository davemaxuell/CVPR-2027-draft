import numpy as np
import math

import torch
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad


class GMapNavAgent(Seq2SeqAgent):

    def _build_model(self):
        self.vln_bert = VLNBert(self.args).to(self.device)
        self.critic = Critic(self.args).to(self.device)
        self.scanvp_cands = {}

    def _language_variable(self, obs, instr_dict, front_txt_feats):
        # Prepare token sequences and masks for instructions
        seq_lengths = [len(ob["instr_encoding"]) for ob in obs]
        batch_size = len(obs)

        # Pad sequences to maximum length
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, : seq_lengths[i]] = ob["instr_encoding"]
            mask[i, : seq_lengths[i]] = True

        # Convert to torch tensors and move to device
        seq_tensor = torch.from_numpy(seq_tensor).long().to(self.device)
        mask = torch.from_numpy(mask).to(self.device)

        # Expand z-instruction features if provided
        instr_z_direction_features, instr_z_direction_pzs = None, None
        instr_z_landmark_features, instr_z_landmark_pzs = None, None
        if instr_dict is not None:
            instr_z_direction_features = instr_dict["instr_direction_features"].unsqueeze(0).expand(batch_size, -1, -1)
            instr_z_direction_pzs = instr_dict["instr_direction_pzs"].expand(batch_size, -1).unsqueeze(2)
            instr_z_landmark_features = instr_dict["instr_landmark_features"].unsqueeze(0).expand(batch_size, -1, -1)
            instr_z_landmark_pzs = instr_dict["instr_landmark_pzs"].expand(batch_size, -1).unsqueeze(2)

        # Return packed dictionary for VLN-BERT language input
        return {
            "txt_ids": seq_tensor,
            "txt_masks": mask,
            "instr_z_direction_features": instr_z_direction_features,
            "instr_z_direction_pzs": instr_z_direction_pzs,
            "instr_z_landmark_features": instr_z_landmark_features,
            "instr_z_landmark_pzs": instr_z_landmark_pzs,
            "front_txt_feats": front_txt_feats,
            "seq_lengths": seq_lengths,
            "irrelevant_words_id": self.irrelevant_words_id,
        }

    def _panorama_feature_variable_do(self, obs, img_zdict=None):
        """Extracts panoramic image and location features for each observation."""

        batch_size = len(obs)
        batch_view_img_fts, batch_loc_fts = [], []
        batch_nav_types, batch_vp_img_ids = [], []
        batch_view_lens, batch_cand_vpids = [], []
        z_img_features, z_img_pzs = None, None

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types = [], [], []
            cand_vpids, vp_img_ids = [], []
            used_viewidxs = set()

            # Handle candidate views (where the agent can navigate)
            for j, cc in enumerate(ob["candidate"]):
                view_img_fts.append(cc["feature"][: self.args.image_feat_size])
                view_ang_fts.append(cc["feature"][self.args.image_feat_size :])
                nav_types.append(1)
                cand_vpids.append(cc["viewpointId"])
                used_viewidxs.add(cc["pointId"])
                vp_img_ids.append(cc["pointId"])

            # Add remaining non-candidate views (for visual context)
            view_img_fts.extend([
                x[:self.args.image_feat_size]
                for k, x in enumerate(ob["feature"]) if k not in used_viewidxs
            ])
            view_ang_fts.extend([
                x[self.args.image_feat_size:]
                for k, x in enumerate(ob["feature"]) if k not in used_viewidxs
            ])
            vp_img_ids.extend([
                k for k, x in enumerate(ob["feature"]) if k not in used_viewidxs
            ])
            nav_types.extend([0] * (36 - len(used_viewidxs)))

            # Finalize tensors
            view_img_fts = np.stack(view_img_fts, 0)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            # Append to batch
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))
            batch_vp_img_ids.append(torch.LongTensor(vp_img_ids))

        # Pad sequences and move to device
        batch_view_img_fts = pad_tensors(batch_view_img_fts).to(self.device)
        batch_loc_fts = pad_tensors(batch_loc_fts).to(self.device)
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).to(self.device)
        batch_vp_img_ids = pad_sequence(batch_vp_img_ids, batch_first=True, padding_value=-1).to(self.device)
        batch_view_lens = torch.LongTensor(batch_view_lens).to(self.device)

        # Expand image z-dictionary if intervention is used
        if img_zdict is not None:
            z_img_features = img_zdict["img_features"].unsqueeze(0).expand(batch_size, -1, -1)
            z_img_pzs = img_zdict["img_pzs"].expand(batch_size, -1).unsqueeze(2)

        return {
            "view_img_fts": batch_view_img_fts,
            "loc_fts": batch_loc_fts,
            "nav_types": batch_nav_types,
            "view_lens": batch_view_lens,
            "cand_vpids": batch_cand_vpids,
            "vp_img_ids": batch_vp_img_ids,
            "z_img_features": z_img_features,
            "z_img_pzs": z_img_pzs,
        }

    def _nav_gmap_variable(self, obs, gmaps, last_embeds=None, remove_cand_vpids_batch=None,vp_id_encountered_history_batch=None,):
        batch_size = len(obs)

        # Prepare containers for batched outputs
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []

            # Separate visited and unvisited nodes based on graph state
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:  # Optional override (default False)
                    if k == obs[i]["viewpoint"]:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        if k not in remove_cand_vpids_batch[i]:  # optional pruning
                            unvisited_vpids.append(k)

            # Optional early pruning of encountered vpids if threshold is exceeded
            if len(vp_id_encountered_history_batch) and self.args.gmap_early_pruning:
                assert self.args.gmap_pruning_t_threshold

                vp_id_encountered_history_i = vp_id_encountered_history_batch[i]
                head_count_to_remove = max(
                    len(unvisited_vpids) - self.args.gmap_pruning_t_threshold, 0
                )

                vp_id_to_remove = []
                for vp_id in vp_id_encountered_history_i.keys():
                    if len(vp_id_to_remove) >= head_count_to_remove:
                        break
                    if vp_id in unvisited_vpids:
                        vp_id_to_remove.append(vp_id)

                visited_vpids = [
                    vp_id for vp_id in visited_vpids if vp_id not in vp_id_to_remove
                ]
                unvisited_vpids = [
                    vp_id for vp_id in unvisited_vpids if vp_id not in vp_id_to_remove
                ]

            # If no new VP left to explore, signal this
            batch_no_vp_left.append(len(unvisited_vpids) == 0)

            if self.args.enc_full_graph:  # True by default
                # Node list for input: [STOP], [MEMORY], visited, unvisited
                gmap_vpids = [None] + [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                # Only include unvisited nodes + [STOP]
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            # Each node’s step ID from the graph
            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]

            # Get image embeddings for each viewpoint (skip [STOP] and [MEM])
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[2:]]

            # Prepend zeros and optionally the last memory embedding
            cat_rec_embeds = [torch.zeros_like(gmap_img_embeds[0])] if last_embeds is None else [last_embeds[i]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + cat_rec_embeds + gmap_img_embeds, 0
            )

            # Positional features relative to current viewpoint
            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]["viewpoint"],
                gmap_vpids,
                obs[i]["heading"],
                obs[i]["elevation"],
            )

            # Precompute pairwise distances between graph nodes
            gmap_pair_dists = torch.zeros(
                (len(gmap_vpids), len(gmap_vpids)), dtype=torch.float32, device="cuda"
            )
            for i in range(2, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = gmap.graph.distance(
                        gmap_vpids[i], gmap_vpids[j]
                    )

            # Append to batch
            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(gmap_pos_fts)
            batch_gmap_pair_dists.append(gmap_pair_dists)
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # Collate and pad batched tensors
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).to(self.device)
        batch_gmap_masks[:, 1] = False  # Mask out the [MEM] token
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).to(self.device)
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).to(self.device)
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).to(self.device)

        # Build full pairwise distance matrix
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, : batch_gmap_lens[i], : batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.to(self.device)

        return {
            "gmap_vpids": batch_gmap_vpids,
            "gmap_img_embeds": batch_gmap_img_embeds,
            "gmap_step_ids": batch_gmap_step_ids,
            "gmap_pos_fts": batch_gmap_pos_fts,
            "gmap_visited_masks": batch_gmap_visited_masks,
            "gmap_pair_dists": gmap_pair_dists,
            "gmap_masks": batch_gmap_masks,
            "no_vp_left": batch_no_vp_left,
            "t": None,
        }

    def _nav_vp_variable_mem(self, obs, gmaps, pano_embeds, cand_vpids, vp_img_ids, view_lens, nav_types, last_embeds=None,):
        batch_size = len(obs)

        # If no previous memory, use zero tensor as the [MEMORY] token. Otherwise, use the last embedding.
        cat_stop_embeds = (
            torch.zeros_like(pano_embeds[:, :1])
            if last_embeds is None
            else last_embeds.unsqueeze(1)
        )

        # Combine [STOP], [MEMORY] (or zeros), and the current pano embeddings
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), cat_stop_embeds, pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):  # For each environment in the batch
            # If candidate VPs exist, get positional features relative to current VP
            if len(cand_vpids[i]):
                cur_cand_pos_fts = gmap.get_pos_fts(
                    obs[i]["viewpoint"],
                    cand_vpids[i],
                    obs[i]["heading"],
                    obs[i]["elevation"],
                )

            # Get positional features to the start viewpoint (used for [STOP] and [MEMORY] tokens)
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]["viewpoint"],
                [gmap.start_vp],
                obs[i]["heading"],
                obs[i]["elevation"],
            )

            # Create a tensor for positional features for all tokens ([STOP], [MEMORY], candidates)
            vp_pos_fts = torch.zeros(
                (vp_img_embeds.size(1), 14), dtype=torch.float32, device="cuda"
            )  # shape: (num_tokens, 14)

            # Fill first 7-dim with positional features to the start viewpoint
            vp_pos_fts[:, :7] = cur_start_pos_fts

            # If candidate viewpoints exist, set the remaining 7-dim per candidate
            if len(cand_vpids[i]):
                vp_pos_fts[2 : len(cur_cand_pos_fts) + 2, 7:] = cur_cand_pos_fts

            batch_vp_pos_fts.append(vp_pos_fts)

        # Pad variable-length position features to a fixed size batch tensor
        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).to(self.device)

        # Mask out non-navigable views: [STOP]=1, [MEM]=0, candidate views (nav_type==1)
        vp_nav_masks = torch.cat(
            [
                torch.ones(batch_size, 1).bool().to(self.device),    # [STOP]
                torch.zeros(batch_size, 1).bool().to(self.device),   # [MEM]
                nav_types.to(self.device) == 1,                      # navigable views
            ],
            1,
        )

        # Generate regular padding masks based on sequence lengths (+2 for STOP + MEMORY)
        vp_masks = gen_seq_masks(view_lens + 2).to(self.device)

        # Add None placeholders for STOP and MEMORY tokens
        vp_cand_vpids = [[None] + [None] + x for x in cand_vpids]

        return {
            "vp_img_embeds": vp_img_embeds,           # visual features
            "vp_pos_fts": batch_vp_pos_fts,           # position features
            "vp_masks": vp_masks,                     # valid token masks
            "vp_nav_masks": vp_nav_masks,             # navigable masks
            "vp_cand_vpids": vp_cand_vpids,           # corresponding viewpoint ids
        }



    def make_equiv_action(self, a_t, gmaps, obs, traj=None, jump_traj=None):
        """
        Interface between Panoramic view and Egocentric view.
        Converts an abstract (panoramic-level) action into low-level simulator calls.
        Updates both `traj` and environment simulator state accordingly.
        """
        for i, ob in enumerate(obs):
            action = a_t[i]

            if action is not None:  # Skip if action is None (i.e., STOP)
                # Append full shortest path from current to target viewpoint
                traj[i]["path"].append(gmaps[i].graph.path(ob["viewpoint"], action))

                # Optionally update jump trajectory with direct jump
                if jump_traj is not None:
                    jump_traj[i]["path"].append([action])

                # Determine previous viewpoint to set heading/elevation
                if len(traj[i]["path"][-1]) == 1:
                    # If just one node in last step, use step before last
                    prev_vp = traj[i]["path"][-2][-1]
                else:
                    # Otherwise use second-to-last node in current step
                    prev_vp = traj[i]["path"][-1][-2]

                # Look up which discrete view index leads to `action` from `prev_vp`
                viewidx = self.scanvp_cands["%s_%s" % (ob["scan"], prev_vp)][action]

                # Convert to heading and elevation angles (each view is 30° apart)
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)

                # Instruct simulator to teleport to viewpoint with desired orientation
                self.env.env.sims[i].newEpisode(
                    [ob["scan"]], [action], [heading], [elevation]
                )
    
    def _update_scanvp_cands(self, obs):
        """
        Cache mapping from (scan, viewpoint) to candidate viewpointId → view index.
        This allows fast lookup of the egocentric view angle to reach a target viewpoint.
        """
        for ob in obs:
            scan = ob["scan"]
            vp = ob["viewpoint"]
            scanvp = "%s_%s" % (scan, vp)

            # Initialize dict for this scan+vp if not already present
            self.scanvp_cands.setdefault(scanvp, {})

            for cand in ob["candidate"]:
                # Set or update the mapping: viewpointId → pointId (i.e., discrete view index)
                self.scanvp_cands[scanvp].setdefault(cand["viewpointId"], {})
                self.scanvp_cands[scanvp][cand["viewpointId"]] = cand["pointId"]

    def _load_intervention(self, obs, z_dicts, z_front_dict):
        # Do intervention
        img_zdict, instr_zdict = self._do_intervention(z_dicts)
        # Frontdoor intervention
        front_txt_feats, front_vp_feats, front_gmap_feats = (
            self._frontdoor_intervention(obs, z_front_dict)
        )
        return img_zdict, instr_zdict, front_txt_feats, front_vp_feats, front_gmap_feats

    def _do_intervention(self, z_dicts):
        """Do intervention"""
        img_zdict, instr_zdict = None, None
        if "img_zdict" in z_dicts:
            img_zdict = z_dicts["img_zdict"]
        if "instr_zdict" in z_dicts:
            instr_zdict = z_dicts["instr_zdict"]
        return img_zdict, instr_zdict

    def _frontdoor_intervention(self, obs, z_front_dict):
        """Frontdoor intervention"""
        front_txt_feats, front_vp_feats, front_gmap_feats = None, None, None
        if z_front_dict is not None:
            if self.args.do_front_txt and "txt_feats" in z_front_dict:
                front_txt_feats = torch.from_numpy(
                    np.stack([z_front_dict["txt_feats"]] * len(obs))
                ).to(self.device)
            if self.args.do_front_img and "vp_feats" in z_front_dict:
                front_vp_feats = torch.from_numpy(
                    np.stack([z_front_dict["vp_feats"]] * len(obs))
                ).to(self.device)
            if self.args.do_front_his and "gmap_feats" in z_front_dict:
                front_gmap_feats = torch.from_numpy(
                    np.stack([z_front_dict["gmap_feats"]] * len(obs))
                ).to(self.device)
        return front_txt_feats, front_vp_feats, front_gmap_feats

    def _build_obs_batch(self, reset):
        """Initializes gmaps and traj from the first observation batch."""

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        self._update_scanvp_cands(obs)

        # Build graph: keep the start viewpoint
        gmaps = [GraphMap(ob["viewpoint"]) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record initial navigation paths
        traj = [
            {
                "instr_id": ob["instr_id"],
                "path": [[ob["viewpoint"]]],
                "details": {},
            }
            for ob in obs
        ]

        return obs, gmaps, traj

    def _calculate_avg_pano_embeds(self, pano_embeds, pano_masks, pano_fused_embeds):
        """Computes average pano embeddings based on fusion strategy."""
        if self.args.adaptive_pano_fusion:
            return pano_fused_embeds
        else:
            # Prevent divide-by-zero by clamping denominator
            mask_sum = torch.clamp(torch.sum(pano_masks, dim=1, keepdim=True), min=1e-6)
            avg_embeds = (
                torch.sum(pano_embeds * pano_masks.unsqueeze(2), dim=1) / mask_sum
            )
            return avg_embeds

    def _select_action(self, logits):
        if self.feedback == "argmax":
            return logits.argmax(1).detach()
        else:
            raise ValueError(f"Unsupported feedback mode: {self.feedback}")

    # @profile
    def rollout(self, reset=True, test=False, z_dicts={}, z_front_dict={}):
    # Reset environment or get current observations
        obs, gmaps, traj = self._build_obs_batch(reset)
        batch_size = len(obs)

        # Load causal intervention data (if any)
        img_zdict, instr_zdict, front_txt_feats, front_vp_feats, front_gmap_feats = (
            self._load_intervention(obs, z_dicts, z_front_dict)
        )

        # Step 1: Language encoding (instruction)
        language_inputs = self._language_variable(obs, instr_zdict, front_txt_feats)
        with torch.no_grad():  # No gradients needed for inference
            txt_outs = self.vln_bert("language", language_inputs)
            if isinstance(txt_outs, tuple):
                txt_embeds = txt_outs[0]
                txt_masks = txt_outs[1]
            else:
                txt_embeds = txt_outs

        # Initialize state trackers
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)
        jump_traj = None
        last_embeds = None
        vp_id_encountered_history_batch = []
        remove_cand_vpids_batch = [[] for _ in range(batch_size)]  # Track pruned candidates

        # Navigation loop (time step t)
        for t in range(self.args.max_action_len):
            # Record timestep at current viewpoint in the graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]["viewpoint"]] = t + 1

            # Step 2: Visual encoding (panorama view)
            pano_inputs = self._panorama_feature_variable_do(obs, img_zdict)
            with torch.no_grad():
                (
                    pano_embeds,
                    pano_masks,
                    pano_fused_embeds,
                    merged_nav_types,
                    merged_vp_img_ids,
                    merged_view_lens,
                ) = self.vln_bert("panorama", pano_inputs)

            # Compute average pano features (or use fused ones)
            avg_pano_embeds = self._calculate_avg_pano_embeds(
                pano_embeds, pano_masks, pano_fused_embeds
            )

            # Identify valid and pruned candidate viewpoints
            new_cand_vpids_batch = []
            for i, cand_vpids in enumerate(pano_inputs["cand_vpids"]):
                new_cand_vpids, remove_cand_vpids = [], []
                for j, cand_vp in enumerate(cand_vpids):
                    if pano_inputs["vp_img_ids"][i][j] in merged_vp_img_ids[i]:
                        new_cand_vpids.append(cand_vp)
                    else:
                        remove_cand_vpids.append(cand_vp)
                new_cand_vpids_batch.append(new_cand_vpids)
                for remove_cand_vp in remove_cand_vpids:
                    if remove_cand_vp not in remove_cand_vpids_batch[i]:
                        remove_cand_vpids_batch[i].append(remove_cand_vp)

            # Update graph embeddings: visited + unvisited nodes
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.update_node_embed(obs[i]["viewpoint"], avg_pano_embeds[i], rewrite=True)
                    for j, i_cand_vp in enumerate(new_cand_vpids_batch[i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # Step 3: Graph encoding
            nav_inputs = self._nav_gmap_variable(
                obs,
                gmaps,
                last_embeds,
                remove_cand_vpids_batch=remove_cand_vpids_batch,
                vp_id_encountered_history_batch=vp_id_encountered_history_batch,
            )

            # Step 4: Candidate encoding (egocentric view)
            nav_inputs.update(
                self._nav_vp_variable_mem(
                    obs,
                    gmaps,
                    pano_embeds,
                    new_cand_vpids_batch,
                    merged_vp_img_ids,
                    merged_view_lens,
                    merged_nav_types,
                    last_embeds,
                )
            )

            # Step 5: Update history of encountered VP IDs (for pruning)
            if len(vp_id_encountered_history_batch) == 0:
                for i in range(batch_size):
                    vp_id_history_dict = {}
                    for j, vp_id in enumerate(nav_inputs["gmap_vpids"][i]):
                        if nav_inputs["gmap_masks"][i][j] and not nav_inputs["gmap_visited_masks"][i][j] and vp_id is not None:
                            vp_id_history_dict[vp_id] = 1
                    vp_id_encountered_history_batch.append(vp_id_history_dict)
            else:
                for i in range(batch_size):
                    if not ended[i]:
                        for k in vp_id_encountered_history_batch[i].keys():
                            vp_id_encountered_history_batch[i][k] += 1
                        for j, vp_id in enumerate(nav_inputs["vp_cand_vpids"][i]):
                            if nav_inputs["vp_nav_masks"][i][j] and vp_id is not None:
                                vp_id_encountered_history_batch[i][vp_id] = 1
                    vp_id_encountered_history_batch[i] = dict(
                        sorted(
                            vp_id_encountered_history_batch[i].items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    )

            # Step 6: Add instruction features and feedback mode
            nav_inputs.update({
                "txt_embeds": txt_embeds,
                "txt_masks": txt_masks,
                "front_txt_feats": front_txt_feats,
                "front_vp_feats": front_vp_feats,
                "front_gmap_feats": front_gmap_feats,
                "feedback": self.feedback,
                "vp_img_ids": pano_inputs["vp_img_ids"],
            })

            # Step 7: Navigation decision
            if test:
                with torch.no_grad():
                    nav_outs = self.vln_bert("navigation", nav_inputs)
            else:
                nav_outs = self.vln_bert("navigation", nav_inputs)
            
            # Logging stats
            self.token_stats['candidate_size'] += [max([len(nav_inputs["vp_cand_vpids"][i]) for i in range(batch_size)])-2]
            self.token_stats["gmap_size"] += [nav_inputs["gmap_masks"].shape[1]]
            self.token_stats["pano_img_size"] += [self.vln_bert.vln_bert.vp_img_embeds_length]
            self.token_stats["text_size"] += [txt_embeds.shape[1]]

            # Track class token embeddings for current step
            last_embeds = nav_outs["cls_embeds"]

            # Choose logits and corresponding vpids according to fusion strategy
            if self.args.fusion == "local":
                nav_logits = nav_outs["local_logits"]
                nav_vpids = nav_inputs["vp_cand_vpids"]
            elif self.args.fusion == "global":
                nav_logits = nav_outs["global_logits"]
                nav_vpids = nav_inputs["gmap_vpids"]
            else:
                nav_logits = nav_outs["fused_logits"]
                nav_vpids = nav_inputs["gmap_vpids"]
            

            # Update stop probabilities in graph nodes
            nav_probs = torch.softmax(nav_logits, 1)
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_stop_scores[obs[i]["viewpoint"]] = {
                        "stop": nav_probs[i, 0].data.item()
                    }

            # Step 8: Decide next action
            a_t = self._select_action(nav_logits)
            a_t_stop = a_t == 0  # Which agents want to stop

            # Translate actions to simulator commands
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs["no_vp_left"][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

            # Step 9: Apply action and update graph
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj, jump_traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {"stop": -float("inf")}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v["stop"] > stop_score["stop"]:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]["viewpoint"] != stop_node:
                        traj[i]["path"].append(gmaps[i].graph.path(obs[i]["viewpoint"], stop_node))
                    if jump_traj is not None:
                        jump_traj[i]["path"].append([stop_node])
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]["details"][k] = {"stop_prob": float(v["stop"])}

            # Step 10: Get new observations from environment
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            # Mark ended agents
            ended[:] = ended | np.array([x is None for x in cpu_a_t])

            # Exit early if all agents are done
            if ended.all():
                break

        self.finished_counter += 1

        # Step 11: Handle submission formatting for evaluation
        if self.args.submit:
            for i, item in enumerate(traj):
                new_paths = []
                for node in item["path"]:
                    for each_sub_node in node:
                        new_paths.append([each_sub_node])
                traj[i]["path"] = new_paths

        return traj

