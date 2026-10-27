
import numpy as np
import math

import torch
from torch.nn.utils.rnn import pad_sequence

from utils.ops import pad_tensors, gen_seq_masks
from models.ops import pad_tensors_wgrad
from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from .agent_base import Seq2SeqAgent


class GMapObjectNavAgent(Seq2SeqAgent):
    """REVERIE object navigation agent with graph memory and causal intervention support."""

    def _build_model(self):
        """Build VLN-BERT navigation model and critic network."""
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        self.scanvp_cands = {}  # Cache candidate view indexing per scan/viewpoint

    def _language_variable(self, obs, instr_dict=None):
        """
        Prepare batched language inputs for the instruction encoder.

        Args:
            obs (list of dict): each dict must contain 'instr_encoding' (list of token IDs)
            instr_dict (dict, optional): intervention dict containing
                'instr_landmark_features' ([N_feats]) and
                'instr_landmark_pzs' ([N_pzs]); if provided, these are repeated per batch.

        Returns:
            dict with keys:
            'txt_ids' (LongTensor [B, L]),
            'txt_masks' (BoolTensor [B, L]),
            'instr_z_landmark_features' (Tensor [B, M, 768] or None),
            'instr_z_landmark_pzs' (Tensor [B, M, 1] or None),
            'seq_lengths' (list of int),
            'irrelevant_words_id' (as stored on self).
        """
        # 1. Gather sequence lengths
        seq_lengths = [len(o['instr_encoding']) for o in obs]
        batch_size = len(obs)
        max_len = max(seq_lengths)

        # 2. Allocate numpy arrays for IDs and mask
        ids_np  = np.zeros((batch_size, max_len), dtype=np.int64)
        mask_np = np.zeros((batch_size, max_len), dtype=bool)
        for i, o in enumerate(obs):
            l = seq_lengths[i]
            ids_np[i, :l]  = o['instr_encoding']
            mask_np[i, :l] = True

        # 3. Convert to torch tensors on GPU
        txt_ids   = torch.from_numpy(ids_np).long().cuda()
        txt_masks = torch.from_numpy(mask_np).cuda()

        # 4. Repeat landmark features if provided
        if instr_dict:
            # [N_feats] → [B, M, 768]
            feats = instr_dict['instr_landmark_features']
            instr_z_landmark_features = feats.repeat(batch_size, 1).view(batch_size, -1, 768)
            # [N_pzs] → [B, M, 1]
            pzs = instr_dict['instr_landmark_pzs']
            instr_z_landmark_pzs = pzs.repeat(batch_size, 1).view(batch_size, -1, 1)
        else:
            instr_z_landmark_features = None
            instr_z_landmark_pzs      = None

        return {
            'txt_ids':                     txt_ids,
            'txt_masks':                   txt_masks,
            'instr_z_landmark_features':   instr_z_landmark_features,
            'instr_z_landmark_pzs':        instr_z_landmark_pzs,
            'seq_lengths':                 seq_lengths,
            'irrelevant_words_id':         self.irrelevant_words_id,
        }


    def _panorama_feature_variable_do(
        self,
        obs,
        img_zdict=None,
        use_objname=True
    ):
        """
        Convert panoramic visual + object features into batched tensors.

        Args:
            obs (list of dict): each dict contains:
                - 'candidate' (list of dicts with 'feature', 'viewpointId', 'pointId')
                - 'feature' (array of all-view features)
                - 'obj_ang_fts', 'obj_box_fts' (np arrays for objects)
                - 'obj_img_fts' (np array of object image features)
                - 'obj_name' (list of int IDs) if use_objname True
            img_zdict (dict, optional): with 'img_features' and 'img_pzs' to repeat per batch
            use_objname (bool): whether to collect and pad object-name IDs

        Returns:
            dict with keys:
            'view_img_fts'        (Tensor [B, V_max, D_img]),
            'loc_fts'             (Tensor [B, V_max+O_max, D_loc]),
            'nav_types'           (LongTensor [B, V_max+O_max]),
            'view_lens'           (LongTensor [B]),
            'cand_vpids'          (list of lists of candidate vpids),
            'vp_img_ids'          (LongTensor [B, V_max]),
            'z_img_features'      (Tensor [B, M, 768] or None),
            'z_img_pzs'           (Tensor [B, M, 1] or None),
            'reverie_obj_img_fts' (Tensor [B, O_max, D_img]),
            'reverie_obj_lens'    (LongTensor [B]),
            'reverie_obj_names'   (Tensor [B, O_max] or None).
        """
        batch_size = len(obs)

        # containers per batch
        view_img_list, obj_img_list = [], []
        loc_list, nav_types_list      = [], []
        cand_vpids, vp_img_ids        = [], []
        view_lens, obj_lens           = [], []
        obj_name_list = [] if use_objname else None

        for o in obs:
            # 1. Split candidate vs. non-candidate views
            view_feats, ang_feats = [], []
            vtypes, v_ids, img_ids = [], [], []
            used_img_idxs = set()

            # a) candidate viewpoints
            for cc in o['candidate']:
                f = cc['feature']
                view_feats.append(f[:self.args.image_feat_size])
                ang_feats.append(f[self.args.image_feat_size:])
                vtypes.append(1)
                v_ids.append(cc['viewpointId'])
                img_ids.append(cc['pointId'])
                used_img_idxs.add(cc['pointId'])

            # b) non-candidate views
            for idx_f, f in enumerate(o['feature']):
                if idx_f not in used_img_idxs:
                    view_feats.append(f[:self.args.image_feat_size])
                    ang_feats.append(f[self.args.image_feat_size:])
                    vtypes.append(0)
                    img_ids.append(idx_f)

            # 2. Build per-view loc features: angles + dummy box
            view_box = np.ones((len(view_feats), 3), dtype=np.float32)
            view_loc = np.concatenate([np.stack(ang_feats), view_box], axis=1)

            # 3. Object-level loc features
            obj_loc = np.concatenate([o['obj_ang_fts'], o['obj_box_fts']], axis=1)
            vtypes.extend([2] * obj_loc.shape[0])

            # 4. Optional object-name IDs
            if use_objname:
                obj_name_list.append(
                    torch.from_numpy(np.array(o['obj_name'], dtype=np.int64))
                )

            # 5. Append to batch lists
            view_img_list.append(torch.from_numpy(np.stack(view_feats)))
            obj_img_list.append(torch.from_numpy(o['obj_img_fts']))
            loc_list.append(torch.from_numpy(np.concatenate([view_loc, obj_loc], axis=0)))
            nav_types_list.append(torch.LongTensor(vtypes))
            cand_vpids.append(v_ids)
            view_lens.append(len(view_feats))
            obj_lens.append(o['obj_img_fts'].shape[0])
            vp_img_ids.append(torch.LongTensor(img_ids))

        # 6. Pad & move to GPU
        view_img_fts = pad_tensors(view_img_list).cuda()
        obj_img_fts  = pad_tensors(obj_img_list).cuda()
        loc_fts      = pad_tensors(loc_list).cuda()
        nav_types    = pad_sequence(nav_types_list, batch_first=True, padding_value=0).cuda()
        view_lens_t  = torch.LongTensor(view_lens).cuda()
        obj_lens_t   = torch.LongTensor(obj_lens).cuda()
        vp_img_ids_t = pad_sequence(vp_img_ids, batch_first=True, padding_value=-1).cuda()

        obj_names_padded = (
            pad_tensors(obj_name_list).cuda()
            if use_objname and obj_name_list else
            None
        )

        # 7. Causal image features
        if img_zdict:
            feats = img_zdict['img_features']
            pzs   = img_zdict['img_pzs']
            z_img_features = feats.repeat(batch_size, 1).view(batch_size, -1, 768)
            z_img_pzs      = pzs.repeat(batch_size, 1).view(batch_size, -1, 1)
        else:
            z_img_features = None
            z_img_pzs      = None

        return {
        'view_img_fts':        view_img_fts,
        'loc_fts':             loc_fts,
        'nav_types':           nav_types,
        'view_lens':           view_lens_t,
        'cand_vpids':          cand_vpids,
        'vp_img_ids':          vp_img_ids_t,
        'z_img_features':      z_img_features,
        'z_img_pzs':           z_img_pzs,
        'reverie_obj_img_fts': obj_img_fts,
        'reverie_obj_lens':    obj_lens_t,
        'reverie_obj_names':   obj_names_padded,
    }



    def _nav_gmap_variable(
            self,
            obs,
            gmaps,
            last_embeds=None,
            remove_cand_vpids_batch=None,
            vp_history_batch=None
        ):
        """
        Build global-graph inputs for navigation.

        Args:
            obs:                      list of observation dicts
            gmaps:                    list of GMap graphs
            last_embeds:              [batch, dim] tensor or None for memory
            remove_cand_vpids_batch:  list of lists of vpid to exclude
            vp_history_batch:         list of dicts {vpid: recency} for pruning

        Returns:
            dict with keys:
            'gmap_vpids', 'gmap_img_embeds', 'gmap_step_ids',
            'gmap_pos_fts', 'gmap_pair_dists', 'gmap_visited_masks',
            'gmap_masks', 'no_vp_left', 't'
        """
        batch_size = len(obs)
        # safe defaults
        if remove_cand_vpids_batch is None:
            remove_cand_vpids_batch = [[] for _ in range(batch_size)]
        if vp_history_batch is None:
            vp_history_batch = [{} for _ in range(batch_size)]

        all_vpids, all_img_embeds, all_step_ids = [], [], []
        all_pos_fts, all_pair_dists, all_visited_masks = [], [], []
        no_vp_left = []

        for idx in range(batch_size):
            ob = obs[idx]
            gmap = gmaps[idx]

            # split visited vs unvisited, filter out pruned vpids
            visited = [vp for vp in gmap.node_positions if gmap.graph.visited(vp)]
            unvisited = [
                vp for vp in gmap.node_positions
                if not gmap.graph.visited(vp)
                and vp not in remove_cand_vpids_batch[idx]
            ]

            # early pruning by encounter history
            if self.args.gmap_early_pruning and vp_history_batch[idx]:
                threshold = self.args.gmap_pruning_t_threshold
                excess = max(len(unvisited) - threshold, 0)
                to_remove = [
                    vp for vp in vp_history_batch[idx]
                    if vp in unvisited
                ][:excess]
                if to_remove:
                    visited = [vp for vp in visited if vp not in to_remove]
                    unvisited = [vp for vp in unvisited if vp not in to_remove]

            no_vp_left.append(len(unvisited) == 0)

            # build vpid list + visited mask
            if self.args.enc_full_graph:
                vpids = [None, None] + visited + unvisited
                visited_mask = [0, 1] + [1]*len(visited) + [0]*len(unvisited)
            else:
                vpids = [None] + unvisited
                visited_mask = [0]*len(vpids)

            # step IDs for each vpid
            step_ids = [gmap.node_step_ids.get(vp, 0) for vp in vpids]

            # image embeds: [NULL] + [MEM] + actual nodes
            raw_embeds = [gmap.get_node_embed(vp) for vp in vpids[2:]]
            if last_embeds is None:
                mem = raw_embeds[0].new_zeros(raw_embeds[0].shape)
            else:
                mem = last_embeds[idx]
            img_embeds = torch.stack(
                [raw_embeds[0].new_zeros(raw_embeds[0].shape), mem] + raw_embeds,
                dim=0
            )

            # positional features
            pos_fts = gmap.get_pos_fts(
                ob['viewpoint'], vpids,
                ob['heading'], ob['elevation']
            )

            # pairwise distances
            n = len(vpids)
            pdist = np.zeros((n, n), dtype=np.float32)
            for i in range(2, n):
                for j in range(i+1, n):
                    d = gmap.graph.distance(vpids[i], vpids[j])
                    pdist[i, j] = pdist[j, i] = d

            # collect
            all_vpids.append(vpids)
            all_img_embeds.append(img_embeds)
            all_step_ids.append(torch.LongTensor(step_ids))
            all_pos_fts.append(pos_fts)
            all_pair_dists.append(torch.from_numpy(pdist))
            all_visited_masks.append(torch.BoolTensor(visited_mask))

        # collate to batch
        lengths = torch.LongTensor([len(v) for v in all_vpids])
        gmap_masks     = gen_seq_masks(lengths).cuda()
        img_embeds_b   = pad_tensors_wgrad(all_img_embeds)
        step_ids_b     = pad_sequence(all_step_ids, batch_first=True).cuda()
        pos_fts_b      = pad_tensors(all_pos_fts).cuda()
        visited_m_b    = pad_sequence(all_visited_masks, batch_first=True).cuda()

        # batch pairwise dists
        max_len = lengths.max().item()
        pair_dists = img_embeds_b.new_zeros((batch_size, max_len, max_len))
        for i, pd in enumerate(all_pair_dists):
            L = pd.shape[0]
            pair_dists[i, :L, :L] = pd
        pair_dists = pair_dists.cuda()

        return {
            'gmap_vpids':       all_vpids,
            'gmap_img_embeds':  img_embeds_b,
            'gmap_step_ids':    step_ids_b,
            'gmap_pos_fts':     pos_fts_b,
            'gmap_pair_dists':  pair_dists,
            'gmap_visited_masks': visited_m_b,
            'gmap_masks':       gmap_masks,
            'no_vp_left':       no_vp_left,
            't':                None,
        }


    def _nav_vp_variable_do(
        self,
        obs,
        gmaps,
        pano_embeds,
        cand_vpids,
        vp_img_ids,
        view_lens,
        nav_types,
        last_embeds=None
    ):
        """
        Build per-viewpoint (local) navigation inputs.

        Args:
        obs:           list of obs dicts
        gmaps:         list of GMap graphs
        pano_embeds:   [batch, num_views, dim]
        cand_vpids:    list of candidate vp lists
        vp_img_ids:    corresponding image IDs
        view_lens:     tensor [batch] of candidate counts
        nav_types:     tensor [batch, total_tokens] (1=nav, 2=obj)
        last_embeds:   [batch, dim] tensor or None

        Returns:
        dict with:
            'vp_img_embeds', 'vp_pos_fts', 'vp_masks',
            'vp_nav_masks', 'vp_obj_masks',
            'vp_cand_vpids', 'vp_img_ids'
        """
        batch_size = len(obs)

        # [STOP] + [MEM] + pano embeddings
        stop_emb = pano_embeds[:, :1].new_zeros(pano_embeds[:, :1].shape)
        if last_embeds is None:
            mem_emb = stop_emb.clone()
        else:
            mem_emb = last_embeds.unsqueeze(1)
        vp_img_embeds = torch.cat([stop_emb, mem_emb, pano_embeds], dim=1)

        # positional features
        all_pos = []
        for idx in range(batch_size):
            ob = obs[idx]
            gmap = gmaps[idx]

            cand_pos = (gmap.get_pos_fts(ob['viewpoint'], cand_vpids[idx],
                                        ob['heading'], ob['elevation'])
                        if cand_vpids[idx] else [])
            start_pos = gmap.get_pos_fts(ob['viewpoint'], [gmap.start_vp],
                                        ob['heading'], ob['elevation'])
            T = vp_img_embeds.size(1)
            D = start_pos.size(1)
            pos = start_pos.new_zeros((T, D*2))
            pos[:, :D] = start_pos
            pos[1, :D] = start_pos  # MEM token

            if len(cand_pos) > 0:
                pos[2:2+len(cand_pos), D:] = cand_pos
            all_pos.append(pos)
        vp_pos_fts = pad_tensors(all_pos).cuda()

        # masks
        vp_masks     = gen_seq_masks(view_lens + 2).cuda()
        vp_nav_masks = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool).cuda(),
            torch.zeros(batch_size, 1, dtype=torch.bool).cuda(),
            (nav_types == 1)
        ], dim=1)
        vp_obj_masks = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool).cuda(),
            torch.zeros(batch_size, 1, dtype=torch.bool).cuda(),
            (nav_types == 2)
        ], dim=1)

        # candidate vpids with placeholders
        vp_cand_vpids = [[None, None] + lst for lst in cand_vpids]

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts':    vp_pos_fts,
            'vp_masks':      vp_masks,
            'vp_nav_masks':  vp_nav_masks,
            'vp_obj_masks':  vp_obj_masks,
            'vp_cand_vpids': vp_cand_vpids,
            'vp_img_ids':    vp_img_ids,
        }


    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Convert panoramic-space actions into egocentric simulator calls.

        Args:
        a_t:    list of selected vpids or None for stop
        gmaps:  list of GMap graphs
        obs:    list of obs dicts
        traj:   list of trajectory dicts to append to
        """
        for idx, action in enumerate(a_t):
            if action is None:
                continue
            cur_vp = obs[idx]['viewpoint']
            path = gmaps[idx].graph.path(cur_vp, action)
            traj[idx]['path'].append(path)

            # pick previous vp for view index lookup
            if len(path) == 1:
                prev_vp = traj[idx]['path'][-2][-1]
            else:
                prev_vp = path[-2]

            key = f"{obs[idx]['scan']}_{prev_vp}"
            view_idx = self.scanvp_cands[key][action]
            heading   = (view_idx % 12) * math.radians(30)
            elevation = (view_idx // 12 - 1) * math.radians(30)

            self.env.env.sims[idx].newEpisode(
                [obs[idx]['scan']], [action], [heading], [elevation]
            )


    def _update_scanvp_cands(self, obs):
        """
        Populate self.scanvp_cands with mapping scan_vp → {vp_id: pointId}.
        """
        for ob in obs:
            key = f"{ob['scan']}_{ob['viewpoint']}"
            self.scanvp_cands.setdefault(key, {})
            for cand in ob['candidate']:
                self.scanvp_cands[key][cand['viewpointId']] = cand['pointId']


    
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
        
    def rollout(self, reset: bool = True,
            z_dicts: dict = None,
            z_front_dict: dict = None) -> list:
        """
        Execute a batch rollout of the navigation policy and return trajectories.

        Args:
            reset (bool): Whether to reset the environment before rollout.
            z_dicts (dict): Intervention dicts for images and instructions.
            z_front_dict (dict): Frontdoor intervention features.

        Returns:
            traj (list): A list of trajectory dicts for each batch element.
        """
        # Safe defaults for mutable args
        z_dicts = z_dicts or {}
        z_front_dict = z_front_dict or {}

        # 1. Build initial batch of observations, graphs, and empty trajectories
        obs, gmaps, traj = self._build_obs_batch(reset)
        batch_size = len(obs)

        # 2. Load any causal‐intervention data
        img_zdict, instr_zdict, front_txt_feats, front_vp_feats, front_gmap_feats = (
            self._load_intervention(obs, z_dicts, z_front_dict)
        )

        # 3. Prepare language embeddings
        lang_inputs = self._language_variable(obs, instr_zdict)
        with torch.no_grad():
            txt_outs = self.vln_bert('language', lang_inputs)
        

        # Unpack BERT outputs
        if isinstance(txt_outs, tuple):
            txt_embeds, txt_masks = txt_outs[0], txt_outs[1]
        else:
            txt_embeds, txt_masks = txt_outs, None

        # 4. Initialize rollout state
        ended = np.zeros(batch_size, dtype=bool)
        just_ended = np.zeros(batch_size, dtype=bool)
        last_embeds = None
        vp_history = [{} for _ in range(batch_size)]
        remove_cand_vpids_batch = [[] for _ in range(batch_size)]

        # 5. Main time‐step loop
        for t in range(self.args.max_action_len):
            # 5.1 Mark current timestamp on each graph node
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    current_vp = obs[i]['viewpoint']
                    gmap.node_step_ids[current_vp] = t + 1

            # 5.2 Extract panorama features
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

            # 5.3 Compute average/fused pano embeddings
            avg_pano_embeds = self._calculate_avg_pano_embeds(
                pano_embeds, pano_masks, pano_fused_embeds
            )

            # 5.4 Filter candidates based on which views survived fusion
            new_cand_vpids_batch = []
            remove_cand_vpids_batch = [[] for _ in range(batch_size)]
            for i, cand_vpids in enumerate(pano_inputs['cand_vpids']):
                keep, remove = [], []
                for vp, img_id in zip(cand_vpids, pano_inputs['vp_img_ids'][i]):
                    if img_id in merged_vp_img_ids[i]:
                        keep.append(vp)
                    else:
                        remove.append(vp)
                new_cand_vpids_batch.append(keep)
                # Avoid duplicates
                remove_cand_vpids_batch[i] = list(dict.fromkeys(remove))

            # 5.5 Update graph embeddings for visited & unvisited nodes
            for i, gmap in enumerate(gmaps):
                if ended[i]:
                    continue
                cur_vp = obs[i]['viewpoint']
                gmap.update_node_embed(cur_vp, avg_pano_embeds[i], rewrite=True)
                for j, vp in enumerate(new_cand_vpids_batch[i]):
                    if not gmap.graph.visited(vp):
                        gmap.update_node_embed(vp, pano_embeds[i, j])

            # 5.6 Prepare navigation inputs
            nav_inputs = self._nav_gmap_variable(
                obs, gmaps, last_embeds, remove_cand_vpids_batch,
                vp_history_batch=vp_history
            )
            nav_inputs.update(self._nav_vp_variable_do(
                obs, gmaps, pano_embeds, new_cand_vpids_batch,
                merged_vp_img_ids, merged_view_lens,
                merged_nav_types, last_embeds
            ))

            # 5.7 Initialize or update viewpoint history
            if all(not h for h in vp_history):
                # First time: record all new reachable vps
                for i in range(batch_size):
                    h = {}
                    for vp, visited_mask, mask in zip(
                        nav_inputs['gmap_vpids'][i],
                        nav_inputs['gmap_visited_masks'][i],
                        nav_inputs['gmap_masks'][i]
                    ):
                        if mask and not visited_mask and vp is not None:
                            h[vp] = 1
                    vp_history[i] = h
            else:
                # Subsequent steps: increment counts & add new visits
                for i in range(batch_size):
                    if ended[i]:
                        continue
                    # increment all existing
                    for vp in vp_history[i]:
                        vp_history[i][vp] += 1
                    # add newly navigable
                    for vp, nav_mask in zip(
                        nav_inputs['vp_cand_vpids'][i],
                        nav_inputs['vp_nav_masks'][i]
                    ):
                        if nav_mask and vp is not None:
                            vp_history[i][vp] = 1
                    # sort by recency
                    vp_history[i] = dict(
                        sorted(vp_history[i].items(),
                            key=lambda item: item[1],
                            reverse=True)
                    )

            # 5.8 Inject text & frontdoor features
            nav_inputs['txt_embeds'] = txt_embeds
            nav_inputs['txt_masks'] = txt_masks
            nav_inputs['front_txt_feats'] = front_txt_feats
            nav_inputs['front_vp_feats'] = front_vp_feats
            nav_inputs['front_gmap_feats'] = front_gmap_feats

            # 5.9 Forward pass for navigation
            nav_outs = self.vln_bert('navigation', nav_inputs)

            # 5.10 Log token stats
            self.token_stats['obj_size'] += [torch.max((pano_inputs['reverie_obj_lens'])).item()]
            self.token_stats['candidate_size'] += [max([len(nav_inputs["vp_cand_vpids"][i]) for i in range(batch_size)])-2]
            self.token_stats["gmap_size"].append(nav_inputs["gmap_masks"].shape[1])
            self.token_stats["pano_img_size"].append(
                self.vln_bert.vln_bert.vp_img_embeds_length
            )
            self.token_stats["text_size"].append(txt_embeds.shape[1])

            last_embeds = nav_outs['cls_embeds']
            
            # 5.11 Choose logits & candidate lists based on fusion strategy
            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, dim=1)
            obj_logits = nav_outs['obj_logits']

            # 5.12 Update stop/object scores in the graph
            for i, gmap in enumerate(gmaps):
                if ended[i]:
                    continue
                cur_vp = obs[i]['viewpoint']
                obj_ids = obs[i]['obj_ids']
                nav_types_i = merged_nav_types[i].tolist()
                # find object‐idx offset
                obj_offset = (nav_types_i.index(2) + 2) if 2 in nav_types_i else None
                logits_slice = (
                    obj_logits[i, obj_offset:]
                    if obj_offset is not None
                    else obj_logits[i, merged_view_lens[i] + 2:]
                )

                stop_score = nav_probs[i, 0].item()
                gmap.node_stop_scores[cur_vp] = {
                    'stop': stop_score,
                    'og': obj_ids[torch.argmax(logits_slice)] if len(obj_ids) > 0 else None,
                    'og_details': {
                        'objids': obj_ids,
                        'logits': logits_slice[: len(obj_ids)]
                    }
                }

            # 5.13 Select and apply action
            a_t = self._select_action(nav_logits)
            cpu_actions = []
            for i in range(batch_size):
                stop = (a_t[i] == 0) or ended[i] or nav_inputs['no_vp_left'][i] \
                    or (t == self.args.max_action_len - 1)
                cpu_actions.append(None if stop else nav_vpids[i][a_t[i]])
                just_ended[i] = stop and not ended[i]

            self.make_equiv_action(cpu_actions, gmaps, obs, traj)

            # 5.14 Record final stops & object picks for newly ended episodes
            for i in range(batch_size):
                if just_ended[i]:
                    best_stop, best_score = None, -float('inf')
                    for vp, data in gmaps[i].node_stop_scores.items():
                        if data['stop'] > best_score:
                            best_score, best_stop = data['stop'], vp
                    if best_stop is not None and obs[i]['viewpoint'] != best_stop:
                        traj[i]['path'].append(gmaps[i].graph.path(
                            obs[i]['viewpoint'], best_stop
                        ))
                    traj[i]['pred_objid'] = gmaps[i].node_stop_scores[best_stop]['og']
                    if self.args.detailed_output:
                        traj[i]['details'] = {
                            vp: {
                                'stop_prob': float(v['stop']),
                                'obj_ids': [str(x) for x in v['og_details']['objids']],
                                'obj_logits': v['og_details']['logits'].tolist()
                            }
                            for vp, v in gmaps[i].node_stop_scores.items()
                        }

            # 5.15 Step the environment
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            # 5.16 Check for termination
            ended |= np.array([act is None for act in cpu_actions])
            if ended.all():
                break

        # 6. Finalize trajectories & counter
        self.finished_counter += 1
        if self.args.submit:
            # Flatten each path into single‐node steps
            for item in traj:
                item['path'] = [[n] for sub in item['path'] for n in sub]

        return traj
