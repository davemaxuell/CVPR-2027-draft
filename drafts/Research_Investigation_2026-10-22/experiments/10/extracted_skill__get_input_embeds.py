def _get_input_embeds(self, obs, language_inputs, pano_inputs, gmaps, ended, expert_indices, t):
    """            
        ### --- Solution 1
        if self.args.routing_mode == 'random':
            set_seed(self.args.seed + t)
            vln_bert_idx = int(random() * len(self.vln_berts))  # Ensure the index is within bounds
            self.vln_bert = self.vln_berts[vln_bert_idx]
                
        for i, gmap in enumerate(gmaps):
            if not ended[i]:
                gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
                
        txt_embeds = self.vln_bert('language', language_inputs)
                            
        # graph representation
        pano_inputs = self._panorama_feature_variable(obs)
        
        pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs) # pano_embeds: (bs, n_panos, dim)
        avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) /                           torch.sum(pano_masks, 1, keepdim=True)
        
        if self.args.debug:
            self.logger.info(f"Step {t}: txt_embeds shape: {txt_embeds.shape}, pano_embeds shape: {pano_embeds.shape}, avg_pano_embeds shape: {avg_pano_embeds.shape}")
            self.logger.info(f"Step {t}: pano_inputs['view_lens'] shape: {pano_inputs['view_lens'].shape}, pano_inputs['cand_vpids'] shape: {len(pano_inputs['cand_vpids'])}, pano_inputs['nav_types'] shape: {pano_inputs['nav_types'].shape}")
        ### --- End of Solution 1
        """
    batch_size = len(obs)
    if expert_indices is None:
        set_seed(self.args.seed + t)
        if len(self.vln_berts) > 1:
            expert_indices = [randint(0, len(self.vln_berts) - 1) for _ in range(batch_size)]
        else:
            expert_indices = [0] * batch_size
    elif len(expert_indices) < batch_size:
        expert_indices += [1] * (batch_size - len(expert_indices))
    elif len(expert_indices) > batch_size:
        expert_indices = expert_indices[:batch_size]
    txt_embeds_list, pano_embeds_list, avg_pano_embeds_list, pano_lengths = ([], [], [], [])
    self.logger.info(f"Step {t}: Using expert indices: {expert_indices} for obs: {[ob['instr_id'] for ob in obs]}")
    for i, (ob, expert_idx) in enumerate(zip(obs, expert_indices)):
        self.vln_bert = self.vln_berts[expert_idx]
        single_language_inputs = {k: v[i:i + 1] for k, v in language_inputs.items()}
        txt_embed = self.vln_bert('language', single_language_inputs)
        txt_embeds_list.append(txt_embed)
        if not ended[i]:
            gmaps[i].node_step_ids[obs[i]['viewpoint']] = t + 1
        single_pano_inputs = self._panorama_feature_variable([obs[i]])
        pano_embed, pano_mask = self.vln_bert('panorama', single_pano_inputs)
        avg_pano_embed = torch.sum(pano_embed * pano_mask.unsqueeze(2), 1) / torch.sum(pano_mask, 1, keepdim=True)
        pano_embeds_list.append(pano_embed.squeeze(0))
        try:
            avg_pano_embeds_list.append(avg_pano_embed)
        except Exception as e:
            self.logger.warning(f"Embed failed for idx {i}, instr_id={ob['instr_id']}: {e}")
            continue
        pano_lengths.append(pano_embed.shape[1])
    txt_embeds = torch.cat(txt_embeds_list, dim=0)
    pano_embeds = pad_tensors(pano_embeds_list)
    avg_pano_embeds = torch.cat(avg_pano_embeds_list, dim=0)
    return (txt_embeds, pano_embeds, avg_pano_embeds, gmaps)