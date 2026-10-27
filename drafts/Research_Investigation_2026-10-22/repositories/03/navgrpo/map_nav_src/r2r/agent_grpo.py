import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad


class GMapNavAgent(Seq2SeqAgent):

    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        instructions = [ob['instruction'] for ob in obs]
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask, 'instr': instructions
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )  # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts, batch_view_img = [], []
        for i, gmap in enumerate(gmaps):
            view_img = []
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            for j, cc in enumerate(obs[i]['candidate']):
                view_img.append(cc['image'])
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))
            batch_view_img.append(view_img)

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_img': batch_view_img,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens + 1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0  # Stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                   + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(
            self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0  # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0  # Stop if arrived
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan],
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][
                                                                   1:],
                                        ob['gt_path'],
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                           + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:  # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
                'instr': language_inputs['instr']
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }

            if train_ml is not None:
                # Supervised training
                if self.args.dataset == 'r2r':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                elif self.args.dataset == 'r4r':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())

            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs[
                        'gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample':  # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

                    # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())
        return traj

    def rollout_grpo_sample(self, num_samples=4, reset=True):
        """
        正确的GRPO采样：只存储状态重建信息和旧策略的log_prob（作为常数）
        """
        self.num_samples = num_samples
        sampled_data = []
        _ = self.env.reset()

        with torch.no_grad():
            for k in range(num_samples):
                obs = self.env.reset_without_change()
                self._update_scanvp_cands(obs)
                batch_size = len(obs)

                gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
                for i, ob in enumerate(obs):
                    gmaps[i].update_graph(ob)

                sample_k = [{
                    'instr_id': ob['instr_id'],
                    'path': [[ob['viewpoint']]],
                    'transitions': [],
                    'scan': ob['scan'],
                    'gt_path': ob['gt_path'],
                    'instr_encoding': ob['instr_encoding'],
                    'instruction': ob['instruction'],
                } for ob in obs]

                # Language encoding
                language_inputs = self._language_variable(obs)
                txt_embeds = self.vln_bert('language', language_inputs)

                # 保存text相关（每个样本只需保存一次）
                for i in range(batch_size):
                    sample_k[i]['txt_embed'] = txt_embeds[i].cpu().numpy()
                    sample_k[i]['txt_mask'] = language_inputs['txt_masks'][i].cpu().numpy()

                ended = np.array([False] * batch_size)

                for t in range(self.args.max_action_len):
                    for i, gmap in enumerate(gmaps):
                        if not ended[i]:
                            gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

                    # 全景编码
                    pano_inputs = self._panorama_feature_variable(obs)
                    pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
                    avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                      torch.sum(pano_masks, 1, keepdim=True)

                    # 更新图
                    for i, gmap in enumerate(gmaps):
                        if not ended[i]:
                            i_vp = obs[i]['viewpoint']
                            gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                            for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                                if not gmap.graph.visited(i_cand_vp):
                                    gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

                    # 导航输入
                    nav_inputs = self._nav_gmap_variable(obs, gmaps)
                    nav_inputs.update(
                        self._nav_vp_variable(
                            obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                            pano_inputs['view_lens'], pano_inputs['nav_types'],
                        )
                    )
                    nav_inputs.update({
                        'txt_embeds': txt_embeds,
                        'txt_masks': language_inputs['txt_masks'],
                    })

                    # 前向计算
                    nav_outs = self.vln_bert('navigation', nav_inputs)

                    if self.args.fusion == 'local':
                        nav_logits = nav_outs['local_logits']
                        nav_vpids = nav_inputs['vp_cand_vpids']
                    elif self.args.fusion == 'global':
                        nav_logits = nav_outs['global_logits']
                        nav_vpids = nav_inputs['gmap_vpids']
                    else:
                        nav_logits = nav_outs['fused_logits']
                        nav_vpids = nav_inputs['gmap_vpids']

                    nav_probs = torch.softmax(nav_logits, 1)
                    nav_log_probs = torch.log_softmax(nav_logits, 1)

                    # 采样
                    dist = torch.distributions.Categorical(nav_probs)
                    a_t = dist.sample()

                    # ==== 完整保存所有navigation参数 ====
                    for i in range(batch_size):
                        if not ended[i]:
                            transition = {
                                # 基本信息
                                'action': a_t[i].item(),
                                'old_log_prob': nav_log_probs[i, a_t[i]].item(),

                                # ========== Graph Map 相关 ==========
                                'gmap_vpids': nav_inputs['gmap_vpids'][i],
                                'gmap_img_embeds': nav_inputs['gmap_img_embeds'][i].cpu().numpy(),
                                'gmap_step_ids': nav_inputs['gmap_step_ids'][i].cpu().numpy(),
                                'gmap_pos_fts': nav_inputs['gmap_pos_fts'][i].cpu().numpy(),
                                'gmap_masks': nav_inputs['gmap_masks'][i].cpu().numpy(),
                                'gmap_pair_dists': nav_inputs['gmap_pair_dists'][i].cpu().numpy(),
                                'gmap_visited_masks': nav_inputs['gmap_visited_masks'][i].cpu().numpy(),

                                # ========== ViewPoint 相关 ==========
                                'vp_img_embeds': nav_inputs['vp_img_embeds'][i].cpu().numpy(),
                                'vp_pos_fts': nav_inputs['vp_pos_fts'][i].cpu().numpy(),
                                'vp_masks': nav_inputs['vp_masks'][i].cpu().numpy(),
                                'vp_nav_masks': nav_inputs['vp_nav_masks'][i].cpu().numpy(),
                                'vp_cand_vpids': nav_inputs['vp_cand_vpids'][i],

                                # ========== Object相关（如果有）==========
                                'vp_obj_masks': nav_inputs.get('vp_obj_masks',
                                                               torch.zeros_like(nav_inputs['vp_masks']))[
                                    i].cpu().numpy(),
                            }

                            sample_k[i]['transitions'].append(transition)

                    # 环境交互
                    cpu_a_t = []
                    for i in range(batch_size):
                        if (a_t[i] == 0) or ended[i] or nav_inputs['no_vp_left'][i] or \
                                (t == self.args.max_action_len - 1):
                            cpu_a_t.append(None)
                        else:
                            cpu_a_t.append(nav_vpids[i][a_t[i]])

                    self.make_equiv_action(cpu_a_t, gmaps, obs, sample_k)
                    obs = self.env._get_obs()
                    self._update_scanvp_cands(obs)

                    for i, ob in enumerate(obs):
                        if not ended[i]:
                            gmaps[i].update_graph(ob)

                    ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

                    if ended.all():
                        break

                # 最终状态
                for i in range(batch_size):
                    sample_k[i]['final_viewpoint'] = obs[i]['viewpoint']

                # 计算奖励
                rewards_k = self._compute_rewards_batch(sample_k)
                for i in range(batch_size):
                    sample_k[i]['reward'] = rewards_k[i]

                sampled_data.append(sample_k)

        return sampled_data

    def _compute_rewards_batch(self, trajectories):
        batch_size = len(trajectories)
        rewards = np.zeros(batch_size, dtype=np.float32)

        for i, traj in enumerate(trajectories):
            scan = traj['scan']
            final_vp = traj['final_viewpoint']
            gt_path = traj['gt_path']
            pred_path = traj['path']  # List[List[vp]]

            # 提取实际走过的viewpoint序列
            path = []
            for step_vps in pred_path:
                if len(step_vps) > 0:
                    path.append(step_vps[0])

            if len(path) == 0:
                rewards[i] = 0.0
                continue

            # 使用环境的shortest_distances（已经预计算好了）
            distances = self.env.shortest_distances[scan]

            # Ground truth终点
            gt_end = gt_path[-1]

            # ========== 1. Success ==========
            nav_error = distances[final_vp][gt_end]
            success = 1.0 if nav_error < 3.0 else 0.0

            # ========== 2. SPL ==========
            if success:
                # GT路径长度
                gt_length = sum([
                    distances[gt_path[j]][gt_path[j + 1]]
                    for j in range(len(gt_path) - 1)
                ])

                # 实际轨迹长度
                traj_length = sum([
                    distances[path[j]][path[j + 1]]
                    for j in range(len(path) - 1)
                ])

                spl = gt_length / max(traj_length, gt_length)
            else:
                spl = 0.0

            # ========== 3. Progress ==========
            start_vp = gt_path[0]
            start_dist = distances[start_vp][gt_end]

            if start_dist > 0.01:
                progress = max(0.0, (start_dist - nav_error) / start_dist)
            else:
                progress = 1.0 if success else 0.0

            # ========== 选择奖励类型 ==========
            if self.args.reward_type == 'success':
                reward = success

            elif self.args.reward_type == 'spl':
                reward = spl

            elif self.args.reward_type == 'progress':
                reward = progress

            elif self.args.reward_type == 'mixed':
                # 成功用SPL，失败用progress的折扣
                reward = spl if success else progress * 0.3

            elif self.args.reward_type == 'dense':
                # 组合奖励
                reward = success * 2.0 + progress * 0.5 - (len(path) / 100.0)

            else:
                # 默认SPL
                reward = spl

            rewards[i] = reward

        return rewards

    def train_grpo(self, sampled_data, grpo_epochs=1):
        """
        正确的GRPO训练：重新前向计算，获得带梯度的新logits
        """
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        num_samples = len(sampled_data)
        batch_size = len(sampled_data[0])

        # 展开所有transitions
        all_transitions = []
        for k in range(num_samples):
            for i in range(batch_size):
                traj = sampled_data[k][i]
                advantage = traj['advantage']
                txt_embed = traj['txt_embed']
                txt_mask = traj['txt_mask']
                num_steps = len(traj['transitions'])

                # advantage = advantage / num_steps if num_steps > 0 else 0.0

                for trans in traj['transitions']:
                    all_transitions.append({
                        'action': trans['action'],
                        'old_log_prob': trans['old_log_prob'],
                        'advantage': advantage,

                        # Text
                        'txt_embed': txt_embed,
                        'txt_mask': txt_mask,

                        # Graph Map
                        'gmap_vpids': trans['gmap_vpids'],
                        'gmap_img_embeds': trans['gmap_img_embeds'],
                        'gmap_step_ids': trans['gmap_step_ids'],
                        'gmap_pos_fts': trans['gmap_pos_fts'],
                        'gmap_masks': trans['gmap_masks'],
                        'gmap_pair_dists': trans['gmap_pair_dists'],
                        'gmap_visited_masks': trans['gmap_visited_masks'],

                        # ViewPoint
                        'vp_img_embeds': trans['vp_img_embeds'],
                        'vp_pos_fts': trans['vp_pos_fts'],
                        'vp_masks': trans['vp_masks'],
                        'vp_nav_masks': trans['vp_nav_masks'],
                        'vp_obj_masks': trans['vp_obj_masks'],
                        'vp_cand_vpids': trans['vp_cand_vpids'],
                    })

        if len(all_transitions) == 0:
            return

        for epoch in range(grpo_epochs):
            random.shuffle(all_transitions)

            mini_batch_size = self.args.grpo_batch_size
            total_loss = 0.0
            num_batches = 0
            loss = 0.0
            coef = self.args.batch_size * self.num_samples

            for start_idx in range(0, len(all_transitions), mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, len(all_transitions))
                mini_batch = all_transitions[start_idx:end_idx]
                curr_batch_size = len(mini_batch)

                # ========== 重建所有输入 ==========

                # Text inputs
                txt_embeds = torch.from_numpy(
                    np.stack([t['txt_embed'] for t in mini_batch])
                ).to(device)

                txt_masks = torch.from_numpy(
                    np.stack([t['txt_mask'] for t in mini_batch])
                ).to(device)

                # Graph Map inputs
                batch_gmap_img_embeds = []
                batch_gmap_step_ids = []
                batch_gmap_pos_fts = []
                batch_gmap_visited_masks = []
                batch_gmap_pair_dists = []
                batch_gmap_lens = []

                for t in mini_batch:
                    actual_gmap_len = len(t['gmap_vpids'])
                    batch_gmap_lens.append(actual_gmap_len)

                    gmap_img_embeds = torch.from_numpy(t['gmap_img_embeds'][:actual_gmap_len])
                    gmap_step_ids = torch.from_numpy(t['gmap_step_ids'][:actual_gmap_len]).long()
                    gmap_pos_fts = torch.from_numpy(t['gmap_pos_fts'][:actual_gmap_len])
                    gmap_visited_masks = torch.from_numpy(t['gmap_visited_masks'][:actual_gmap_len]).bool()

                    gmap_pair_dists = torch.from_numpy(
                        t['gmap_pair_dists'][:actual_gmap_len, :actual_gmap_len]
                    )

                    batch_gmap_img_embeds.append(gmap_img_embeds)
                    batch_gmap_step_ids.append(gmap_step_ids)
                    batch_gmap_pos_fts.append(gmap_pos_fts)
                    batch_gmap_visited_masks.append(gmap_visited_masks)
                    batch_gmap_pair_dists.append(gmap_pair_dists)

                batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
                gmap_masks = gen_seq_masks(batch_gmap_lens).to(device)

                gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds).to(device)
                gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).to(device)
                gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).to(device)
                gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).to(device)

                max_gmap_len = max(batch_gmap_lens)
                gmap_pair_dists = torch.zeros(curr_batch_size, max_gmap_len, max_gmap_len).float()
                for i in range(curr_batch_size):
                    gmap_len = batch_gmap_lens[i]
                    gmap_pair_dists[i, :gmap_len, :gmap_len] = batch_gmap_pair_dists[i]
                gmap_pair_dists = gmap_pair_dists.to(device)

                gmap_vpids = [t['gmap_vpids'] for t in mini_batch]

                # ViewPoint inputs
                batch_vp_img_embeds = []
                batch_vp_pos_fts = []
                batch_vp_masks = []
                batch_vp_nav_masks = []

                for t in mini_batch:
                    vp_len = t['vp_masks'].sum()

                    vp_img_embeds = torch.from_numpy(t['vp_img_embeds'][:vp_len])
                    vp_pos_fts = torch.from_numpy(t['vp_pos_fts'][:vp_len])
                    vp_masks = torch.from_numpy(t['vp_masks'][:vp_len]).bool()
                    vp_nav_masks = torch.from_numpy(t['vp_nav_masks'][:vp_len]).bool()

                    batch_vp_img_embeds.append(vp_img_embeds)
                    batch_vp_pos_fts.append(vp_pos_fts)
                    batch_vp_masks.append(vp_masks)
                    batch_vp_nav_masks.append(vp_nav_masks)

                vp_img_embeds = pad_tensors_wgrad(batch_vp_img_embeds).to(device)
                vp_pos_fts = pad_tensors(batch_vp_pos_fts).to(device)
                vp_masks = pad_sequence(batch_vp_masks, batch_first=True).to(device)
                vp_nav_masks = pad_sequence(batch_vp_nav_masks, batch_first=True).to(device)

                vp_cand_vpids = [t['vp_cand_vpids'] for t in mini_batch]

                # ========== 重新前向 ==========
                nav_outs = self.vln_bert.vln_bert.forward_navigation_per_step(
                    txt_embeds, txt_masks, gmap_img_embeds,
                    gmap_step_ids, gmap_pos_fts, gmap_masks,
                    gmap_pair_dists, gmap_visited_masks, gmap_vpids,
                    vp_img_embeds, vp_pos_fts, vp_masks,
                    vp_nav_masks, None, vp_cand_vpids
                )

                # 获取logits
                if self.args.fusion == 'local':
                    new_logits = nav_outs['local_logits']
                elif self.args.fusion == 'global':
                    new_logits = nav_outs['global_logits']
                else:
                    new_logits = nav_outs['fused_logits']

                new_log_probs = torch.log_softmax(new_logits, -1)

                # 收集信息
                actions = torch.LongTensor([t['action'] for t in mini_batch]).to(device)
                old_log_probs = torch.FloatTensor([t['old_log_prob'] for t in mini_batch]).to(device)
                advantages = torch.FloatTensor([t['advantage'] for t in mini_batch]).to(device)

                # 计算损失
                new_selected_log_probs = new_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                ratios = torch.exp(new_selected_log_probs - old_log_probs)

                clip_range = getattr(self.args, 'grpo_clip_range', 0.2)
                ratio_clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range)
                surr1 = advantages * ratios
                surr2 = advantages * ratio_clipped
                surr_loss = -torch.min(surr1, surr2).mean()

                loss += surr_loss
                num_batches += 1

            avg_loss = loss / coef
            self.loss += avg_loss
            self.logs['RL_loss'].append(avg_loss.item())

    def compute_advantages(self, sampled_data):
        """
        计算GRPO的组内相对优势

        Args:
            sampled_data: List[List[Dict]], shape=(K, batch_size)

        Returns:
            sampled_data with 'advantage' field added
        """
        num_samples = len(sampled_data)
        batch_size = len(sampled_data[0])

        # 转换为numpy数组: (K, batch_size)
        rewards_array = np.array([
            [traj['reward'] for traj in sample_k]
            for sample_k in sampled_data
        ])

        if not hasattr(self, 'all_failed_count'):
            self.all_failed_count = 0

        # 对每个instruction计算组内优势
        for i in range(batch_size):
            group_rewards = rewards_array[:, i]  # (K,)

            # GRPO: advantage = reward - mean(group_rewards)
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8

            # advantages = (group_rewards - mean_reward) / std_reward
            advantages = group_rewards - mean_reward

            for k in range(num_samples):
                sampled_data[k][i]['advantage'] = advantages[k]

        return sampled_data