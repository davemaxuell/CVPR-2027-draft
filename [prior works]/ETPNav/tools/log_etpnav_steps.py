#!/usr/bin/env python3
"""Run the released ETPNav checkpoints on R2R-CE / RxR-CE and log every decision for the navigation explorer
(ScaleVLN/analysis/vln_viz). Navigation is ETPNav's own evaluation rollout (argmax, back_algo=control, the
released eval settings); this only records what the model saw and chose.

  cd "[prior works]/ETPNav"
  conda activate etpnav
  python tools/log_etpnav_steps.py --task r2r --split val_unseen                       # all 1,839 episodes
  python tools/log_etpnav_steps.py --task rxr --split val_unseen --episodes 100        # 100 evenly spaced
  python tools/log_etpnav_steps.py --task r2r --split val_unseen --ids 1 2 3

RxR-CE uses English guide episodes (--languages en-US,en-IN), like the explorer's RxR runs.

Writes, under --out (default data/logs/explorer/etpnav_<task>):
  episodes_<split>.jsonl   one line per finished episode, appended as episodes finish; a rerun skips episodes
                           already logged, so an interrupted run resumes
  views_<split>/<episode>/n<k>_<i>.jpg   six world-aligned cube faces at every map node (skybox stand-ins)
  meta_<split>.json        settings and the aggregate metrics of the logged episodes
Positions are habitat coordinates (y up); the explorer converts them to Matterport's (x, -z, y).

Each step (one high-level decision at map node n<t>) has:
  vp        node where the decision is taken ('n<t>')
  gmap      the topological map in ETPNav's order: visited nodes 'n<k>', then ghost waypoints 'g<k>'
  visited   1/0 per gmap entry
  fused     action probabilities over [STOP] + gmap (ETPNav has one, global, branch: global == fused)
  choice    chosen ghost, or null for STOP; forced=true when STOP was imposed (step budget or no ghost left)
  stop_at   for STOP: the visited node with the highest stop score, where the agent walks back to and stops
  front     for a move: the visited node the agent walks to before heading for the ghost
  cands     this node's waypoint predictions: [[x, y, z], id they were merged into ('n<k>' or 'g<k>')]
  heading   agent heading, Matterport convention (0 = +y, clockwise)
  attn      cross-attention of the chosen node (STOP: this node) over instruction tokens
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('MAGNUM_LOG', 'quiet')

import numpy as np  # noqa: E402
import faulthandler  # noqa: E402
import signal  # noqa: E402

faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 <pid> prints every thread's stack

# vlnce_baselines does a bare `import tensorflow` it never uses; stand in for it rather than install TF next to
# torch 2.4 (TF 2.13 pins typing-extensions below what torch needs).
try:
    import tensorflow  # noqa: F401
except ImportError:
    from tensorboard.compat import tensorflow_stub   # what tensorboard itself falls back to without TF
    sys.modules['tensorflow'] = tensorflow_stub

TASKS = {
    'r2r': {'config': 'run_r2r/iter_train.yaml', 'ckpt': 'data/logs/checkpoints/release_r2r/ckpt.iter12000.pth',
            'sliding': True, 'data': 'data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz',
            'tokenizer': 'bert_config/bert-base-uncased'},
    'rxr': {'config': 'run_rxr/iter_train.yaml', 'ckpt': 'data/logs/checkpoints/release_rxr/ckpt.iter19600.pth',
            'sliding': False, 'data': 'data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_guide.json.gz',
            'tokenizer': 'bert_config/xlm-roberta-base'},
}
SAMPLE_SPACING = 0.5   # metres between the track points the explorer plays back (and gets geodesics for)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--split', default='val_unseen')
    p.add_argument('--episodes', default='all', help="'all' or N evenly spaced episodes")
    p.add_argument('--ids', nargs='+', help='episode ids to run')
    p.add_argument('--languages', default='en-US,en-IN', help='RxR-CE languages (comma separated, * = all)')
    p.add_argument('--envs', type=int, default=8, help='parallel habitat environments (one scene set each)')
    p.add_argument('--no-views', action='store_true', help='skip the cube-map renders at map nodes')
    p.add_argument('--view-size', type=int, default=448, help='cube face resolution')
    p.add_argument('--out', help='output directory (default data/logs/explorer/etpnav_<task>)')
    return p.parse_args()


def load_episodes(task, split, languages):
    data = json.load(gzip.open(TASKS[task]['data'].format(split=split)))['episodes']
    if task == 'rxr' and '*' not in languages:
        data = [e for e in data if e['instruction']['language'] in languages]
    return data


def select(episodes, opts):
    ids = [str(e['episode_id']) for e in episodes]
    if opts.ids:
        missing = set(opts.ids) - set(ids)
        if missing:
            sys.exit(f'unknown episode ids: {sorted(missing)[:5]}')
        return [i for i in ids if i in set(opts.ids)]
    if opts.episodes == 'all':
        return ids
    n = int(opts.episodes)
    return ids if n >= len(ids) else [ids[round(k * len(ids) / n)] for k in range(n)]


def to_tokens(tokenizer, ids, task):
    """Token strings in BERT's convention, which the explorer merges into words ('##' continues a word)."""
    toks = tokenizer.convert_ids_to_tokens(ids)
    if task != 'rxr':
        return toks
    special = {'<s>': '[CLS]', '</s>': '[SEP]', '<pad>': '[PAD]', '▁': '[PAD]'}
    return [special.get(t) or (t[1:] if t.startswith('▁') else '##' + t) for t in toks]


def heading_of(ori):
    """Matterport heading (0 = +y, clockwise towards +x) of a habitat rotation [x, y, z, w]."""
    import quaternion
    q = quaternion.quaternion(ori[3], ori[0], ori[1], ori[2])
    f = quaternion.rotate_vectors(q, np.array([0.0, 0.0, -1.0]))
    return math.atan2(f[0], -f[2]) % (2 * math.pi)


def r3(p):
    return [round(float(x), 3) for x in p]


def main():
    opts = parse_args()
    spec = TASKS[opts.task]
    languages = [x for x in opts.languages.split(',') if x]
    out = Path(opts.out or f'data/logs/explorer/etpnav_{opts.task}')
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / f'episodes_{opts.split}.jsonl'
    views_dir = out / f'views_{opts.split}'
    done = set()
    if log_file.is_file():
        for line in open(log_file):
            if line.strip():
                done.add(json.loads(line)['episode_id'])
    episodes = load_episodes(opts.task, opts.split, languages)
    wanted = select(episodes, opts)
    todo = [i for i in wanted if i not in done]
    scenes = {str(e['episode_id']): e['scene_id'] for e in episodes}
    n_scenes = len({scenes[i] for i in todo})
    print(f'{len(wanted)} episodes selected, {len(wanted) - len(todo)} already logged, {len(todo)} to run '
          f'over {n_scenes} scenes', flush=True)
    if not todo:
        return
    num_envs = max(1, min(opts.envs, n_scenes))

    import torch
    import torch.nn.functional as F
    from habitat import logger
    from habitat_baselines.common.baseline_registry import baseline_registry
    from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
    from habitat_baselines.utils.common import batch_obs
    from fastdtw import fastdtw
    from transformers import AutoTokenizer
    import habitat_extensions  # noqa: F401
    import vlnce_baselines  # noqa: F401
    import explorer_env
    from habitat_extensions.measures import NDTW
    from vlnce_baselines.common.utils import extract_instruction_tokens
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.models.graph_utils import GraphMap
    from vlnce_baselines.ss_trainer_ETP import RLTrainer

    exp = f'explorer_{opts.task}'
    cfg_opts = ['EVAL.SPLIT', opts.split, 'EVAL.CKPT_PATH_DIR', spec['ckpt'], 'IL.back_algo', 'control',
                'TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING', str(spec['sliding']),
                'NUM_ENVIRONMENTS', str(num_envs), 'MODEL.pretrained_path', 'None', 'ENV_NAME', 'VLNCEExplorerEnv',
                'SIMULATOR_GPU_IDS', '[0]', 'TORCH_GPU_IDS', '[0]', 'GPU_NUMBERS', '1']
    if opts.task == 'rxr':
        cfg_opts += ['EVAL.LANGUAGES', str(languages)]
    config = get_config(spec['config'], cfg_opts)
    config.defrost()
    config.TENSORBOARD_DIR += exp
    config.CHECKPOINT_FOLDER += exp
    config.RESULTS_DIR += exp
    config.VIDEO_DIR += exp
    config.LOG_FILE = exp + '_' + config.LOG_FILE
    config.local_rank = 0
    # Skybox stand-ins: six square 90-degree RGB cameras at the RGB camera's height (see explorer_env.CUBE_FACES)
    sim = config.TASK_CONFIG.SIMULATOR
    if not opts.no_views:
        for i, (uuid, orient) in enumerate(explorer_env.CUBE_FACES.items()):
            cam = sim.RGB_SENSOR.clone()
            cam.WIDTH = cam.HEIGHT = opts.view_size
            cam.HFOV = 90
            cam.ORIENTATION = [float(x) for x in orient]
            cam.UUID = uuid
            setattr(sim, uuid.upper(), cam)
            sim.AGENT_0.SENSORS.append(uuid.upper())
    camera_height = float(sim.RGB_SENSOR.POSITION[1])
    config.freeze()
    os.makedirs('data/logs/running_log', exist_ok=True)
    logger.add_filehandler('data/logs/running_log/' + config.LOG_FILE)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    tokenizer = AutoTokenizer.from_pretrained(spec['tokenizer'])
    cube_ids = list(explorer_env.CUBE_FACES)
    t_start = time.time()

    class ExplorerTrainer(RLTrainer):
        """ETPNav's trainer with a recording copy of its eval rollout."""

        def collect_val_traj(self):
            keep = set(todo)
            return [t for t in super().collect_val_traj() if t in keep]

        def _initialize_policy(self, *a, **kw):
            out = super()._initialize_policy(*a, **kw)
            ckpt = {k.replace('module.', ''): v for k, v in
                    torch.load(self.config.IL.ckpt_to_load, map_location='cpu')['state_dict'].items()}
            own = self.policy.state_dict()
            missing = [k for k in own if k not in ckpt]
            unexpected = [k for k in ckpt if k not in own]
            print(f'checkpoint: {len(own) - len(missing)}/{len(own)} policy tensors loaded; '
                  f'missing {missing[:6]}; unused {unexpected[:6]}', flush=True)
            self._attn = []
            for layer in self.policy.net.vln_bert.global_encoder.encoder.x_layers:
                layer.visual_attention.att.register_forward_hook(self._capture_attn)
            return out

        def _capture_attn(self, module, inputs, output):
            # output[1]: masked pre-softmax scores (batch, heads, gmap nodes, tokens)
            self._attn.append(torch.softmax(output[1], -1).mean(1))

        def rollout(self, mode, ml_weight=None, sample_ratio=None):
            assert mode == 'eval'
            self.envs.resume_all()
            observations = self.envs.reset()
            instr_max_len = self.config.IL.max_text_len
            instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                      max_length=instr_max_len, pad_id=instr_pad_id)
            cubes = [{k: o.pop(k, None) for k in cube_ids} for o in observations]  # never fed to the model
            del cubes
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) if ep.episode_id in self.stat_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0:
                return

            # one record per running env, in env order (popped alongside the gmaps when an env finishes)
            recs = []
            for i, info in enumerate(self.envs.call(['explorer_episode'] * self.envs.num_envs)):
                ids = batch['instruction'][i].tolist()
                n_tok = sum(1 for x in ids if x != instr_pad_id)
                info.pop('tokens', None)
                info.update(tokens=to_tokens(tokenizer, ids[:n_tok], opts.task), steps=[], seg_end=[1],
                            nodes={}, ghosts={}, ghost_fronts={}, views=not opts.no_views)
                recs.append(info)

            all_txt_ids = batch['instruction']
            all_txt_masks = (all_txt_ids != instr_pad_id)
            all_txt_embeds = self.policy.net(mode='language', txt_ids=all_txt_ids, txt_masks=all_txt_masks)

            not_done_index = list(range(self.envs.num_envs))
            self.gmaps = [GraphMap(False, self.config.IL.loc_noise, self.config.MODEL.merge_ghost, 0)
                          for _ in range(self.envs.num_envs)]
            prev_vp = [None] * self.envs.num_envs

            for stepk in range(self.max_len):
                if not opts.no_views:
                    self.envs.call(['explorer_cube'] * self.envs.num_envs,
                                   [{'prefix': str(views_dir / r['episode_id'] / f'n{stepk}')} for r in recs])
                txt_masks = all_txt_masks[not_done_index]
                txt_embeds = all_txt_embeds[not_done_index]

                wp_outputs = self.policy.net(mode="waypoint", waypoint_predictor=self.waypoint_predictor,
                                             observations=batch, in_train=False)
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs.update({'mode': 'panorama'})
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                    torch.sum(pano_masks, 1, keepdim=True)

                cur_pos, cur_ori = self.get_pos_ori()
                cur_vp, cand_vp, cand_pos = [], [], []
                for i in range(self.envs.num_envs):
                    cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                        cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    cur_vp.append(cur_vp_i)
                    cand_vp.append(cand_vp_i)
                    cand_pos.append(cand_pos_i)

                for i in range(self.envs.num_envs):
                    cur_embeds = avg_pano_embeds[i]
                    cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i] == 1]
                    self.gmaps[i].update_graph(prev_vp[i], stepk + 1, cur_vp[i], cur_pos[i], cur_embeds,
                                               cand_vp[i], cand_pos[i], cand_embeds, None)

                nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
                nav_inputs.update({'mode': 'navigation', 'txt_embeds': txt_embeds, 'txt_masks': txt_masks})
                no_vp_left = nav_inputs.pop('no_vp_left')
                self._attn = []
                nav_outs = self.policy.net(**nav_inputs)
                nav_logits = nav_outs['global_logits']
                nav_probs = F.softmax(nav_logits, 1)
                attn = torch.stack(self._attn).mean(0) if self._attn else None
                for i, gmap in enumerate(self.gmaps):
                    gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

                a_t = nav_logits.argmax(dim=-1)
                cpu_a_t = a_t.cpu().numpy()

                env_actions = []
                use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
                for i, gmap in enumerate(self.gmaps):
                    rec = recs[i]
                    vid = lambda v: f'n{v}' if not v.startswith('g') else v
                    gmap_ids = nav_inputs['gmap_vp_ids'][i]
                    n = len(gmap_ids)
                    probs = [round(x, 5) for x in nav_probs[i, :n].tolist()]
                    rec['nodes'][f'n{cur_vp[i]}'] = r3(cur_pos[i])
                    for g in gmap.ghost_aug_pos:
                        rec['ghosts'][g] = r3(gmap.ghost_aug_pos[g])
                        rec['ghost_fronts'][g] = [f'n{v}' for v in gmap.ghost_fronts[g]]
                    cands = []
                    for c in cand_pos[i]:
                        nid = gmap._localize(c, gmap.node_pos)
                        if nid is not None:
                            cands.append([r3(c), f'n{nid}'])
                            continue
                        gid = next((g for g, ps in gmap.ghost_pos.items() if any(np.allclose(c, q) for q in ps)), None)
                        cands.append([r3(c), gid])
                    step = {'t': stepk, 'vp': f'n{cur_vp[i]}', 'gmap': [vid(v) for v in gmap_ids[1:]],
                            'visited': [int(x) for x in nav_inputs['gmap_visited_masks'][i, 1:n].tolist()],
                            'fused': probs, 'global': probs, 'cands': cands,
                            'heading': round(heading_of(cur_ori[i]), 4)}
                    stop = cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]
                    if stop:
                        vp_stop_scores = [(vp, s) for vp, s in gmap.node_stop_scores.items()]
                        stop_vp = vp_stop_scores[int(np.argmax([s[1] for s in vp_stop_scores]))][0]
                        stop_pos = gmap.node_pos[stop_vp]
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]][1:]
                        env_actions.append({'action': {'act': 0, 'cur_vp': cur_vp[i], 'stop_vp': stop_vp,
                                                       'stop_pos': stop_pos, 'back_path': back_path,
                                                       'tryout': use_tryout}, 'vis_info': None})
                        step.update(choice=None, forced=bool(cpu_a_t[i] != 0), stop_at=f'n{stop_vp}',
                                    stop_scores={f'n{k}': round(v, 5) for k, v in gmap.node_stop_scores.items()})
                        row = 1 + list(gmap.node_pos).index(cur_vp[i])
                    else:
                        ghost_vp = gmap_ids[cpu_a_t[i]]
                        ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                        _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                        front_pos = gmap.node_pos[front_vp]
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]][1:]
                        env_actions.append({'action': {'act': 4, 'cur_vp': cur_vp[i], 'front_vp': front_vp,
                                                       'front_pos': front_pos, 'ghost_vp': ghost_vp,
                                                       'ghost_pos': ghost_pos, 'back_path': back_path,
                                                       'tryout': use_tryout}, 'vis_info': None})
                        prev_vp[i] = front_vp
                        step.update(choice=ghost_vp, forced=False, front=f'n{front_vp}')
                        row = int(cpu_a_t[i])
                        if self.config.MODEL.consume_ghost:
                            gmap.delete_ghost(ghost_vp)
                    if attn is not None:
                        step['attn'] = [round(x, 4) for x in attn[i, row, :len(rec['tokens'])].tolist()]
                    rec['steps'].append(step)

                outputs = self.envs.step(env_actions)
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                for o in observations:
                    for k in cube_ids:
                        o.pop(k, None)

                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    rec = recs[i]
                    rec['seg_end'].append(len(infos[i]['position']['position']))
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    first = ep_id not in self.stat_eps
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()
                    if first:
                        self._finish(i, rec, metric, pred_path, distances, gt_path)

                if sum(dones) > 0:
                    for i in reversed(list(range(self.envs.num_envs))):
                        if dones[i]:
                            not_done_index.pop(i)
                            self.envs.pause_at(i)
                            observations.pop(i)
                            self.gmaps.pop(i)
                            prev_vp.pop(i)
                            recs.pop(i)
                if self.envs.num_envs == 0:
                    break
                observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        def _finish(self, i, rec, metric, track, distances, gt_locations):
            """Geodesics for the explorer (while env i still holds the episode) and one JSONL line."""
            track = [r3(p) for p in track]
            # Track points the explorer plays back: every SAMPLE_SPACING metres, plus every decision's arrival.
            keep, last = {0}, np.array(track[0])
            ends = {e - 1 for e in rec['seg_end']}
            for k in range(1, len(track)):
                if k in ends or np.linalg.norm(np.array(track[k]) - last) >= SAMPLE_SPACING:
                    keep.add(k)
                    last = np.array(track[k])
            keep = sorted(keep)
            refs = rec['reference_path'] + [rec['goal']]
            points = list(rec['nodes'].values()) + list(rec['ghosts'].values()) + [track[k] for k in keep]
            geo = self.envs.call_at(i, 'explorer_geodesics', {'points': points, 'targets': refs})
            n_nodes, n_ghosts = len(rec['nodes']), len(rec['ghosts'])
            rnd = lambda row: [round(x, 3) if math.isfinite(x) else None for x in row]
            rec.update(
                track=track, track_goal_dist=[round(float(d), 3) for d in distances], samples=keep,
                geo={'node': {k: rnd(geo[j]) for j, k in enumerate(rec['nodes'])},
                     'ghost': {k: rnd(geo[n_nodes + j]) for j, k in enumerate(rec['ghosts'])},
                     'sample': [rnd(row) for row in geo[n_nodes + n_ghosts:]]},
                ref_geo=[rnd(row) for row in self.envs.call_at(i, 'explorer_geodesics',
                                                                 {'points': refs, 'targets': refs})],
                gt_locations=[r3(p) for p in gt_locations],
                metrics={k: round(float(v), 4) for k, v in metric.items()},
                camera_height=camera_height, max_steps=self.max_len)
            with open(log_file, 'a') as f:
                f.write(json.dumps(rec, separators=(',', ':')) + '\n')

    trainer = ExplorerTrainer(config)
    trainer.eval()

    rows = [json.loads(line)['metrics'] for line in open(log_file) if line.strip()]
    agg = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]} if rows else {}
    meta = {'model': f'etpnav_{opts.task}', 'checkpoint': spec['ckpt'], 'task': f'{opts.task}-ce', 'split': opts.split,
            'languages': languages if opts.task == 'rxr' else None, 'max_action_len': int(config.IL.max_traj_len),
            'episodes': len(rows), 'metrics': agg, 'views': not opts.no_views,
            'settings': {'back_algo': 'control', 'allow_sliding': spec['sliding'], 'camera_height': camera_height},
            'note': 'continuous (VLN-CE)'}
    (out / f'meta_{opts.split}.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(rows)} episodes logged in {log_file} ({time.time() - t_start:.0f} s)', flush=True)
    print(json.dumps(agg), flush=True)


if __name__ == '__main__':
    main()
