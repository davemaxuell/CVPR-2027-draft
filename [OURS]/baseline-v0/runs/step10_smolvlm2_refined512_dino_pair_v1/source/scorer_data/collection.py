"""Actual prefixes, exact replay restoration, and four forced-action outcomes."""
import hashlib
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from policy.online import OnlineInputs
from Scorer.interfaces import ActionMap
from Scorer.labels import utility_target
from sim.transition_branch import pose
from .common import ACTION_NAMES, RUN, now, read, sha, write
from .metrics import terminal_ndtw

ACTION_MAP = ActionMap((1, 2, 3, 0))


def pixel_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def metric_record(env):
    result = {}
    for key in ('success', 'spl', 'distance_to_goal'):
        value = float(env.get_metrics()[key])
        result[key] = value if np.isfinite(value) else str(value)
    return result


def reset_episode(env, episode, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.episode_iterator = iter([episode])
    obs = env.reset()
    assert not env.episode_over and env._elapsed_steps == 0 and not env.task.is_stop_called
    assert not env.sim.sim_config.sim_cfg.enable_physics
    assert not env.sim.get_rigid_object_manager().get_object_handles()
    assert not env.sim.get_articulated_object_manager().get_object_handles()
    return obs


def choose(policy, online, instruction, actions):
    batch = online(instruction, actions)
    raw = policy.raw_action_scores(batch)
    assert raw.shape == (1, 4) and torch.isfinite(raw).all()
    adjusted = raw.clone()
    adjusted[:, 0] += policy.stop_bias
    action = int(adjusted.argmax(1).item())
    return action, raw[0].float().cpu().tolist(), int(batch['lengths'][0])


def root_steps(action_count, maximum=8):
    eligible = np.arange(3, action_count, dtype=int)
    if not len(eligible):
        return []
    return eligible[np.linspace(0, len(eligible) - 1, min(maximum, len(eligible))).round().astype(int)].tolist()


def carrier(env, episode, policy, mode, expert, directory, seed, budget, protocol_sha):
    """Persist full actual history. Expert eligibility uses only its actual replay."""
    directory = Path(directory)
    if (directory / 'meta.json').exists():
        row = read(directory / 'meta.json')
        assert row['protocol_sha256'] == protocol_sha
        online = OnlineInputs(policy.inputs, policy.device)
        if row['eligible']:
            for t in range(row['num_actions'] + 1):
                filename = f'rgb/{t:04d}.png'
                assert sha(directory / filename) == row['file_sha256'][filename]
                with Image.open(directory / filename) as image:
                    pixels = np.array(image.convert('RGB'))
                assert pixel_sha(pixels) == row['rgb_sha256'][t]
                online.append(Image.fromarray(pixels))
        return row, online
    temp = directory.with_name('.' + directory.name + '.partial')
    if temp.exists():
        shutil.rmtree(temp)
    (temp / 'rgb').mkdir(parents=True)
    obs = reset_episode(env, episode, seed)
    online = OnlineInputs(policy.inputs, policy.device)
    actions, scores, poses, hashes, collisions, metrics, files, lengths, deviations = [], [], [], [], [], [], {}, [], []
    started, forward_steps = time.time(), 0

    def save_observation():
        t = len(actions)
        rgb = np.array(obs['rgb'], copy=True)
        relative = f'rgb/{t:04d}.png'
        Image.fromarray(rgb).save(temp / relative, compress_level=1)
        files[relative] = sha(temp / relative)
        hashes.append(pixel_sha(rgb))
        poses.append(pose(env.sim))
        metrics.append(metric_record(env))
        online.append(Image.fromarray(rgb))

    while not env.episode_over:
        save_observation()
        if mode == 'expert':
            if len(actions) >= len(expert['actions']):
                raise RuntimeError('Expert schedule ended before episode termination')
            action, raw, length = int(expert['actions'][len(actions)]), None, None
        else:
            action, raw, length = choose(policy, online, episode.instruction.instruction_text, actions)
        assert action in range(4) and len(actions) < budget
        obs = env.step(action)
        actions.append(action)
        scores.append(raw)
        lengths.append(length)
        collisions.append(bool(action == 1 and env.sim.previous_step_collided))
        if mode == 'expert' and action == 1:
            forward_steps += 1
            target = expert['locations'][min(forward_steps, len(expert['locations']) - 1)]
            deviations.append(float(np.linalg.norm(env.sim.get_agent_state().position - target)))
    save_observation()
    assert len(actions) == env._elapsed_steps <= budget
    steps = root_steps(len(actions))
    valid_expert = mode != 'expert' or (metrics[-1]['success'] == 1 and len(actions) == len(expert['actions']) and max(deviations, default=0) < .5)
    reasons = ([] if valid_expert else ['expert replay failed success/length/drift filter']) + ([] if steps else ['fewer than four observations before termination'])
    row = {'format': 'scorer_actual_carrier_v1', 'protocol_sha256': protocol_sha, 'mode': mode,
           'episode_id': str(episode.episode_id), 'scene': episode.scene_id.split('/')[-2],
           'instruction': {'text': episode.instruction.instruction_text, 'language': getattr(episode.instruction, 'language', 'en')},
           'seed': seed, 'budget': budget, 'num_actions': len(actions), 'actions': actions, 'raw_action_scores': scores,
           'prompt_lengths': lengths, 'poses': poses, 'rgb_sha256': hashes, 'forward_collisions': collisions,
           'metrics_trace': metrics, 'file_sha256': files, 'root_steps': steps if not reasons else [],
           'eligible': not reasons, 'exclusion_reasons': reasons, 'max_expert_deviation_m': max(deviations, default=0),
           'termination': 'STOP' if actions[-1] == 0 else 'step_limit', 'seconds': time.time() - started,
           'policy_identity_sha256': policy.identity_sha256, 'completed_utc': now()}
    write(temp / 'meta.json', row)
    assert not directory.exists()
    temp.rename(directory)
    return row, online


def restore_root(env, episode, recorded, t):
    """Replay every prefix step and assert RGB, pose, metrics and original clock."""
    obs = reset_episode(env, episode, recorded['seed'])
    for i in range(t + 1):
        assert pose(env.sim) == recorded['poses'][i], ('Prefix pose drift', i)
        assert pixel_sha(obs['rgb']) == recorded['rgb_sha256'][i], ('Prefix RGB drift', i)
        assert metric_record(env) == recorded['metrics_trace'][i], ('Prefix metric drift', i)
        assert env._elapsed_steps == i and not env.episode_over
        if i < t:
            action = recorded['actions'][i]
            assert action != 0
            obs = env.step(action)
            assert bool(action == 1 and env.sim.previous_step_collided) == recorded['forward_collisions'][i]
    return obs


def branch(env, episode, recorded, cached, policy, reference_locations, t, candidate, image_path):
    started = time.time()
    obs = restore_root(env, episode, recorded, t)
    root_rgb = pixel_sha(obs['rgb'])
    online = OnlineInputs(policy.inputs, policy.device)
    online.frames = cached.frames[:t + 1]
    prefix = list(recorded['actions'][:t])
    actions, collisions, poses, hashes, scores, lengths = [], [], [pose(env.sim)], [root_rgb], [], []
    budget = recorded['budget']
    while not env.episode_over:
        if not actions:
            action, raw, length = candidate, None, None
        else:
            online.append(Image.fromarray(np.array(obs['rgb'], copy=True)))
            action, raw, length = choose(policy, online, episode.instruction.instruction_text, prefix + actions)
        obs = env.step(action)
        actions.append(action)
        scores.append(raw)
        lengths.append(length)
        collisions.append(bool(action == 1 and env.sim.previous_step_collided))
        poses.append(pose(env.sim))
        hashes.append(pixel_sha(obs['rgb']))
        if len(actions) == 1:
            Image.fromarray(obs['rgb']).save(image_path, compress_level=1)
            if candidate == 0:
                assert hashes[-1] == root_rgb and poses[-1] == poses[0]
        assert t + len(actions) == env._elapsed_steps <= budget
    assert actions[0] == candidate and (actions[-1] == 0 or env._elapsed_steps == budget)
    if candidate == 0:
        assert actions == [0] and collisions == [False]
    full_positions = [p['position'] for p in recorded['poses'][:t]] + [p['position'] for p in poses]
    metrics = metric_record(env)
    success = int(metrics['success'])
    ndtw = terminal_ndtw(full_positions, reference_locations)
    value = utility_target(success, ndtw, len(actions), int(sum(collisions)), budget)
    row = {'candidate_live': candidate, 'candidate_canonical': int(ACTION_MAP.from_simulator(torch.tensor(candidate))),
           'candidate_name': ACTION_NAMES[candidate], 'root_t': t, 'root_rgb_sha256': root_rgb,
           'original_budget': budget, 'remaining_budget': budget - t, 'prefix_actions': t,
           'actions': actions, 'raw_action_scores': scores, 'prompt_lengths': lengths,
           'forward_collisions': collisions, 'poses': poses, 'rgb_sha256': hashes,
           'first_next_file': image_path.name, 'first_next_sha256': sha(image_path),
           'immediate_forward_collision': collisions[0] if candidate == 1 else False,
           'remaining_actions': len(actions), 'remaining_collisions': int(sum(collisions)),
           'terminal_elapsed_steps': env._elapsed_steps, 'metrics': metrics, 'success': success,
           'ndtw': ndtw, 'sdtw': success * ndtw, 'utility': value,
           'termination': 'STOP' if actions[-1] == 0 else 'step_limit',
           'policy_identity_sha256': policy.identity_sha256, 'seconds': time.time() - started,
           'prefix_replay_verified': True, 'completed_utc': now()}
    return row


def collect_root(env, episode, recorded, cached, policy, expert, directory, t, group, dataset, mode, protocol_sha):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'labels.json').exists():
        row = read(directory / 'labels.json')
        assert row['protocol_sha256'] == protocol_sha
        return row
    root_online = OnlineInputs(policy.inputs, policy.device)
    root_online.frames = cached.frames[:t + 1]
    base, raw, length = choose(policy, root_online, episode.instruction.instruction_text, recorded['actions'][:t])
    if mode == 'policy':
        assert base == recorded['actions'][t], ('Policy replay action drift', dataset, episode.episode_id, t)
        assert np.array_equal(raw, recorded['raw_action_scores'][t]), 'Native policy score replay drift'
    rows = []
    for candidate in range(4):
        path = directory / f'branch_{candidate}.json'
        if path.exists():
            outcome = read(path)
            assert outcome['protocol_sha256'] == protocol_sha and outcome['root_t'] == t
            assert sha(directory / outcome['first_next_file']) == outcome['first_next_sha256']
        else:
            outcome = branch(env, episode, recorded, cached, policy, expert['locations'], t, candidate,
                             directory / f'first_next_{candidate}.png')
            outcome['protocol_sha256'] = protocol_sha
            write(path, outcome)
        rows.append(outcome)
    row = {'format': 'four_action_scorer_labels_v1', 'protocol_sha256': protocol_sha, 'group': group,
           'dataset': dataset, 'mode': mode, 'episode_id': str(episode.episode_id), 'scene': recorded['scene'],
           'root_t': t, 'base_action_live': base, 'base_action_canonical': int(ACTION_MAP.from_simulator(torch.tensor(base))),
           'base_raw_action_scores': raw, 'root_prompt_tokens': length, 'carrier': '../../carrier/meta.json',
           'carrier_sha256': sha(directory.parent.parent / 'carrier/meta.json'),
           'root_rgb_sha256': recorded['rgb_sha256'][t], 'policy_identity_sha256': policy.identity_sha256,
           'branch_files': {str(a): f'branch_{a}.json' for a in range(4)},
           'branch_sha256': {str(a): sha(directory / f'branch_{a}.json') for a in range(4)},
           'utilities_live_order': [b['utility'] for b in rows], 'successes_live_order': [b['success'] for b in rows],
           'best_candidate_gain': max(b['utility'] for b in rows) - rows[base]['utility'], 'completed_utc': now()}
    write(directory / 'labels.json', row)
    return row
