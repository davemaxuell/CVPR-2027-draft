"""Independent metric arithmetic and live four-way restore/budget/policy checks."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
for key in ('MAGNUM_LOG', 'HABITAT_SIM_LOG'):
    os.environ.setdefault(key, 'quiet')
os.environ.setdefault('GLOG_minloglevel', '2')

import habitat
import numpy as np
import torch
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from PIL import Image
from scipy.spatial.distance import cdist

from policy.common import past_frame_indices
from policy.experimental_reference import ExperimentalReference
from policy.online import OnlineInputs
from policy.stop_diagnostics import ndtw_trace
from Scorer.labels import utility_target
from scorer_data.collection import branch, carrier, choose, metric_record, pixel_sha, reset_episode, restore_root
from scorer_data.common import REFERENCE, RUN, now, read, sha, verify_protocol, write
from scorer_data.metrics import terminal_ndtw
from sim.config import load_expert, make_ce_config, scene_group
from sim.transition_branch import pose


def independent_ndtw(positions, target):
    points = np.asarray(positions, dtype=np.float64)
    points = points[np.r_[True, np.any(points[1:] != points[:-1], axis=1)]]
    costs = cdist(points, np.asarray(target, dtype=np.float64))
    table = np.full((len(points) + 1, len(target) + 1), np.inf)
    table[0, 0] = 0
    for i in range(1, len(points) + 1):
        for j in range(1, len(target) + 1):
            table[i, j] = costs[i - 1, j - 1] + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(np.exp(-table[-1, -1] / (len(target) * 3)))


def main():
    protocol = verify_protocol()
    signature = sha(RUN / 'protocol.json')
    rng = np.random.default_rng(6)
    for _ in range(32):
        points = rng.normal(size=(20, 3)).cumsum(0)
        points[8:11] = points[7]
        target = rng.normal(size=(17, 3)).cumsum(0)
        measured = terminal_ndtw(points, target)
        assert abs(measured - independent_ndtw(points, target)) < 1e-12
        assert abs(measured - ndtw_trace(points, target)[-1]) < 1e-12
    assert utility_target(1, 1., 1, 0, 500) == .99998
    assert utility_target(0, 0., 500, 500, 500) == -.03
    policy = ExperimentalReference(REFERENCE)
    selection = read(RUN / 'selection.json')
    checks = {'independent_nDTW_cases': 32, 'live_datasets': {}, 'policy_trainable_parameters': sum(p.numel() for p in policy.model.parameters() if p.requires_grad)}
    for name in ('r2r', 'rxr'):
        cfg = make_ce_config(name, 'train', scenes=scene_group('B'), gpu=0)
        full = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        lookup = {str(ep.episode_id): ep for ep in full.episodes}
        experts = load_expert(name, 'train')
        candidates = selection['B'][name]['expert'][:20]
        ds = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
        ds.episodes = [lookup[candidates[0]]]
        env = habitat.Env(config=cfg, dataset=ds)
        output = RUN / 'checks' / name
        output.mkdir(parents=True, exist_ok=True)
        budget = protocol['simulator']['max_episode_steps'][name]
        try:
            for eid in candidates:
                ep = lookup[eid]
                directory = output / eid / 'carrier'
                recorded, cached = carrier(env, ep, policy, 'expert', experts[eid], directory, 6, budget, signature)
                if recorded['eligible']:
                    break
                del cached
            else:
                raise AssertionError('No valid expert control episode')
            t = recorded['num_actions'] - 1
            results = {}
            for action in (3, 0, 2, 1):
                results[action] = branch(env, ep, recorded, cached, policy, experts[eid]['locations'], t, action,
                                         output / f'goal_first_next_{action}.png')
            repeat = branch(env, ep, recorded, cached, policy, experts[eid]['locations'], t, 1,
                            output / 'goal_forward_repeat.png')
            for key in ('actions', 'poses', 'rgb_sha256', 'raw_action_scores', 'forward_collisions', 'metrics', 'success', 'ndtw', 'utility'):
                assert repeat[key] == results[1][key], ('Branch order/repeat drift', name, key)
            stop = results[0]
            assert stop['success'] == 1 and stop['remaining_actions'] == 1 and stop['remaining_collisions'] == 0
            full_positions = [p['position'] for p in recorded['poses'][:t]] + [p['position'] for p in stop['poses']]
            assert abs(stop['ndtw'] - independent_ndtw(full_positions, experts[eid]['locations'])) < 1e-12
            suffix_only = terminal_ndtw([p['position'] for p in stop['poses']], experts[eid]['locations'])
            assert abs(stop['ndtw'] - suffix_only) > 1e-5, 'Control must expose omitted-prefix nDTW error'

            # Check native prefill after the forced first action, including its real view/action.
            if len(results[1]['actions']) > 1:
                actual = results[1]
                actions = recorded['actions'][:t] + [1]
                online = OnlineInputs(policy.inputs, policy.device)
                online.frames = cached.frames[:t + 1]
                image = Image.open(output / 'goal_first_next_1.png').convert('RGB')
                online.append(image)
                indices = past_frame_indices(t + 1, policy.cfg) + [t + 1]
                images = [Image.open(directory / f'rgb/{i:04d}.png').convert('RGB') if i <= t else image for i in indices]
                step = {'instruction': ep.instruction.instruction_text, 'executed_actions': actions, 't': t + 1, 'images': images}
                native = policy.inputs([step], with_targets=False)
                current = online(step['instruction'], actions)
                for key in native:
                    assert torch.equal(native[key].cpu(), current[key].cpu()), ('Forced-action prompt drift', key)
                action, raw, _ = choose(policy, online, step['instruction'], actions)
                assert action == actual['actions'][1] and raw == actual['raw_action_scores'][1]
                del online
            del cached
            torch.cuda.empty_cache()

            # A legal wall-contact prefix leaves exactly one original-budget action.
            obs = reset_episode(env, ep, 6)
            edge = {'seed': 6, 'budget': budget, 'actions': [], 'poses': [], 'rgb_sha256': [], 'forward_collisions': [], 'metrics_trace': []}
            for i in range(budget):
                edge['poses'].append(pose(env.sim))
                edge['rgb_sha256'].append(pixel_sha(obs['rgb']))
                edge['metrics_trace'].append(metric_record(env))
                if i < budget - 1:
                    obs = env.step(1)
                    edge['actions'].append(1)
                    edge['forward_collisions'].append(bool(env.sim.previous_step_collided))
            assert sum(edge['forward_collisions']) > 0
            dummy = SimpleNamespace(frames=[None] * budget)  # no policy call is legal after the last allowed action
            for action in range(4):
                row = branch(env, ep, edge, dummy, policy, experts[eid]['locations'], budget - 1, action,
                             output / f'budget_first_next_{action}.png')
                assert row['remaining_actions'] == 1 and row['terminal_elapsed_steps'] == budget
                assert row['raw_action_scores'] == [None]
                if action == 1:
                    assert row['immediate_forward_collision'] and row['remaining_collisions'] == 1
                else:
                    assert not row['immediate_forward_collision'] and row['remaining_collisions'] == 0
                expected = .7 * row['success'] + .3 * row['ndtw'] - .01 / budget - .02 * row['remaining_collisions'] / budget
                assert abs(expected - row['utility']) < 1e-15
            checks['live_datasets'][name] = {'episode_id': eid, 'all_four_candidates': True,
                'candidate_order_and_repeat_identical': True, 'STOP_success_counts_and_identity': True,
                'complete_path_ndtw': stop['ndtw'], 'incorrect_suffix_only_ndtw': suffix_only,
                'forced_action_native_prefill_parity': len(results[1]['actions']) > 1,
                'original_budget_enforced_all_actions': True, 'blocked_forward_mask_and_cost': True}
            print(json.dumps(checks['live_datasets'][name]), flush=True)
        finally:
            env.close()
    checks.update(status='passed', completed_utc=now(), protocol_sha256=signature,
                  policy_identity_sha256=policy.identity_sha256, STOP_bias=policy.stop_bias)
    write(RUN / 'check_runtime.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    main()
