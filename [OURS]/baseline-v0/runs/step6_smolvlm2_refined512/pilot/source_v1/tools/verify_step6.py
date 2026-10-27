"""Audit every completed Step 6 root and summarize measured candidate advantage."""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from policy.stop_diagnostics import ndtw_trace
from scorer_data.common import DATA, RUN, job_seed, now, read, sha, verify_protocol, write
from sim.config import load_expert, scene_group


def describe(rows):
    branches = [b for r in rows for b in r['branches']]
    gains = np.asarray([r['best_candidate_gain'] for r in rows])
    gaps = [max(r['utilities_live_order']) - min(r['utilities_live_order']) for r in rows]
    outcomes = [{(b['success'], round(b['ndtw'], 9), b['remaining_actions'], b['remaining_collisions']) for b in r['branches']} for r in rows]
    return {'roots': len(rows), 'branches': len(branches),
            'episodes': len({(r['dataset'], r['mode'], r['episode_id']) for r in rows}),
            'scenes': sorted({r['scene'] for r in rows}),
            'successful_branches': sum(b['success'] for b in branches),
            'roots_with_successful_candidate': sum(any(b['success'] for b in r['branches']) for r in rows),
            'base_successes': sum(r['branches'][r['base_action_live']]['success'] for r in rows),
            'best_utility_candidate_successes': sum(r['branches'][int(np.argmax(r['utilities_live_order']))]['success'] for r in rows),
            'mean_base_utility': float(np.mean([r['utilities_live_order'][r['base_action_live']] for r in rows])),
            'mean_best_measured_utility': float(np.mean([max(r['utilities_live_order']) for r in rows])),
            'mean_best_candidate_gain': float(gains.mean()), 'roots_with_gain_over_1e_4': int((gains > 1e-4).sum()),
            'mean_best_worst_label_gap': float(np.mean(gaps)), 'median_best_worst_label_gap': float(np.median(gaps)),
            'distinct_outcome_counts': dict(collections.Counter(str(len(x)) for x in outcomes)),
            'timeouts': sum(b['termination'] == 'step_limit' for b in branches),
            'remaining_actions': sum(b['remaining_actions'] for b in branches),
            'remaining_forward_collisions': sum(b['remaining_collisions'] for b in branches),
            'immediate_forward_collisions': sum(b['immediate_forward_collision'] for b in branches),
            'by_candidate': {str(a): {'name': rows[0]['branches'][a]['candidate_name'],
                            'successes': sum(r['branches'][a]['success'] for r in rows),
                            'mean_ndtw': float(np.mean([r['branches'][a]['ndtw'] for r in rows])),
                            'mean_remaining_actions': float(np.mean([r['branches'][a]['remaining_actions'] for r in rows]))} for a in range(4)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    args = ap.parse_args()
    protocol = verify_protocol()
    signature = sha(RUN / 'protocol.json')
    stage = RUN / args.stage
    status, index = read(stage / 'status.json'), read(stage / 'index.json')
    assert status['status'] == 'complete' and index['protocol_sha256'] == signature
    assert len(index['roots']) == len(set(index['roots'])) == status['roots']
    experts = {name: load_expert(name, 'train') for name in ('r2r', 'rxr')}
    selection = read(RUN / 'selection.json')
    records, carriers, image_files, decoded = [], {}, 0, 0
    for relative in index['roots']:
        path = DATA / relative
        r = read(path)
        group, name, mode, eid, _, t_text, filename = Path(relative).parts
        t = int(t_text)
        assert r['group'] == group and r['dataset'] == name and r['mode'] == mode and r['episode_id'] == eid and r['root_t'] == t
        assert r['protocol_sha256'] == signature and r['policy_identity_sha256'] == protocol['policy_identity_sha256']
        assert eid in selection[group][name][mode] and r['scene'] in scene_group(group)
        assert group in ('B', 'C') and r['scene'] not in set(scene_group('A')) | set(scene_group('E'))
        carrier_path = (path.parent / r['carrier']).resolve()
        key = str(carrier_path)
        if key not in carriers:
            c = read(carrier_path)
            assert sha(carrier_path) == r['carrier_sha256']
            assert c['protocol_sha256'] == signature and c['eligible'] and c['mode'] == mode
            assert c['seed'] == job_seed(group, name, mode, eid)
            assert len(c['poses']) == len(c['rgb_sha256']) == c['num_actions'] + 1
            assert len(c['actions']) == len(c['forward_collisions']) == c['num_actions']
            assert c['num_actions'] <= c['budget'] == protocol['simulator']['max_episode_steps'][name]
            if mode == 'policy':
                raw = np.asarray(c['raw_action_scores'])
                raw[:, 0] -= .25
                assert np.array_equal(raw.argmax(1), c['actions'])
            for filename, expected in c['file_sha256'].items():
                image = carrier_path.parent / filename
                assert sha(image) == expected
                with Image.open(image) as im:
                    assert im.size == (640, 480) and im.mode == 'RGB' and im.format == 'PNG'
                    step = int(Path(filename).stem)
                    if step in (0, c['num_actions']) or step in c['root_steps']:
                        import hashlib
                        assert hashlib.sha256(np.array(im).tobytes()).hexdigest() == c['rgb_sha256'][step]
                        decoded += 1
                image_files += 1
            carriers[key] = c
        c = carriers[key]
        assert t >= 3 and t in c['root_steps'] and t < c['num_actions']
        assert r['root_rgb_sha256'] == c['rgb_sha256'][t]
        assert r['base_action_canonical'] == [3, 0, 1, 2][r['base_action_live']]
        shifted = np.asarray(r['base_raw_action_scores']).copy()
        shifted[0] -= .25
        assert int(shifted.argmax()) == r['base_action_live']
        branches = []
        for candidate in range(4):
            bp = path.parent / r['branch_files'][str(candidate)]
            assert sha(bp) == r['branch_sha256'][str(candidate)]
            b = read(bp)
            assert b['protocol_sha256'] == signature and b['policy_identity_sha256'] == protocol['policy_identity_sha256']
            assert b['candidate_live'] == candidate and b['candidate_canonical'] == [3, 0, 1, 2][candidate]
            assert b['root_t'] == t and b['poses'][0] == c['poses'][t] and b['rgb_sha256'][0] == c['rgb_sha256'][t]
            n = len(b['actions'])
            assert n == b['remaining_actions'] == len(b['forward_collisions']) == len(b['raw_action_scores'])
            assert len(b['poses']) == len(b['rgb_sha256']) == n + 1
            assert b['actions'][0] == candidate and b['raw_action_scores'][0] is None
            assert b['terminal_elapsed_steps'] == t + n <= c['budget']
            assert b['original_budget'] == c['budget'] and b['remaining_budget'] == c['budget'] - t
            assert b['actions'][-1] == 0 or t + n == c['budget']
            assert not any(a == 0 for a in b['actions'][:-1])
            for action, collision, raw in zip(b['actions'][1:], b['forward_collisions'][1:], b['raw_action_scores'][1:]):
                assert raw is not None and np.isfinite(raw).all()
                scores = np.asarray(raw).copy(); scores[0] -= .25
                assert int(scores.argmax()) == action
                assert not collision or action == 1
            assert b['remaining_collisions'] == sum(b['forward_collisions'])
            assert b['immediate_forward_collision'] == (candidate == 1 and b['forward_collisions'][0])
            first = path.parent / b['first_next_file']
            assert sha(first) == b['first_next_sha256']
            with Image.open(first) as image:
                import hashlib
                assert image.size == (640, 480) and image.mode == 'RGB'
                assert hashlib.sha256(np.array(image).tobytes()).hexdigest() == b['rgb_sha256'][1]
            image_files += 1; decoded += 1
            if candidate == 0:
                assert n == 1 and b['remaining_collisions'] == 0 and b['rgb_sha256'][1] == c['rgb_sha256'][t]
            positions = np.asarray([p['position'] for p in c['poses'][:t]] + [p['position'] for p in b['poses']], dtype=np.float64)
            ndtw = float(ndtw_trace(positions, np.asarray(experts[name][eid]['locations'], dtype=np.float64))[-1])
            assert abs(ndtw - b['ndtw']) < 1e-12
            utility = .7 * b['success'] + .3 * ndtw - .01 * n / c['budget'] - .02 * b['remaining_collisions'] / c['budget']
            assert abs(utility - b['utility']) < 1e-12
            assert b['success'] in (0, 1) and b['success'] == b['metrics']['success']
            assert b['sdtw'] == b['success'] * ndtw
            branches.append(b)
        assert r['utilities_live_order'] == [b['utility'] for b in branches]
        assert r['successes_live_order'] == [b['success'] for b in branches]
        assert r['best_candidate_gain'] == max(r['utilities_live_order']) - r['utilities_live_order'][r['base_action_live']]
        r['branches'] = branches
        records.append(r)
    report = {'status': 'passed', 'stage': args.stage, 'completed_utc': now(), 'protocol_sha256': signature,
              'source_sha256': sha(__file__), 'index_sha256': sha(stage / 'index.json'),
              'all_branch_labels_checked': len(records) * 4, 'image_files_hashed_and_headers_checked': image_files,
              'images_fully_decoded_and_raw_hash_checked': decoded, 'unique_carriers_checked': len(carriers),
              'all_original_budgets_counts_decoders_full_path_metrics_checked': True,
              'by_group': {}, 'by_stratum': {}, 'interpretation': 'Measured one-action branch advantage under the frozen continuation policy; privileged oracle diagnostic, not scorer or navigation improvement'}
    for group in status['groups']:
        rows = [r for r in records if r['group'] == group]
        report['by_group'][group] = describe(rows)
        for name in ('r2r', 'rxr'):
            for mode in ('expert', 'policy'):
                subset = [r for r in rows if r['dataset'] == name and r['mode'] == mode]
                assert len(subset) == status[f'roots_{group}'] // 4
                report['by_stratum'][f'{group}/{name}/{mode}'] = describe(subset)
    write(stage / 'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
