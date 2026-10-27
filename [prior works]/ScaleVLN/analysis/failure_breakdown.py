"""Break down R2R failures of a DUET/ScaleVLN run into stop errors vs. exploration errors.

usage (from anywhere, scalevln env):
  python analysis/failure_breakdown.py VLN-DUET/datasets/R2R/exprs_map/eval/released_b16/preds/detail_val_unseen.json

Needs predictions written with --submit --detailed_output (per-node stop probabilities).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'VLN-DUET', 'map_nav_src'))
from utils.data import load_nav_graphs  # noqa: E402

ERROR_MARGIN = 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file')
    parser.add_argument('--split', default=None, help='default: parsed from pred file name')
    parser.add_argument('--max_action_len', type=int, default=15)
    args = parser.parse_args()

    split = args.split or os.path.basename(args.pred_file).split('_', 1)[1].rsplit('.', 1)[0]
    data_dir = os.path.join(ROOT, 'VLN-DUET', 'datasets', 'R2R')
    annos = json.load(open(os.path.join(data_dir, 'annotations', 'R2R_%s_enc.json' % split)))
    gt = {}
    for item in annos:
        for j in range(len(item['instructions'])):
            gt['%s_%d' % (item['path_id'], j)] = item
    preds = json.load(open(args.pred_file))

    scans = sorted({gt[p['instr_id']]['scan'] for p in preds})
    graphs = load_nav_graphs(os.path.join(data_dir, 'connectivity_mp3d'), scans)
    dists = {s: dict(nx.all_pairs_dijkstra_path_length(g)) for s, g in graphs.items()}

    cats = Counter()
    by_hops = defaultdict(lambda: [0, 0])
    for p in preds:
        item = gt[p['instr_id']]
        d = dists[item['scan']]
        goal = item['path'][-1]
        segments = p['trajectory']
        path = [vp for seg in segments for vp in seg]
        visited = set(path)
        final = path[-1]
        hops = len(item['path']) - 1
        by_hops[hops][1] += 1

        if d[final][goal] < ERROR_MARGIN:
            cats['success'] += 1
            by_hops[hops][0] += 1
            continue

        near_goal = [vp for vp in visited if d[vp][goal] < ERROR_MARGIN]
        n_actions = len(segments) - 1
        if near_goal:
            # The agent passed within 3 m of the goal but ended elsewhere.
            stop_probs = {k: v['stop_prob'] for k, v in p.get('details', {}).items()}
            best_near = max(stop_probs.get(vp, 0.0) for vp in near_goal)
            if stop_probs and stop_probs.get(final, 0.0) > best_near:
                cats['fail: passed goal, stop score ranked a wrong node higher'] += 1
            else:
                cats['fail: passed goal, other'] += 1
        elif n_actions >= args.max_action_len - 1:  # forced stop at t = max_action_len - 1
            cats['fail: never near goal, ran out of steps'] += 1
        else:
            cats['fail: never near goal, stopped early/elsewhere'] += 1

    n = len(preds)
    print('%s: %d episodes' % (split, n))
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print('  %-58s %5d  %5.1f%%' % (k, v, 100.0 * v / n))
    print('SR by ground-truth path length (hops):')
    for h in sorted(by_hops):
        s, t = by_hops[h]
        print('  %2d hops: SR %5.1f%%  (n=%d)' % (h, 100.0 * s / t, t))


if __name__ == '__main__':
    main()
