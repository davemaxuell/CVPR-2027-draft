"""Run fresh R2R inference and export an offline trajectory and photo player."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['b16', 'h14'], default='b16')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path)
    options = parser.parse_args()
    output = options.output.resolve()
    nav = ROOT / 'VLN-DUET/map_nav_src'
    data = ROOT / 'VLN-DUET/datasets'
    os.chdir(nav)
    sys.path.insert(0, str(nav))
    from r2r.parser import parse_args
    sys.argv = [sys.argv[0], '--root_dir', str(data), '--output_dir', str(output),
                '--features', 'clip.' + options.model, '--image_feat_size',
                '512' if options.model == 'b16' else '1024', '--batch_size', str(min(3, options.episodes)),
                '--fusion', 'dynamic', '--enc_full_graph', '--graph_sprels', '--train_alg', 'dagger',
                '--num_l_layers', '9', '--num_x_layers', '4', '--num_pano_layers', '2',
                '--max_action_len', '15', '--max_instr_len', '200', '--detailed_output']
    args = parse_args()
    from utils.misc import set_random_seed
    from r2r.data_utils import construct_instrs
    from r2r.env import R2RNavBatch
    from r2r.agent import GMapNavAgent
    import h5py
    import torch
    set_random_seed(0)

    class LazyFeatures:
        def get_image_feature(self, scan, viewpoint):
            with h5py.File(args.val_ft_file, 'r') as stream:
                return stream[f'{scan}_{viewpoint}'][...][:, :args.image_feat_size].astype('float32')

    rows = construct_instrs(args.anno_dir, 'r2r', ['val_unseen'], 'bert', max_instr_len=200)
    if options.episodes > len(rows):
        raise ValueError(f'Maximum number of val_unseen instructions: {len(rows)}')
    selected = [rows[index * len(rows) // options.episodes] for index in range(options.episodes)]
    # The environment shuffles its list in place, so keep the selection order separately.
    selected_ids = [row['instr_id'] for row in selected]
    env = R2RNavBatch(LazyFeatures(), list(selected), args.connectivity_dir,
                      batch_size=args.batch_size, name='demo_val_unseen')
    agent = GMapNavAgent(args, env, rank=0)
    checkpoint = options.checkpoint or data / 'R2R/trained_models/finetune' / f'duet_vit-{options.model}_ft_best_val_unseen'
    agent.load(str(checkpoint))
    with torch.no_grad():
        agent.test(use_dropout=False, feedback='argmax')
    predictions = agent.get_results(detailed_output=True)
    metrics, _ = env.eval_metrics(predictions)
    by_id = {item['instr_id']: item for item in selected}
    pred_by_id = {item['instr_id']: item for item in predictions}
    scenes = ROOT.parents[1] / 'data/mattersim/v1/scans'
    episodes = []
    for instr_id in selected_ids:
        item, prediction = by_id[instr_id], pred_by_id[instr_id]
        path = [vp for segment in prediction['trajectory'] for vp in segment]
        graph = env.graphs[item['scan']]
        images = {}
        for vp in dict.fromkeys(path):
            images[vp] = []
            for face in range(6):
                name = f'{vp}_skybox{face}_sami.jpg'
                source = scenes / item['scan'] / 'matterport_skybox_images' / name
                if not source.is_file():
                    raise FileNotFoundError(f'Demo photograph missing: {source}')
                target = output / 'assets' / item['scan'] / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
                images[vp].append(str(target.relative_to(output)))
        episodes.append(dict(instr_id=instr_id, scan=item['scan'], instruction=item['instruction'],
                             reference=item['path'], predicted=path, images=images,
                             nodes={vp: list(graph.nodes[vp]['position']) for vp in graph},
                             edges=list(graph.edges),
                             navigation_error=env.shortest_distances[item['scan']][path[-1]][item['path'][-1]],
                             success=env.shortest_distances[item['scan']][path[-1]][item['path'][-1]] < 3))
    report = dict(model=options.model, checkpoint=str(checkpoint), split='val_unseen',
                  generated_utc=datetime.now(timezone.utc).isoformat(), fresh_inference=True,
                  metrics=metrics, episodes=episodes)
    (output / 'demo.json').write_text(json.dumps(report, indent=2))
    (output / 'predictions.json').write_text(json.dumps(predictions, indent=2))
    template = (ROOT / 'tools/demo.html').read_text()
    # Keep annotations as data, including any literal HTML characters.
    payload = json.dumps(report).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (output / 'index.html').write_text(template.replace('__DEMO_DATA__', payload))
    print('Fresh-inference demo:', output / 'index.html')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
