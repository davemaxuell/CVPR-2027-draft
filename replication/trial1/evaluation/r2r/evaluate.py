"""Full official-split greedy CE evaluator with resumable per-episode records.

Habitat owns reset, termination, geodesic distance, Success and SPL. Supplementary metrics
are the actual upstream VLN-CE measure classes loaded by official_metrics.py. This adapter
changes neither the driving policy nor the benchmark's individual-step action space.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import *
sys.path.insert(0, str(CODE))

import habitat
import numpy as np
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from omegaconf import OmegaConf

from sim.config import EXPERT, make_ce_config
from sim.episodes import select_episodes
from stream_policy.rollout import python_command, run_workers
from official_metrics import configure
from policy_client import RecordedClient
from memory import ObservableMemory

EXPECTED_R2R = {'train': 10819, 'val_seen': 778, 'val_unseen': 1839}


def heading(rotation):
    direction = quaternion_rotate_vector(rotation.inverse(), np.array([0, 0, -1]))
    return float(cartesian_to_polar(-direction[2], direction[0])[1])


def configuration(dataset, split):
    # No scene-group filtering: this is the entire official split.
    cfg = make_ce_config(dataset, split, scenes=None, gpu=0)
    return configure(cfg, EXPERT[dataset], split)


def episodes_for(args):
    cfg = configuration(args.dataset, args.split)
    data = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    # Official VLN-CE's NDTW indexes the JSON ground-truth mapping by episode.episode_id.
    # Habitat 0.2.4 retains numeric annotation IDs, while that mapping has string keys.
    for ep in data.episodes:
        ep.episode_id = str(ep.episode_id)
    if args.dataset == 'r2r' and len(data.episodes) != EXPECTED_R2R[args.split]:
        raise RuntimeError(f'Unexpected official R2R split count: {len(data.episodes)}')
    episodes = select_episodes(data.episodes, args.episodes, args.seed)
    return cfg, sorted(episodes, key=lambda e: (e.scene_id, str(e.episode_id)))


def episode(env, client, args, out,side=None):
    obs = env.reset()
    ep = env.current_episode
    key = f'{args.dataset}/{args.split}/{ep.episode_id}'
    client.reset(ep.instruction.instruction_text, key)
    if side is not None:side.reset(ep.instruction.instruction_text,key)
    memory = ObservableMemory(args.memory_mode,every_n=args.every_n,
                              seed=args.seed+zlib.crc32(key.encode()))
    began = time.monotonic()
    actions, decisions, substeps = [], [], []
    policy_seconds = 0.0
    walked = 0.0
    frame_dir = OUTPUT / 'cache' / args.name / key if args.save_frames else None
    if frame_dir:
        frame_dir.mkdir(parents=True, exist_ok=True)
    while not env.episode_over:
        step = len(actions)
        pose = env.sim.get_agent_state()
        position, rotation = pose.position.tolist(), pose.rotation
        if frame_dir:
            from PIL import Image
            Image.fromarray(obs['rgb']).save(frame_dir / f'{step:06d}.jpg', quality=95)
        # Inference input allowlist: actual RGB, instruction, executed action history only.
        shown = memory.inputs(step)
        input_sha256=hashlib.sha256(obs['rgb'].tobytes()).hexdigest() if getattr(args,'capture_input_hashes',False) else None
        client.set_records(shown)
        action, reply = client.act(step, actions[-1] if actions else None, obs['rgb'])
        if side is not None:side.observe(step,actions[-1] if actions else None,obs['rgb'])
        policy_seconds += reply['seconds']
        if reply['replanned']:
            chunk = reply['action_chunk']
            if chunk[0] != action or not 1 <= len(chunk) <= 4:
                raise RuntimeError('Policy chunk/log mismatch.')
            decisions.append(dict(t=len(decisions), sim_step=step, position=position,
                heading=heading(rotation), rotation_wxyz=[float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)],
                action_chunk=chunk, is_stop=0 in chunk,
                action_token_probabilities=reply['action_token_probabilities'],
                action_logits=reply['action_logits'], generated_positions=reply['generated_positions'],
                distribution_support=reply['distribution_support'], feature_path=reply.get('feature_path'),
                saved_places=shown,path_length=walked,
                frame_id=str(frame_dir / f'{step:06d}.jpg') if frame_dir else None))
            if side is not None:side.check(decisions[-1],client,obs)
            memory.observe_decision(dict(t=len(decisions)-1,frame=step,position=position,
                                         heading=heading(rotation),path_length=walked,keep_score=1.0))
        substeps.append(dict(sim_step=step, position=position, heading=heading(rotation),
                             action=action, decision_t=len(decisions) - 1,
                             observation_sha256=input_sha256,
                             distance_to_goal=float(env.get_metrics()['distance_to_goal'])))
        obs = env.step(action)
        walked += float(np.linalg.norm(env.sim.get_agent_state().position-np.asarray(position)))
        actions.append(action)
    final = env.sim.get_agent_state()
    metrics = {k: float(v) for k, v in env.get_metrics().items()}
    if any(not math.isfinite(v) for v in metrics.values()):
        raise RuntimeError('Non-finite official metric; preserve error instead of inventing a score.')
    return dict(episode_id=str(ep.episode_id), trajectory_id=ep.trajectory_id, scene_id=ep.scene_id,
                dataset=args.dataset, split=args.split, mode='greedy_natural',
                instruction=ep.instruction.instruction_text, steps=decisions,
                action_steps=substeps, final_position=final.position.tolist(),
                final_heading=heading(final.rotation), executed_actions=actions,
                metrics=metrics, evidence_label='MEASURED', seconds=time.monotonic() - began,
                policy_seconds=policy_seconds, termination='STOP' if actions[-1] == 0 else 'step_limit')


def worker(args, out):
    cfg, episodes = episodes_for(args)
    pending = [e for e in episodes[args.worker_index::args.workers]
               if not (out / 'episodes' / f'{e.episode_id}.json').exists()]
    if not pending:
        atomic_json(out / 'workers' / f'{args.worker_index}.done.json', dict(status='complete', pending=0))
        return
    data = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
    data.episodes = pending
    env = habitat.Env(config=cfg, dataset=data)
    try:
        assert list(cfg.habitat.task.actions.keys()) == ['stop', 'move_forward', 'turn_left', 'turn_right']
        feature_dir = OUTPUT / 'cache/features' / args.name if args.capture_features else ''
        with (out / 'workers' / f'{args.worker_index}.server.log').open('a') as log:
            with RecordedClient(args.run_dir, args.weights, log, feature_dir,args.force_slot) as client:
                atomic_json(out / 'workers' / f'{args.worker_index}.identity.json', client.identity)
                for _ in pending:
                    row = episode(env, client, args, out)
                    atomic_json(out / 'episodes' / f'{row["episode_id"]}.json', row)
                    print(json.dumps({k: row[k] for k in ['episode_id', 'metrics', 'seconds']}), flush=True)
                atomic_json(out / 'workers' / f'{args.worker_index}.stats.json', client.call({'op': 'stats'}))
    finally:
        env.close()
    atomic_json(out / 'workers' / f'{args.worker_index}.done.json', dict(status='complete'))


def summarize(rows):
    names = sorted(rows[0]['metrics'])
    return {name: sum(row['metrics'][name] for row in rows) / len(rows) for name in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['r2r', 'rxr'], default='r2r')
    ap.add_argument('--split', choices=['train', 'val_seen', 'val_unseen'], required=True)
    ap.add_argument('--name', required=True, help='Immutable job output identifier')
    ap.add_argument('--run-dir', default='')
    ap.add_argument('--weights', default='')
    ap.add_argument('--selection',default='',help='Resolve the selected run and weights before creating the evaluation identity')
    ap.add_argument('--episodes', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--capture-features', action='store_true')
    ap.add_argument('--save-frames', action='store_true')
    ap.add_argument('--memory-mode',choices=['empty','every_n','random'],default='empty')
    ap.add_argument('--every-n',type=float,default=3.0)
    ap.add_argument('--force-slot',action='store_true')
    ap.add_argument('--worker-index', type=int, default=None)
    args = ap.parse_args()
    settings = read_json(OUTPUT / 'settings.json')
    if args.selection:
        selected = read_json(args.selection)['selected']
        args.run_dir,args.weights = selected['run_dir'],selected['weights']
        args.selection = ''  # Workers receive the resolved immutable paths.
    args.run_dir = args.run_dir or settings['current_run_dir']
    args.weights = args.weights or read_json(OUTPUT / 'phase0/checkpoint.json')['checkpoint']
    if args.name.startswith('phase1') and args.episodes:
        ap.error('Phase 1 requires the complete official split.')
    out = OUTPUT / args.name
    (out / 'episodes').mkdir(parents=True, exist_ok=True)
    (out / 'workers').mkdir(exist_ok=True)
    if args.worker_index is not None:
        worker(args, out)
        return
    cfg, episodes = episodes_for(args)
    signature = dict(dataset=args.dataset, split=args.split, selected_episodes=[str(e.episode_id) for e in episodes],
                     seed=args.seed, run_dir=args.run_dir, weights=args.weights,
                     checkpoint_sha256={p.name: sha256(p) for p in Path(args.weights).glob('model*.safetensors')},
                     simulator=OmegaConf.to_container(cfg, resolve=True),
                     official_measure_source=read_json(OUTPUT / 'vendor/vlnce/SOURCE.json'),
                     instrumentation_sha256=sha256(QUEUE_CODE / 'policy_server.py'),
                     evaluation_sha256=sha256(QUEUE_CODE / 'evaluate.py'),
                     metric_loader_sha256=sha256(QUEUE_CODE / 'official_metrics.py'),
                     slot_layout_sha256=sha256(QUEUE_CODE/'memory.py'),
                     memory_mode=args.memory_mode,every_n=args.every_n,force_slot=args.force_slot,
                     queue_policy_metadata=read_json(Path(args.run_dir)/'queue_policy.json',{}),
                     capture_features=args.capture_features, save_frames=args.save_frames)
    old = read_json(out / 'config.json')
    if old and old != signature:
        raise RuntimeError('Different evaluation identity in this output; create a new job identifier.')
    atomic_json(out / 'config.json', signature)
    if read_json(out / 'complete.json', {}).get('status') == 'complete':
        print((out / 'complete.json').read_text())
        return
    def progress():
        files = list((out / 'episodes').glob('*.json'))
        atomic_json(out / 'progress.json', dict(completed=len(files), expected=len(episodes), time=utc()))
    run_workers(python_command(Path(__file__).resolve(), args), args.workers, out, progress, poll=20)
    rows = [read_json(out / 'episodes' / f'{e.episode_id}.json') for e in episodes]
    if any(r is None for r in rows):
        raise RuntimeError('Missing evaluation episodes.')
    ids = [read_json(p) for p in (out / 'workers').glob('*.identity.json')]
    if len({json.dumps({k: v for k, v in x.items() if k not in {'device', 'seconds', 'nbytes'}}, sort_keys=True) for x in ids}) != 1:
        raise RuntimeError('Evaluation policy identities differ.')
    trace = out / f'trajectories_{args.split}.jsonl'
    # Rebuild from immutable per-episode records. A crash cannot duplicate JSONL entries.
    temporary = trace.with_suffix('.partial')
    with temporary.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, trace)
    scalar_path = out / 'metrics_per_episode.csv'
    scalar_tmp = scalar_path.with_suffix('.partial')
    with scalar_tmp.open('w', newline='') as f:
        fields = ['episode_id','trajectory_id','scene_id','dataset','split'] + sorted(rows[0]['metrics'])
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k] for k in fields if k in row}, **row['metrics']})
        f.flush()
        os.fsync(f.fileno())
    os.replace(scalar_tmp, scalar_path)
    report = dict(status='complete', evidence_label='MEASURED', split=args.split, dataset=args.dataset,
                  episodes=len(rows), scenes=len({r['scene_id'] for r in rows}), metrics=summarize(rows),
                  per_episode_path=str(out / 'episodes'), trajectory_path=str(trace),
                  per_episode_scalar_metrics=str(scalar_path),
                  official_measure_source=signature['official_measure_source'], completed_utc=utc(),
                  evaluation_role='report_only', model_role='expanded_stage_two')
    atomic_json(out / 'metrics.json', report)
    atomic_json(out / 'complete.json', report)
    refresh_sync()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
