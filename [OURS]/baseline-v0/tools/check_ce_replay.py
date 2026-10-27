"""Replay R2R-CE / RxR-CE expert actions under the pinned habitat stack (Step 1/2 simulator check).

The *_gt.json.gz experts were generated with an older Habitat. Before rendering SFT data
from them, confirm that replaying the stored actions under habitat-sim 0.2.4 still reaches
the stored per-forward-step locations and succeeds. Episodes are sampled evenly across the
scene group. Also records the live action IDs, sensor settings and sliding flag for the
manifest, and saves a few RGB frames.

    python tools/check_ce_replay.py --dataset rxr --gpu 0 --group A --episodes 20 --out logs/check_ce_replay_rxr.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import habitat  # noqa: E402
from habitat.datasets import make_dataset  # noqa: E402

from sim.config import load_expert, make_ce_config, scene_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["r2r", "rxr"], required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--group", default="A")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--save-frames", type=int, default=2, help="episodes whose RGB frames are saved")
    ap.add_argument("--gpu", type=int, default=0, help="Habitat EGL device; do not remap with CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    expert = load_expert(args.dataset, args.split)
    cfg = make_ce_config(args.dataset, args.split, scenes=scene_group(args.group), gpu=args.gpu)
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    pool = len(dataset.episodes)
    picks = np.unique(np.linspace(0, pool - 1, min(args.episodes, pool)).round().astype(int))
    dataset.episodes = [dataset.episodes[i] for i in picks]
    env = habitat.Env(config=cfg, dataset=dataset)
    frames_dir = f"{ROOT}/logs/replay_frames/{args.dataset}"

    rows = []
    for i in range(len(dataset.episodes)):
        obs = env.reset()
        ep = env.current_episode
        ref = expert[str(ep.episode_id)]
        locs, k, dev = np.array(ref["locations"]), 0, []
        frames = [obs["rgb"]]
        for a in ref["actions"]:
            if env.episode_over:
                break
            obs = env.step(a)
            if a == 1:
                k += 1
                pos = env.sim.get_agent_state().position
                dev.append(float(np.linalg.norm(pos - locs[min(k, len(locs) - 1)])))
            if i < args.save_frames:
                frames.append(obs["rgb"])
        m = env.get_metrics()
        rows.append({"episode_id": ep.episode_id, "scene": ep.scene_id.split("/")[-2], "steps": len(ref["actions"]),
                     "language": getattr(ep.instruction, "language", "en"),
                     "max_loc_dev_m": max(dev) if dev else 0.0, "final_loc_dev_m": dev[-1] if dev else 0.0,
                     "success": m["success"], "distance_to_goal": m["distance_to_goal"], "spl": m["spl"]})
        if i < args.save_frames:
            os.makedirs(frames_dir, exist_ok=True)
            np.savez_compressed(f"{frames_dir}/ep{ep.episode_id}.npz", rgb=np.stack(frames), actions=ref["actions"])
        print(rows[-1], flush=True)

    rgb = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "group": args.group,
        "episodes_in_group": pool,
        "n": len(rows),
        "scenes_covered": len({r["scene"] for r in rows}),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "max_loc_dev_m": max(r["max_loc_dev_m"] for r in rows),
        "mean_final_loc_dev_m": float(np.mean([r["final_loc_dev_m"] for r in rows])),
        "task_action_order": list(cfg.habitat.task.actions.keys()),
        "rgb_obs_shape": [rgb.height, rgb.width, 3],
        "sensor_position": list(rgb.position),
        "agent_height": cfg.habitat.simulator.agents.main_agent.height,
        "allow_sliding": env.sim.habitat_config.habitat_sim_v0.allow_sliding,
        "episodes": rows,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print({k: v for k, v in report.items() if k != "episodes"})
    env.close()


if __name__ == "__main__":
    main()
