"""Render expert trajectories (R2R-CE + RxR-CE) to JPEG frames and per-episode metadata.

Layout under --out:
    {dataset}/{episode_id}/rgb/{t:04d}.jpg   frame t = RGB observation before action t
    {dataset}/{episode_id}/meta.json         instruction, actions, poses, collisions, metrics
    {dataset}_index.jsonl                    one summary row per finished episode
    render_manifest.json                     settings, source hashes, counts

An episode with T actions (ending in STOP) stores T frames. `expert_action` and
`executed_action` are kept separate (identical for expert replay; DAgger data reuses the
format). Each episode is written to a temporary directory and renamed when complete, so an
interrupted run resumes by skipping finished episodes. Every task runs in a fresh process.

    python tools/render_expert.py --out /data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupA --workers 16
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _var in ("MAGNUM_LOG", "HABITAT_SIM_LOG"):
    os.environ.setdefault(_var, "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

from sim.config import load_expert, load_manifest, make_ce_config, scene_group  # noqa: E402
from sim.episodes import select_episodes  # noqa: E402

ACTION_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT"]  # live habitat task order


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def plan_tasks(args):
    """(dataset, scene, episodes, expert subset) chunks, largest first for load balance."""
    from habitat.datasets import make_dataset

    tasks, totals = [], {}
    for name in args.datasets:
        cfg = make_ce_config(name, args.split, scenes=scene_group(args.group))
        episodes = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset).episodes
        expert = load_expert(name, args.split)
        if args.max_episodes:
            episodes = (select_episodes(episodes, args.max_episodes, args.seed)
                        if args.selection == "scene_balanced" else episodes[:args.max_episodes])
        by_scene = {}
        for ep in episodes:
            by_scene.setdefault(ep.scene_id.split("/")[-2], []).append(ep)
        totals[name] = {"episodes": len(episodes), "scenes": len(by_scene),
                        "frames": sum(len(expert[str(ep.episode_id)]["actions"]) for ep in episodes)}
        for scene, eps in by_scene.items():
            for i in range(0, len(eps), args.chunk):
                chunk = eps[i:i + args.chunk]
                tasks.append((name, scene, chunk, {str(e.episode_id): expert[str(e.episode_id)] for e in chunk}))
    tasks.sort(key=lambda t: -sum(len(r["actions"]) for r in t[3].values()))
    return tasks, totals


def render_task(task, args):
    import habitat
    from habitat.core.registry import registry
    from PIL import Image

    name, scene, episodes, expert = task
    out_dir = os.path.join(args.out, name)
    todo = [ep for ep in episodes if not os.path.exists(os.path.join(out_dir, str(ep.episode_id), "meta.json"))]
    if not todo:
        return name, scene, []
    for ep in todo:
        for stale in glob.glob(os.path.join(out_dir, f".tmp_{ep.episode_id}_*")):
            shutil.rmtree(stale, ignore_errors=True)

    cfg = make_ce_config(name, args.split, scenes=[scene], gpu=args.gpu)
    dataset = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
    dataset.episodes = todo
    env = habitat.Env(config=cfg, dataset=dataset)
    rows = []
    for _ in range(len(todo)):
        obs = env.reset()
        ep = env.current_episode
        eid = str(ep.episode_id)
        ref = expert[eid]
        locations = np.asarray(ref["locations"])
        tmp = os.path.join(out_dir, f".tmp_{eid}_{os.getpid()}")
        os.makedirs(os.path.join(tmp, "rgb"))
        positions, rotations, collided, executed, dev = [], [], [], [], []
        for t, action in enumerate(ref["actions"]):
            if env.episode_over:
                break
            state = env.sim.get_agent_state()
            positions.append([float(v) for v in state.position])
            rotations.append([float(state.rotation.w), float(state.rotation.x), float(state.rotation.y), float(state.rotation.z)])
            Image.fromarray(obs["rgb"]).save(os.path.join(tmp, "rgb", f"{t:04d}.jpg"), quality=args.jpeg_quality)
            obs = env.step(action)
            executed.append(action)
            collided.append(bool(env.sim.previous_step_collided))
            if action == 1:
                target = locations[min(sum(a == 1 for a in executed), len(locations) - 1)]
                dev.append(float(np.linalg.norm(env.sim.get_agent_state().position - target)))
        metrics = env.get_metrics()
        instruction = ep.instruction
        meta = {
            "dataset": name, "split": args.split, "group": args.group, "scene": scene,
            "episode_id": eid, "trajectory_id": ep.trajectory_id,
            "instruction": {"text": instruction.instruction_text,
                            "language": getattr(instruction, "language", "en"),
                            "instruction_id": getattr(instruction, "instruction_id", None)},
            "num_frames": len(executed),
            "expert_action": list(ref["actions"][:len(executed)]),
            "executed_action": executed,
            "position": positions,
            "rotation_wxyz": rotations,
            "collided_after_action": collided,
            "truncated": len(executed) < len(ref["actions"]),
            "max_location_deviation_m": max(dev) if dev else 0.0,
            "metrics": {k: float(metrics[k]) for k in ("success", "spl", "distance_to_goal")},
        }
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f)
        final = os.path.join(out_dir, eid)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.rename(tmp, final)
        rows.append({"episode_id": eid, "num_frames": len(executed), "success": meta["metrics"]["success"]})
    env.close()
    return name, scene, rows


def write_indexes(args):
    summary = {}
    for name in args.datasets:
        rows = []
        for meta_path in sorted(glob.glob(os.path.join(args.out, name, "*", "meta.json"))):
            with open(meta_path) as f:
                m = json.load(f)
            rows.append({"episode_id": m["episode_id"], "scene": m["scene"], "num_frames": m["num_frames"],
                         "language": m["instruction"]["language"], "success": m["metrics"]["success"],
                         "truncated": m["truncated"], "max_location_deviation_m": m["max_location_deviation_m"],
                         "collisions": sum(m["collided_after_action"])})
        with open(os.path.join(args.out, f"{name}_index.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        dev = [r["max_location_deviation_m"] for r in rows]
        summary[name] = {"episodes": len(rows), "frames": sum(r["num_frames"] for r in rows),
                         "success_rate": float(np.mean([r["success"] for r in rows])) if rows else None,
                         "failed_episodes": [r["episode_id"] for r in rows if r["success"] < 1],
                         "truncated_episodes": [r["episode_id"] for r in rows if r["truncated"]],
                         "max_location_deviation_m": max(dev) if dev else None,
                         "episodes_over_0.5m_deviation": sum(d > 0.5 for d in dev)}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", default=["r2r", "rxr"], choices=["r2r", "rxr"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--group", default="A")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=150, help="max episodes per task (one scene load each)")
    ap.add_argument("--gpu", type=int, default=0, help="Habitat EGL device; do not remap with CUDA_VISIBLE_DEVICES")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--max-episodes", type=int, default=0, help="per dataset, for pilot runs")
    ap.add_argument("--selection", choices=["first", "scene_balanced"], default="first")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.datasets:
        os.makedirs(os.path.join(args.out, name), exist_ok=True)

    tasks, totals = plan_tasks(args)
    print(f"planned {len(tasks)} tasks: {totals}", flush=True)
    manifest = load_manifest()
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "planned": totals,
        "frame_convention": "frame t is the RGB observation before action t; T actions (ending in STOP) -> T frames",
        "action_names_by_live_id": ACTION_NAMES,
        "simulator": {k: manifest["simulator"][k] for k in ("rgb", "forward_step_m", "turn_angle_deg", "max_episode_steps",
                                                             "success_distance_m", "rxr_turn_handling")},
        "rxr_filter": manifest["datasets"]["rxr_ce"],
        "source_sha256": {p: sha256(os.path.join(ROOT, p)) for p in
                          ("tools/render_expert.py", "sim/config.py", "sim/rxr_dataset.py",
                           "manifest/experiment_manifest.yaml", "manifest/scene_partition.json")},
    }

    start, done_frames, planned_frames = time.time(), 0, sum(t["frames"] for t in totals.values())
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for i, (name, scene, rows) in enumerate(pool.imap_unordered(_worker, [(t, args) for t in tasks]), 1):
            done_frames += sum(r["num_frames"] for r in rows)
            rate = done_frames / max(time.time() - start, 1e-6)
            print(f"[{i}/{len(tasks)}] {name}/{scene}: {len(rows)} episodes | {done_frames}/{planned_frames} frames "
                  f"this run, {rate:.0f} frames/s", flush=True)

    info["finished_utc"] = datetime.now(timezone.utc).isoformat()
    info["wall_seconds_this_run"] = round(time.time() - start, 1)
    info["rendered"] = write_indexes(args)
    with open(os.path.join(args.out, "render_manifest.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info["rendered"], indent=1)[:2000], flush=True)


def _worker(packed):
    task, args = packed
    return render_task(task, args)


if __name__ == "__main__":
    main()
