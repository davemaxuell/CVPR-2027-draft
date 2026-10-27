"""Collect round 2 with a goal-first 2m STOP expert; preserve round-1 artifacts.

The policy executes every action (beta=0). The expert never modifies its inputs
or overrides its choices. Completed episode folders are atomic and resumable.
T actions store T+1 actual RGB frames, including the final transition observation.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for name in ("MAGNUM_LOG", "HABITAT_SIM_LOG"):
    os.environ.setdefault(name, "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

import habitat
import numpy as np
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from PIL import Image

from sim.config import load_manifest, make_ce_config, scene_group
from sim.episodes import select_episodes
from sim.recovery_stop_v2 import RECOVERY_RULE, RouteExpert
from tools.eval_policy import atomic_json, sha256

EXPERT_ROOT = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupA")


def selected_episodes(dataset, name, count, seed):
    rows = [json.loads(line) for line in (EXPERT_ROOT / f"{name}_index.jsonl").read_text().splitlines()]
    valid = {str(row["episode_id"]) for row in rows
             if row["success"] == 1 and not row["truncated"] and row["max_location_deviation_m"] < 0.5}
    return select_episodes([ep for ep in dataset.episodes if str(ep.episode_id) in valid], count, seed)


def summarize(out):
    rows = [json.loads(path.read_text()) for path in sorted(out.glob("*/*/meta.json")) if not path.parent.name.startswith(".")]
    counts, errors = Counter(), Counter()
    for row in rows:
        counts.update(a for a in row["expert_action"] if a is not None)
        errors.update(d["error"] for d in row["expert_details"] if d["error"] is not None)
    return {"completed_episodes": len(rows), "actions": sum(r["num_actions"] for r in rows),
            "valid_labels": sum(counts.values()), "label_action_counts": {str(k): v for k, v in sorted(counts.items())},
            "label_errors": dict(errors),
            "disagreements": sum(a is not None and a != b for r in rows for a, b in zip(r["expert_action"], r["executed_action"])),
            "by_dataset": {name: {"episodes": sum(r["dataset"] == name for r in rows),
                                   "scenes": len({r["scene"] for r in rows if r["dataset"] == name}),
                                   "successes": sum(r["metrics"]["success"] for r in rows if r["dataset"] == name)}
                           for name in ("r2r", "rxr")}}


def worker(args, signature):
    out = Path(args.out)
    if args.controller == "policy":
        import torch
        from policy.common import PolicyConfig
        from policy.online import OnlineInputs
        from policy.smolvlm2 import SmolVLM2Inputs, SmolVLM2Policy
        torch.manual_seed(args.seed)
        config = PolicyConfig.load(Path(args.run_dir) / "policy_config.json")
        inputs = SmolVLM2Inputs(config)
        policy = SmolVLM2Policy(config, adapter=args.checkpoint, device="cuda:0")
        policy.model.eval()
    for name in ("r2r", "rxr"):
        cfg = make_ce_config(name, "train", scenes=scene_group("A"), gpu=0)
        full = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        chosen = selected_episodes(full, name, args.episodes, args.seed)
        chosen = chosen[args.worker_index::args.workers]
        pending = [ep for ep in chosen if not (out / name / str(ep.episode_id) / "meta.json").exists()]
        if not pending:
            continue
        dataset = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
        dataset.episodes = sorted(pending, key=lambda e: (e.scene_id, str(e.episode_id)))
        env = habitat.Env(config=cfg, dataset=dataset)
        try:
            assert list(cfg.habitat.task.actions) == ["stop", "move_forward", "turn_left", "turn_right"]
            for _ in pending:
                obs = env.reset()
                episode = env.current_episode
                eid = str(episode.episode_id)
                scene = episode.scene_id.split("/")[-2]
                assert scene in scene_group("A")
                temp = out / name / f".tmp_{eid}_{args.worker_index}"
                if temp.exists():
                    shutil.rmtree(temp)
                (temp / "rgb").mkdir(parents=True)
                reference_path = EXPERT_ROOT / name / eid / "meta.json"
                reference = json.loads(reference_path.read_text())
                expert = RouteExpert(env.sim, episode, reference["position"])
                online = OnlineInputs(inputs, policy.device) if args.controller == "policy" else None
                executed, labels, details, positions, rotations, collisions, distances = [], [], [], [], [], [], []
                started = time.time()
                while not env.episode_over:
                    t = len(executed)
                    state = env.sim.get_agent_state()
                    position = state.position.copy()
                    rotation = [float(state.rotation.w), float(state.rotation.x), float(state.rotation.y), float(state.rotation.z)]
                    positions.append(position.tolist())
                    rotations.append(rotation)
                    distances.append(float(env.get_metrics()["distance_to_goal"]))
                    image = Image.fromarray(obs["rgb"])
                    image.save(temp / "rgb" / f"{t:04d}.jpg", quality=95)
                    label, detail = expert.query()
                    after = env.sim.get_agent_state()
                    assert np.array_equal(position, after.position), "Expert query changed the actual pose"
                    assert rotation == [float(after.rotation.w), float(after.rotation.x), float(after.rotation.y), float(after.rotation.z)]
                    if args.controller == "policy":
                        online.append(image)
                        batch = online(episode.instruction.instruction_text, executed)
                        action = int(policy.action_scores(batch).argmax(1).item())
                    elif t < args.perturb_steps:
                        action = (3, 3, 1, 1, 2, 1, 2, 2)[t % 8]
                    else:
                        if label is None:
                            raise RuntimeError(f"Invalid expert action in control check: {name}/{eid}, t={t}, {detail}")
                        action = label
                    labels.append(label)
                    details.append(detail)
                    obs = env.step(action)
                    executed.append(action)
                    collisions.append(bool(env.sim.previous_step_collided))
                    if len(executed) % 100 == 0:
                        print(json.dumps({"dataset": name, "episode_id": eid, "actions": len(executed),
                                          "valid_labels": sum(a is not None for a in labels), "seconds": time.time() - started}), flush=True)
                final_state = env.sim.get_agent_state()
                Image.fromarray(obs["rgb"]).save(temp / "rgb" / f"{len(executed):04d}.jpg", quality=95)
                metrics = env.get_metrics()
                meta = {"data_kind": "dagger", "dataset": name, "split": "train", "group": "A", "scene": scene,
                        "episode_id": eid, "trajectory_id": episode.trajectory_id,
                        "instruction": {"text": episode.instruction.instruction_text,
                                        "language": getattr(episode.instruction, "language", "en")},
                        "num_actions": len(executed), "num_frames": len(executed) + 1,
                        "expert_action": labels, "executed_action": executed,
                        "expert_details": details, "expert_route": [p.tolist() for p in expert.route],
                        "expert_reference_meta": str(reference_path), "expert_reference_sha256": sha256(reference_path),
                        "reference_snap_distances_m": expert.snap_distances,
                        "position": positions, "rotation_wxyz": rotations, "collided_after_action": collisions,
                        "final_position": final_state.position.tolist(),
                        "final_rotation_wxyz": [float(final_state.rotation.w), float(final_state.rotation.x), float(final_state.rotation.y), float(final_state.rotation.z)],
                        "distances_to_goal": distances + [float(metrics["distance_to_goal"])],
                        "termination": "STOP" if executed[-1] == 0 else "step_limit",
                        "metrics": {k: float(metrics[k]) for k in ("success", "spl", "distance_to_goal")},
                        "collection_signature_sha256": sha256(out / "collection_config.json"),
                        "seconds": time.time() - started}
                atomic_json(temp / "meta.json", meta)
                final = out / name / eid
                if final.exists():
                    raise RuntimeError(f"Unexpected existing incomplete episode: {final}")
                temp.rename(final)
                print(json.dumps({"dataset": name, "episode_id": eid, "actions": len(executed),
                                  "success": meta["metrics"]["success"], "seconds": meta["seconds"]}), flush=True)
        finally:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", default="runs/step3_smolvlm2/train")
    ap.add_argument("--checkpoint", default="runs/step3_smolvlm2/train/checkpoints/step_001024")
    ap.add_argument("--episodes", type=int, default=64, help="episodes per dataset")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worker-index", type=int)
    ap.add_argument("--controller", choices=["policy", "expert"], default="policy")
    ap.add_argument("--perturb-steps", type=int, default=0, help="legal perturbation prefix for expert control checks only")
    args = ap.parse_args()
    if args.episodes < args.workers or args.workers < 1 or args.perturb_steps < 0:
        ap.error("Require episodes >= workers >= 1 and nonnegative perturbation steps")
    if args.controller == "policy" and args.perturb_steps:
        ap.error("Policy collection must execute only the policy")
    out = Path(args.out).resolve()
    args.out = str(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = {"group": "A", "split": "train", "controller": args.controller, "beta": 0.0,
                 "episodes_per_dataset": args.episodes, "seed": args.seed, "workers": args.workers,
                 "perturb_steps": args.perturb_steps, "recovery_rule": RECOVERY_RULE,
                 "selection": "seeded scene-balanced selection from the validated Step 2 group-A expert pool; policy failures retained", "jpeg_quality": 95,
                 "expert_pool_sha256": {name: sha256(EXPERT_ROOT / f"{name}_index.jsonl") for name in ("r2r", "rxr")},
                 "frames": "T actions and T+1 actual RGB frames; prefix uses executed actions",
                 "simulator": load_manifest()["simulator"],
                 "source_sha256": {name: sha256(ROOT / name) for name in
                                   ("tools/collect_dagger_stop.py", "sim/recovery_stop_v2.py", "sim/recovery.py", "sim/config.py", "sim/episodes.py", "sim/rxr_dataset.py",
                                    "policy/smolvlm2.py", "policy/common.py", "policy/online.py", "manifest/scene_partition.json")}}
    if args.controller == "policy":
        signature.update(checkpoint=str(Path(args.checkpoint).resolve()),
                         adapter_sha256=sha256(Path(args.checkpoint) / "adapter_model.safetensors"),
                         policy_config=json.loads((Path(args.run_dir) / "policy_config.json").read_text()))
    config_path = out / "collection_config.json"
    if args.worker_index is None:
        if config_path.exists() and json.loads(config_path.read_text()) != signature:
            raise ValueError("Collection settings changed; choose a new output directory")
        atomic_json(config_path, signature)
    elif json.loads(config_path.read_text()) != signature:
        raise ValueError("Worker collection settings differ from parent")
    if args.worker_index is not None:
        return worker(args, signature)
    if (out / "summary.json").exists():
        print((out / "summary.json").read_text(), flush=True)
        return
    children, logs = [], []
    started, done = time.time(), -1
    def terminate(signum, frame):
        raise KeyboardInterrupt("Collection interrupted")
    signal.signal(signal.SIGTERM, terminate)
    try:
        (out / "workers").mkdir(exist_ok=True)
        for rank in range(args.workers):
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(out),
                       "--run-dir", args.run_dir, "--checkpoint", args.checkpoint,
                       "--episodes", str(args.episodes), "--seed", str(args.seed),
                       "--workers", str(args.workers), "--worker-index", str(rank),
                       "--controller", args.controller, "--perturb-steps", str(args.perturb_steps)]
            log = open(out / "workers" / f"{rank}.stdout.log", "a")
            logs.append(log)
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Collection worker failed: {codes}; inspect {out}/workers")
            report = summarize(out)
            if report["completed_episodes"] != done:
                report["seconds_this_run"] = time.time() - started
                atomic_json(out / "progress.json", report)
                print(json.dumps(report), flush=True)
                done = report["completed_episodes"]
            if all(code is not None for code in codes):
                break
            time.sleep(5)
        assert report["completed_episodes"] == 2 * args.episodes
        for name in ("r2r", "rxr"):
            cfg = make_ce_config(name, "train", scenes=scene_group("A"), gpu=0)
            dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
            selected = selected_episodes(dataset, name, args.episodes, args.seed)
            expected = {str(ep.episode_id) for ep in selected}
            actual = {path.parent.name for path in (out / name).glob("*/meta.json") if not path.parent.name.startswith(".")}
            assert actual == expected
            rows = [json.loads((out / name / eid / "meta.json").read_text()) for eid in sorted(expected)]
            (out / f"{name}_index.jsonl").write_text("".join(json.dumps({"episode_id": r["episode_id"], "scene": r["scene"],
                "num_actions": r["num_actions"], "valid_labels": sum(a is not None for a in r["expert_action"]),
                "success": r["metrics"]["success"]}) + "\n" for r in rows))
        report.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), seconds_this_run=time.time() - started)
        atomic_json(out / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
