"""Group-A visits from one immutable experimental policy plus real F/L/R branches."""
import argparse
import fcntl
import hashlib
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
for key in ("MAGNUM_LOG", "HABITAT_SIM_LOG"):
    os.environ.setdefault(key, "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

import habitat
import numpy as np
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from PIL import Image

from policy.experimental_reference import ExperimentalReference, sha256, verify_bundle
from policy.online import OnlineInputs
from sim.config import load_manifest, make_ce_config, scene_group
from sim.episodes import select_episodes
from sim.transition_branch import movement_branches, pose


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def pixel_sha(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def summarize(out):
    rows = [read(p) for p in sorted(out.glob("*/*/meta.json")) if not p.parent.name.startswith(".")]
    counts = Counter(a for r in rows for a in r["executed_action"])
    branches = [b for r in rows for b in r["branches"]]
    return {"episodes": len(rows), "actions": sum(r["num_actions"] for r in rows),
            "movement_actions": sum(v for a, v in counts.items() if a != 0),
            "action_counts": {str(k): v for k, v in sorted(counts.items())},
            "branch_roots": len(branches) // 3, "branch_transitions": len(branches),
            "branch_action_counts": dict(Counter(str(b["action"]) for b in branches)),
            "forward_attempts": counts[1],
            "forward_collisions": sum(a == 1 and c for r in rows for a, c in zip(r["executed_action"], r["collided_after_action"])),
            "branch_forward_collisions": sum(b["action"] == 1 and b["collided"] for b in branches),
            "by_dataset": {name: {"episodes": sum(r["dataset"] == name for r in rows),
                                    "scenes": len({r["scene"] for r in rows if r["dataset"] == name}),
                                    "successes": sum(r["metrics"]["success"] for r in rows if r["dataset"] == name)}
                           for name in ("r2r", "rxr")}}


def worker(args, config, chosen):
    import torch
    torch.manual_seed(args.seed)
    policy = ExperimentalReference(args.reference)
    assert not any(p.requires_grad for p in policy.model.parameters())
    out = Path(args.out)
    for name, episodes in chosen.items():
        episodes = episodes[args.worker_index::args.workers]
        pending = []
        for ep in episodes:
            target = out / name / str(ep.episode_id) / "meta.json"
            if target.exists():
                assert read(target)["collection_config_sha256"] == sha256(out / "collection_config.json")
            else:
                pending.append(ep)
        if not pending:
            continue
        cfg = make_ce_config(name, "train", scenes=scene_group("A"), gpu=0)
        dataset = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
        dataset.episodes = sorted(pending, key=lambda ep: (ep.scene_id, str(ep.episode_id)))
        env = habitat.Env(config=cfg, dataset=dataset)
        try:
            assert list(cfg.habitat.task.actions) == ["stop", "move_forward", "turn_left", "turn_right"]
            for _ in pending:
                obs = env.reset()
                ep = env.current_episode
                eid, scene = str(ep.episode_id), ep.scene_id.split("/")[-2]
                assert scene in scene_group("A")
                temp = out / name / f".tmp_{eid}_{args.worker_index}"
                if temp.exists():
                    shutil.rmtree(temp)
                (temp / "rgb").mkdir(parents=True)
                (temp / "branches").mkdir()
                actions, poses, collisions, distances, scores, hashes, branches, files = [], [], [], [], [], [], [], {}
                online = OnlineInputs(policy.inputs, policy.device)
                started = time.time()

                def save_rgb(rgb, relative):
                    target = temp / relative
                    Image.fromarray(rgb).save(target, compress_level=1)
                    files[relative] = sha256(target)

                while not env.episode_over:
                    t = len(actions)
                    rgb = np.array(obs["rgb"], copy=True)
                    poses.append(pose(env.sim))
                    distances.append(float(env.get_metrics()["distance_to_goal"]))
                    hashes.append(pixel_sha(rgb))
                    save_rgb(rgb, f"rgb/{t:04d}.png")
                    online.append(Image.fromarray(rgb))
                    batch = online(ep.instruction.instruction_text, actions)
                    raw = policy.raw_action_scores(batch)
                    adjusted = raw.clone()
                    adjusted[:, 0] += policy.stop_bias
                    action = int(adjusted.argmax(1).item())
                    assert np.isfinite(raw.cpu().numpy()).all()
                    scores.append(raw[0].tolist())
                    if t >= 3 and (t - 3) % args.branch_every == 0:
                        for branch in movement_branches(env):
                            image = branch.pop("rgb")
                            relative = f"branches/{t:04d}_{branch['action']}.png"
                            save_rgb(image, relative)
                            branches.append({**branch, "t": t, "target_frame": relative,
                                             "root_rgb_sha256": hashes[-1], "next_rgb_sha256": pixel_sha(image)})
                    # Only this step updates task metrics and real policy history.
                    obs = env.step(action)
                    actions.append(action)
                    collisions.append(bool(env.sim.previous_step_collided))
                    if len(actions) % 100 == 0:
                        print(json.dumps({"dataset": name, "episode_id": eid, "steps": len(actions),
                                          "branches": len(branches), "seconds": time.time() - started}), flush=True)
                rgb = np.array(obs["rgb"], copy=True)
                save_rgb(rgb, f"rgb/{len(actions):04d}.png")
                hashes.append(pixel_sha(rgb))
                poses.append(pose(env.sim))
                distances.append(float(env.get_metrics()["distance_to_goal"]))
                meta = {"data_kind": "wm_selected_policy", "group": "A", "split": "train", "dataset": name,
                        "episode_id": eid, "trajectory_id": ep.trajectory_id, "scene": scene,
                        "instruction": {"text": ep.instruction.instruction_text, "language": getattr(ep.instruction, "language", "en")},
                        "policy_identity_sha256": policy.identity_sha256,
                        "collection_config_sha256": sha256(out / "collection_config.json"),
                        "num_actions": len(actions), "num_frames": len(actions) + 1,
                        "executed_action": actions, "raw_action_scores": scores, "stop_bias": policy.stop_bias,
                        "position": [p["position"] for p in poses], "rotation_wxyz": [p["rotation_wxyz"] for p in poses],
                        "collided_after_action": collisions, "distances_to_goal": distances,
                        "raw_rgb_sha256": hashes, "file_sha256": files, "branches": branches,
                        "termination": "STOP" if actions[-1] == 0 else "step_limit",
                        "metrics": {k: float(env.get_metrics()[k]) for k in ("success", "spl", "distance_to_goal")},
                        "seconds": time.time() - started}
                write(temp / "meta.json", meta)
                final = out / name / eid
                assert not final.exists()
                temp.rename(final)
                print(json.dumps({"dataset": name, "episode_id": eid, "steps": len(actions),
                                  "branches": len(branches), "success": meta["metrics"]["success"]}), flush=True)
        finally:
            env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=str(ROOT / "runs/step4_smolvlm2_refined512/reference"))
    ap.add_argument("--episodes", type=int, default=128, help="per dataset")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--branch-every", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worker-index", type=int)
    args = ap.parse_args()
    assert args.episodes >= args.workers >= 1 and args.branch_every > 0
    reference = verify_bundle(args.reference)
    assert load_manifest()["simulator"] == reference["identity_fields"]["simulator"]
    chosen = {}
    for name in ("r2r", "rxr"):
        cfg = make_ce_config(name, "train", scenes=scene_group("A"), gpu=0)
        dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        chosen[name] = select_episodes(dataset.episodes, args.episodes, args.seed)
        assert all(ep.scene_id.split("/")[-2] in scene_group("A") for ep in chosen[name])
    files = ["tools/collect_wm_policy.py", "sim/transition_branch.py", "sim/config.py", "sim/episodes.py", "sim/rxr_dataset.py", "manifest/scene_partition.json"]
    config = {"format": "wm_policy_visits_v1", "group": "A", "split": "train", "seed": args.seed,
              "reference": str(Path(args.reference).resolve()), "policy_identity_sha256": reference["policy_identity_sha256"],
              "reference_manifest_sha256": sha256(Path(args.reference) / "file_sha256.json"),
              "episodes_per_dataset": args.episodes, "workers": args.workers,
              "selection": "scene-balanced seeded group-A episodes, unfiltered by expert or policy success",
              "selected_episodes": {name: [str(ep.episode_id) for ep in eps] for name, eps in chosen.items()},
              "decoder": reference["decoder"], "simulator": load_manifest()["simulator"],
              "branch_rule": {"first_t": 3, "every": args.branch_every, "actions": [1, 2, 3],
                              "restore": "native body/sensor translations and quaternions; static scenes; task metrics and actual history unchanged"},
              "command_duration": {"ce_commands": 1, "sim_step_seconds": 1 / 60},
              "frame_encoding": "PNG lossless RGB uint8 640x480; compress_level=1; no resize/crop",
              "policy_preprocessing": reference["identity_fields"]["pixel_preprocessing"],
              "frames": "T executed actions, T+1 actual frames; STOP excluded when indexing dynamics",
              "source_sha256": {p: sha256(ROOT / p) for p in files}}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        assert 0 <= args.worker_index < args.workers
        assert read(out / "collection_config.json") == config
        return worker(args, config, chosen)
    lock = open(out / "collection.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "collection_config.json").exists():
        assert read(out / "collection_config.json") == config, "Collection source or settings changed"
    else:
        write(out / "collection_config.json", config)
    if (out / "summary.json").exists():
        assert read(out / "summary.json")["status"] == "complete"
        print((out / "summary.json").read_text(), flush=True)
        return
    children, logs = [], []
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Collection interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    start, last = time.time(), -1
    try:
        for rank in range(args.workers):
            log = open(out / f"worker_{rank}.log", "a")
            logs.append(log)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(out), "--reference", args.reference,
                       "--episodes", str(args.episodes), "--seed", str(args.seed), "--branch-every", str(args.branch_every),
                       "--workers", str(args.workers), "--worker-index", str(rank)]
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [child.poll() for child in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Collection worker failed: {codes}; inspect worker logs")
            done = len([p for p in out.glob("*/*/meta.json") if not p.parent.name.startswith(".")])
            if done != last:
                write(out / "progress.json", {"episodes": done, "expected": 2 * args.episodes, "elapsed_seconds": time.time() - start})
                print(json.dumps({"episodes": done, "expected": 2 * args.episodes}), flush=True)
                last = done
            if all(code is not None for code in codes):
                break
            time.sleep(5)
        rows = [read(p) for p in out.glob("*/*/meta.json") if not p.parent.name.startswith(".")]
        assert {(r["dataset"], r["episode_id"]) for r in rows} == {(name, str(ep.episode_id)) for name, eps in chosen.items() for ep in eps}
        summary = summarize(out)
        assert summary["episodes"] == args.episodes * 2
        write(out / "summary.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                                     "policy_identity_sha256": reference["policy_identity_sha256"],
                                     "seconds_this_run": time.time() - start, **summary})
        print((out / "summary.json").read_text(), flush=True)
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
