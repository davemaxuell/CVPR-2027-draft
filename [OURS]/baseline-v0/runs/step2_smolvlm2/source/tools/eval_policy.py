"""Closed-loop group-C navigation, with resumable per-episode records.

The model sees only the instruction and its actual RGB/action prefix. Habitat
computes terminal success, SPL and distance; oracle success is reported separately.
Episode selection does not depend on expert replay success. No group-E tuning.

    python tools/eval_policy.py --run-dir runs/sft --checkpoint ... --out runs/eval
    python tools/eval_policy.py --controller expert --episodes 8 --out runs/check_eval
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# This host exposes a full GPU 0 alongside MIG devices. Restricting visibility to
# physical GPU 0 keeps its EGL index unchanged and prevents PyTorch miscounting
# mixed full/MIG devices when no visibility variable was inherited from a shell.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for var in ("MAGNUM_LOG", "HABITAT_SIM_LOG"):
    os.environ.setdefault(var, "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

import habitat
import numpy as np
from habitat.core.registry import registry
from habitat.datasets import make_dataset
from PIL import Image

from sim.config import load_expert, load_manifest, make_ce_config, scene_group
from sim.episodes import select_episodes


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def aggregate(rows):
    """Pooled and per-dataset metrics; checkpoint selection uses dataset macro SR/SPL."""
    def summary(items):
        return {"episodes": len(items), "scenes": len({r["scene"] for r in items}),
                **{key: float(np.mean([r[key] for r in items])) for key in
                   ("success", "spl", "distance_to_goal", "oracle_success", "steps", "collisions", "stopped")}}
    by_dataset = {name: summary([r for r in rows if r["dataset"] == name])
                  for name in sorted({r["dataset"] for r in rows})}
    return {"overall": summary(rows), "by_dataset": by_dataset,
            "macro_success": float(np.mean([v["success"] for v in by_dataset.values()])),
            "macro_spl": float(np.mean([v["spl"] for v in by_dataset.values()]))}


def parallel_evaluation(args):
    """Independent batch-one workers; merge only complete, disjoint episode sets."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    signature = {"args": vars(args), "evaluator_sha256": sha256(__file__)}
    if args.checkpoint:
        signature["adapter_sha256"] = sha256(Path(args.checkpoint) / "adapter_model.safetensors")
    manifest_path = out / "parallel_config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise ValueError("Parallel evaluation settings changed; use a new output directory")
    atomic_json(manifest_path, signature)
    if (out / "summary.json").exists():
        print((out / "summary.json").read_text(), flush=True)
        return
    children, logs = [], []
    started = time.time()
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Evaluation workers interrupted for handoff")
    previous_signal = signal.signal(signal.SIGTERM, interrupt)
    try:
        for rank in range(args.workers):
            worker_out = out / "workers" / str(rank)
            worker_out.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--out", str(worker_out), "--controller", args.controller,
                       "--datasets", *args.datasets, "--episodes", str(args.episodes),
                       "--seed", str(args.seed), "--gpu", str(args.gpu),
                       "--workers", str(args.workers), "--worker-index", str(rank)]
            if args.run_dir:
                command += ["--run-dir", args.run_dir]
            if args.checkpoint:
                command += ["--checkpoint", args.checkpoint]
            log = open(worker_out / "stdout.log", "a")
            logs.append(log)
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        last_done = -1
        while True:
            codes = [c.poll() for c in children]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f"Evaluation worker failed: exit codes {codes}; see {out}/workers/*/stdout.log")
            rows = [json.loads(p.read_text()) for p in sorted(out.glob("workers/*/episodes/*.json"))]
            if rows and len(rows) != last_done:
                ids = [(r["dataset"], r["episode_id"]) for r in rows]
                assert len(ids) == len(set(ids)), "Duplicate episodes across workers"
                progress = {"completed": len(rows), "metrics": aggregate(rows)}
                atomic_json(out / "progress.json", progress)
                print(json.dumps({"completed": len(rows), "macro_success": progress["metrics"]["macro_success"]}), flush=True)
                last_done = len(rows)
            if all(c is not None for c in codes):
                break
            time.sleep(5)

        configs = [json.loads((out / "workers" / str(rank) / "eval_config.json").read_text())
                   for rank in range(args.workers)]
        common = {k: v for k, v in configs[0].items() if k not in ("selected_episodes", "worker_index")}
        for config in configs[1:]:
            assert {k: v for k, v in config.items() if k not in ("selected_episodes", "worker_index")} == common
        chosen = {}
        for name in args.datasets:
            subsets = [c["selected_episodes"][name] for c in configs]
            chosen[name] = [pool[i] for i in range(max(map(len, subsets))) for pool in subsets if i < len(pool)]
        expected = {(name, eid) for name, ids in chosen.items() for eid in ids}
        assert {(r["dataset"], r["episode_id"]) for r in rows} == expected
        assert len(rows) == sum(map(len, chosen.values()))
        (out / "episodes").mkdir(exist_ok=True)
        for source in out.glob("workers/*/episodes/*.json"):
            target = out / "episodes" / source.name
            shutil.copyfile(source, target.with_suffix(".json.tmp"))
            target.with_suffix(".json.tmp").replace(target)
        atomic_json(out / "eval_config.json", {**common, "worker_index": None, "selected_episodes": chosen})
        report = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "seconds_this_run": time.time() - started, "group": "C", "development_only": True,
                  "selection": "scene-balanced seeded subset, unfiltered by expert success",
                  "rxr_protocol": "15-degree turns; differs from official 30-degree RxR-CE",
                  "workers": args.workers, "decoding_batch_size": 1, **aggregate(rows)}
        atomic_json(out / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--controller", choices=["policy", "expert", "stop"], default="policy")
    ap.add_argument("--datasets", nargs="+", choices=["r2r", "rxr"], default=["r2r", "rxr"])
    ap.add_argument("--episodes", type=int, default=64, help="per dataset, scene-balanced; 0 means all C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0, choices=[0], help="physical Habitat EGL device 0 on this host")
    ap.add_argument("--workers", type=int, default=4, help="independent batch-one simulator workers")
    ap.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.controller == "policy" and (not args.run_dir or not args.checkpoint):
        ap.error("policy evaluation requires --run-dir and --checkpoint")
    if args.episodes < 0:
        ap.error("--episodes must be nonnegative")
    if args.workers < 1 or (args.worker_index is not None and not 0 <= args.worker_index < args.workers):
        ap.error("invalid worker count/index")
    if args.episodes and args.workers > args.episodes:
        ap.error("workers must not exceed episodes per dataset")
    if args.workers > 1 and args.worker_index is None:
        return parallel_evaluation(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    manifest = load_manifest()
    configs, chosen, experts = {}, {}, {}
    for name in args.datasets:
        cfg = make_ce_config(name, "train", scenes=scene_group("C"), gpu=args.gpu)
        dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        chosen[name] = select_episodes(dataset.episodes, args.episodes, args.seed)
        if args.worker_index is not None:
            chosen[name] = chosen[name][args.worker_index::args.workers]
        assert all(ep.scene_id.split("/")[-2] in scene_group("C") for ep in chosen[name])
        configs[name] = cfg
        if args.controller == "expert":
            experts[name] = load_expert(name, "train")

    signature = {"controller": args.controller, "group": "C", "split": "train", "seed": args.seed,
                 "workers": args.workers, "worker_index": args.worker_index,
                 "selected_episodes": {n: [str(e.episode_id) for e in eps] for n, eps in chosen.items()},
                 "simulator": manifest["simulator"],
                 "evaluator_sha256": sha256(__file__),
                 "partition_sha256": sha256(ROOT / "manifest/scene_partition.json"),
                 "selection_sha256": sha256(ROOT / "sim/episodes.py"),
                 "decoding_batch_size": 1, "rgb_source": "live simulator RGB, no JPEG compression"}
    if args.controller == "policy":
        signature.update(checkpoint=str(Path(args.checkpoint).resolve()),
                         adapter_sha256=sha256(Path(args.checkpoint) / "adapter_model.safetensors"),
                         policy_config=json.loads((Path(args.run_dir) / "policy_config.json").read_text()),
                         policy_source_sha256=sha256(ROOT / "policy/smolvlm2.py"),
                         online_source_sha256=sha256(ROOT / "policy/online.py"),
                         pixel_preprocessing="cached native pixel tensors; full native vision forward each step")
    config_path = out / "eval_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != signature:
            raise ValueError("Output contains a different evaluation; choose a new --out directory")
    else:
        atomic_json(config_path, signature)

    if args.controller == "policy":
        import torch
        from policy.common import PolicyConfig
        from policy.online import OnlineInputs
        from policy.smolvlm2 import SmolVLM2Inputs, SmolVLM2Policy
        torch.manual_seed(args.seed)
        pcfg = PolicyConfig.load(Path(args.run_dir) / "policy_config.json")
        inputs = SmolVLM2Inputs(pcfg)
        policy = SmolVLM2Policy(pcfg, adapter=args.checkpoint, device=f"cuda:{args.gpu}")
        policy.model.eval()

    rows = []
    started = time.time()
    for name in args.datasets:
        pending = []
        for ep in chosen[name]:
            record_path = out / "episodes" / f"{name}_{ep.episode_id}.json"
            if record_path.exists():
                rows.append(json.loads(record_path.read_text()))
            else:
                pending.append(ep)
        if not pending:
            continue
        # Selection is fixed above; grouping minimizes expensive scene loads.
        dataset = registry.get_dataset(configs[name].habitat.dataset.type)(config=None)
        dataset.episodes = sorted(pending, key=lambda e: (e.scene_id, str(e.episode_id)))
        env = habitat.Env(config=configs[name], dataset=dataset)
        try:
            expected_actions = ["stop", "move_forward", "turn_left", "turn_right"]
            assert list(configs[name].habitat.task.actions.keys()) == expected_actions
            for _ in pending:
                obs = env.reset()
                ep = env.current_episode
                eid = str(ep.episode_id)
                begin = time.time()
                images, actions, collisions = [], [], []
                online = OnlineInputs(inputs, policy.device) if args.controller == "policy" else None
                distances = [float(env.get_metrics()["distance_to_goal"])]
                positions = [env.sim.get_agent_state().position.tolist()]
                budget = manifest["simulator"]["max_episode_steps"][name]
                while not env.episode_over:
                    t = len(actions)
                    if args.controller == "expert":
                        reference = experts[name][eid]["actions"]
                        if t >= len(reference):
                            raise RuntimeError("Expert actions ended before STOP/episode termination")
                        action = int(reference[t])
                    elif args.controller == "stop":
                        action = 0
                    else:
                        online.append(Image.fromarray(obs["rgb"]))
                        batch = online(ep.instruction.instruction_text, actions)
                        action = int(policy.action_scores(batch).argmax(1).item())
                    assert 0 <= action <= 3 and t < budget
                    obs = env.step(action)
                    actions.append(action)
                    collisions.append(bool(env.sim.previous_step_collided))
                    distances.append(float(env.get_metrics()["distance_to_goal"]))
                    positions.append(env.sim.get_agent_state().position.tolist())
                    if len(actions) % 100 == 0:
                        print(json.dumps({"dataset": name, "episode_id": eid, "step": len(actions),
                                          "elapsed_seconds": time.time() - begin}), flush=True)
                metrics = env.get_metrics()
                row = {"dataset": name, "episode_id": eid, "scene": ep.scene_id.split("/")[-2],
                       "instruction": ep.instruction.instruction_text,
                       "success": float(metrics["success"]), "spl": float(metrics["spl"]),
                       "distance_to_goal": float(metrics["distance_to_goal"]),
                       "oracle_success": float(min(distances) < manifest["simulator"]["success_distance_m"]),
                       "steps": len(actions), "collisions": sum(collisions), "stopped": actions[-1] == 0,
                       "termination": "STOP" if actions[-1] == 0 else "step_limit",
                       "executed_actions": actions, "collided_after_action": collisions,
                       "distances_to_goal": distances, "positions": positions,
                       "seconds": time.time() - begin}
                atomic_json(out / "episodes" / f"{name}_{eid}.json", row)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("dataset", "episode_id", "success", "spl", "steps", "seconds")}), flush=True)
                atomic_json(out / "progress.json", {"completed": len(rows), "metrics": aggregate(rows)})
        finally:
            env.close()
    expected = sum(len(eps) for eps in chosen.values())
    assert len(rows) == expected and len({(r["dataset"], r["episode_id"]) for r in rows}) == expected
    report = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "seconds_this_run": time.time() - started, "group": "C", "development_only": True,
              "selection": "scene-balanced seeded subset, unfiltered by expert success",
              "rxr_protocol": "15-degree turns; differs from official 30-degree RxR-CE",
              **aggregate(rows)}
    atomic_json(out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
