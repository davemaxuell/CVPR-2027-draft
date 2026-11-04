"""Closed-loop evaluation of a streaming policy, resumable per episode (env rolevln; policy via its server).

The policy sees only the instruction and its actual RGB/action history. Habitat computes success,
SPL and distance; oracle success and the path-length ratio are reported separately. Episodes are the
trial_0 scene-balanced seeded subset (sim/episodes.py), unfiltered by expert success, and records
carry the fields of tools/eval_policy.py so its `aggregate` applies. Group E requires
--allow-group-e: choose settings on C first and report any reuse of E.

    python tools/eval_stream_policy.py --run-dir RUN --weights RUN/checkpoints/step_N --episodes 32 --workers 4 --out OUT
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Habitat EGL sees only physical GPU 0 on this host
for name in ("MAGNUM_LOG", "HABITAT_SIM_LOG"):
    os.environ.setdefault(name, "quiet")
os.environ.setdefault("GLOG_minloglevel", "2")

import habitat  # noqa: E402
from habitat.core.registry import registry  # noqa: E402
from habitat.datasets import make_dataset  # noqa: E402

from sim.config import load_manifest, make_ce_config, scene_group  # noqa: E402
from sim.episodes import select_episodes  # noqa: E402
from stream_policy.client import PolicyClient  # noqa: E402
from stream_policy.rollout import python_command, run_episode, run_workers  # noqa: E402
from tools.eval_policy import aggregate, atomic_json, sha256  # noqa: E402

SPLITS = {"C": "train", "E": "val_unseen"}


def chosen_episodes(args):
    chosen = {}
    for name in args.datasets:
        cfg = make_ce_config(name, SPLITS[args.group], scenes=scene_group(args.group), gpu=args.gpu)
        dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        chosen[name] = (cfg, select_episodes(dataset.episodes, args.episodes, args.seed))
    return chosen


def flat(row, dataset):
    m = row["metrics"]
    return {"dataset": dataset, **row, **{k: m[k] for k in ("success", "spl", "distance_to_goal", "oracle_success", "steps",
                                                             "collisions", "path_length_ratio")},
            "stopped": row["termination"] == "STOP"}


def worker(args, out):
    chosen = chosen_episodes(args)
    success_distance = load_manifest()["simulator"]["success_distance_m"]
    with open(out / "workers" / f"{args.worker_index}.server.log", "a") as server_log, \
            PolicyClient(args.run_dir, args.weights, args.policy_gpu, args.stop_bias, log=server_log) as client:
        atomic_json(out / "workers" / f"{args.worker_index}.identity.json", client.identity)
        for name, (cfg, episodes) in chosen.items():
            pending = [ep for ep in episodes[args.worker_index::args.workers]
                       if not (out / "episodes" / f"{name}_{ep.episode_id}.json").exists()]
            if not pending:
                continue
            dataset = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
            dataset.episodes = sorted(pending, key=lambda e: (e.scene_id, str(e.episode_id)))
            env = habitat.Env(config=cfg, dataset=dataset)
            try:
                assert list(cfg.habitat.task.actions.keys()) == ["stop", "move_forward", "turn_left", "turn_right"]
                for _ in pending:
                    row = flat(run_episode(env, client, success_distance), name)
                    atomic_json(out / "episodes" / f"{name}_{row['episode_id']}.json", row)
                    print(json.dumps({k: row[k] for k in ("dataset", "episode_id", "success", "spl", "steps", "seconds")}), flush=True)
            finally:
                env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="", help="training run (stream_config.json, processor/); empty = base model defaults")
    ap.add_argument("--weights", default="", help="checkpoint directory; empty = the pretrained backbone")
    ap.add_argument("--stop-bias", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="C", choices=sorted(SPLITS))
    ap.add_argument("--allow-group-e", action="store_true")
    ap.add_argument("--datasets", nargs="+", choices=["r2r", "rxr"], default=["r2r", "rxr"])
    ap.add_argument("--episodes", type=int, default=32, help="per dataset, scene-balanced; 0 = all of the group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0, choices=[0], help="Habitat EGL device (only 0 works on this host)")
    ap.add_argument("--policy-gpu", type=int, default=0, help="CUDA device of the policy servers")
    ap.add_argument("--workers", type=int, default=2, help="simulator workers, each with its own policy server")
    ap.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.group == "E" and not args.allow_group_e:
        ap.error("group E is the final evaluation; pass --allow-group-e once settings are frozen on C")
    if args.workers < 1 or (args.episodes and args.workers > args.episodes):
        ap.error("need 1 <= workers <= episodes per dataset")
    out = Path(args.out).resolve()
    args.out = str(out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    if args.worker_index is not None:
        return worker(args, out)

    chosen = chosen_episodes(args)
    signature = {"group": args.group, "split": SPLITS[args.group], "seed": args.seed, "datasets": args.datasets,
                 "selected_episodes": {n: [str(e.episode_id) for e in eps] for n, (_, eps) in chosen.items()},
                 "run_dir": str(Path(args.run_dir).resolve()) if args.run_dir else None,
                 "weights": str(Path(args.weights).resolve()) if args.weights else None, "stop_bias": args.stop_bias,
                 "simulator": load_manifest()["simulator"], "rgb_source": "live simulator RGB, no JPEG compression",
                 "source_sha256": {n: sha256(ROOT / n) for n in ("tools/eval_stream_policy.py", "stream_policy/rollout.py",
                                                                  "stream_policy/server.py", "stream_policy/layout.py",
                                                                  "stream_policy/qwen35.py", "manifest/scene_partition.json",
                                                                  "sim/episodes.py")}}
    config = out / "eval_config.json"
    if config.exists() and json.loads(config.read_text()) != signature:
        raise ValueError("this output holds a different evaluation; choose a new --out")
    atomic_json(config, signature)
    if (out / "summary.json").exists():
        print((out / "summary.json").read_text())
        return
    started, seen = time.time(), [-1]

    def progress():
        rows = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob("*.json"))]
        if rows and len(rows) != seen[0]:
            seen[0] = len(rows)
            atomic_json(out / "progress.json", {"completed": len(rows), "metrics": aggregate(rows)})
            print(json.dumps({"completed": len(rows), "macro_success": aggregate(rows)["macro_success"]}), flush=True)

    run_workers(python_command(Path(__file__).resolve(), args), args.workers, out, progress)
    rows = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob("*.json"))]
    expected = {(n, str(e.episode_id)) for n, (_, eps) in chosen.items() for e in eps}
    if {(r["dataset"], r["episode_id"]) for r in rows} != expected:
        raise RuntimeError("completed episodes differ from the selection")
    identities = [json.loads((out / "workers" / f"{i}.identity.json").read_text()) for i in range(args.workers)]
    if any({k: v for k, v in x.items() if k != "device"} != {k: v for k, v in identities[0].items() if k != "device"}
           for x in identities):
        raise RuntimeError("workers ran different policies")
    identity = identities[0]
    report = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "seconds_this_run": time.time() - started, "group": args.group, "development_only": args.group == "C",
              "selection": "scene-balanced seeded subset, unfiltered by expert success",
              "rxr_protocol": "15-degree turns; differs from official 30-degree RxR-CE",
              "policy_identity": identity, "workers": args.workers, **aggregate(rows),
              "mean_replans_per_episode": sum(r["replans"] for r in rows) / len(rows),
              "mean_policy_seconds_per_step": sum(r["policy_seconds"] for r in rows) / max(1, sum(r["steps"] for r in rows))}
    atomic_json(out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
