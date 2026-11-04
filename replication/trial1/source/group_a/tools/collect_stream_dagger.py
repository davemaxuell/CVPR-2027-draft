"""Collect one StreamVLN DAgger round on group A: policy rollouts with expert takeovers (env rolevln).

stream_policy/controller.py states the rule. Every episode's record is written; its frames are kept
only if the rule keeps the episode (or with --keep-all-frames). Episode folders appear atomically,
so an interrupted round resumes; it is complete when summary.json exists. The layout matches the
expert renders, so tools/train_stream_policy.py reads a round directly as another --train-root:
<out>/<dataset>/<episode_id>/{meta.json, rgb/0000.jpg ...} with data_kind "streamvln_dagger".
Episodes are drawn, seeded and scene-balanced, from those with a validated group-A expert render
(as in trial_0); the render's length is the reference action count of the error-ratio rule.

    python tools/collect_stream_dagger.py --run-dir RUN --weights RUN/checkpoints/step_N --episodes 64 --workers 4 --out OUT
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import replace
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
from stream_policy.config import StreamConfig  # noqa: E402
from stream_policy.controller import DaggerRule  # noqa: E402
from stream_policy.rollout import python_command, run_episode, run_workers  # noqa: E402
from tools.eval_policy import atomic_json, sha256  # noqa: E402

EXPERT_ROOT = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupA")


def expert_lengths(name):
    rows = [json.loads(line) for line in (EXPERT_ROOT / f"{name}_index.jsonl").read_text().splitlines()]
    return {str(r["episode_id"]): r["num_frames"] for r in rows
            if r["success"] == 1 and not r["truncated"] and r["max_location_deviation_m"] < 0.5}


def chosen_episodes(args, name):
    cfg = make_ce_config(name, "train", scenes=scene_group("A"), gpu=args.gpu)
    dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    lengths = expert_lengths(name)
    episodes = select_episodes([ep for ep in dataset.episodes if str(ep.episode_id) in lengths], args.episodes, args.seed)
    return cfg, episodes, lengths


def rule_from(args):
    return replace(DaggerRule(), **json.loads(args.rule_overrides)) if args.rule_overrides else DaggerRule()


def worker(args, out, signature_sha):
    stream = StreamConfig.load(Path(args.run_dir) / "stream_config.json") if args.run_dir else StreamConfig()
    rule, success_distance = rule_from(args), load_manifest()["simulator"]["success_distance_m"]
    with open(out / "workers" / f"{args.worker_index}.server.log", "a") as server_log, \
            PolicyClient(args.run_dir, args.weights, args.policy_gpu, args.stop_bias, log=server_log) as client:
        atomic_json(out / "workers" / f"{args.worker_index}.identity.json", client.identity)
        for name in args.datasets:
            cfg, episodes, lengths = chosen_episodes(args, name)
            pending = [ep for ep in episodes[args.worker_index::args.workers]
                       if not (out / name / str(ep.episode_id) / "meta.json").exists()]
            if not pending:
                continue
            dataset = registry.get_dataset(cfg.habitat.dataset.type)(config=None)
            dataset.episodes = sorted(pending, key=lambda e: (e.scene_id, str(e.episode_id)))
            env = habitat.Env(config=cfg, dataset=dataset)
            dagger = {"rule": rule, "actions_per_turn": stream.actions_per_turn, "beta": args.beta,
                      "seed": f"{args.seed}:{name}", "reference_lengths": lengths}
            try:
                for _ in pending:
                    temp = out / name / f".tmp_{args.worker_index}"
                    if temp.exists():
                        shutil.rmtree(temp)
                    (temp / "rgb").mkdir(parents=True)
                    row = run_episode(env, client, success_distance, dagger=dagger, frames=temp / "rgb",
                                      jpeg_quality=args.jpeg_quality, log=lambda s: print(s, flush=True))
                    eid = row["episode_id"]
                    if not (row["keep"] or args.keep_all_frames):
                        shutil.rmtree(temp / "rgb")
                    meta = {"data_kind": "streamvln_dagger", "dataset": name, "split": "train", "group": "A", **row,
                            "beta": args.beta, "frames_saved": row["keep"] or args.keep_all_frames,
                            "collection_signature_sha256": signature_sha}
                    atomic_json(temp / "meta.json", meta)
                    final = out / name / eid
                    if final.exists():
                        raise RuntimeError(f"unexpected existing episode folder {final}")
                    temp.rename(final)
                    print(json.dumps({"dataset": name, "episode_id": eid, "steps": row["num_frames"], "keep": row["keep"],
                                      "takeovers": len(row["takeover_steps"]), "success": row["metrics"]["success"],
                                      "seconds": round(row["seconds"], 1)}), flush=True)
            finally:
                env.close()


def summarize(out, datasets):
    rows = [json.loads(p.read_text()) for name in datasets for p in sorted((out / name).glob("*/meta.json"))
            if not p.parent.name.startswith(".")]
    sources, kept = Counter(), [r for r in rows if r["keep"]]
    for r in rows:
        sources.update(r["action_source"])
    return {"completed_episodes": len(rows), "kept_episodes": len(kept), "kept_steps": sum(r["num_frames"] for r in kept),
            "actions": sum(r["num_frames"] for r in rows), "action_sources": dict(sources),
            "episodes_with_takeover": sum(bool(r["takeover_steps"]) for r in rows),
            "finished_at_goal": sum(r["finished"] for r in rows), "aborted": sum(r["aborted"] is not None for r in rows),
            "by_dataset": {name: {"episodes": sum(r["dataset"] == name for r in rows),
                                  "kept": sum(r["dataset"] == name and r["keep"] for r in rows),
                                  "finished_without_takeover": sum(r["dataset"] == name and r["finished"] and not r["takeover_steps"]
                                                                   for r in rows)}
                           for name in datasets}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="", help="training run of the collecting policy")
    ap.add_argument("--weights", default="", help="checkpoint of the collecting policy; empty = pretrained backbone")
    ap.add_argument("--stop-bias", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", choices=["r2r", "rxr"], default=["r2r", "rxr"])
    ap.add_argument("--episodes", type=int, default=64, help="per dataset; 0 = every validated group-A episode")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.0, help="probability that the expert drives a turn (StreamVLN: p**round, p=0)")
    ap.add_argument("--rule-overrides", default="", help="JSON of DaggerRule fields to change, recorded in the signature")
    ap.add_argument("--keep-all-frames", action="store_true", help="also keep frames of episodes the rule rejects")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--gpu", type=int, default=0, choices=[0], help="Habitat EGL device (only 0 works on this host)")
    ap.add_argument("--policy-gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.workers < 1 or not 0 <= args.beta <= 1:
        ap.error("need workers >= 1 and 0 <= beta <= 1")
    out = Path(args.out).resolve()
    args.out = str(out)
    out.mkdir(parents=True, exist_ok=True)
    selected = {name: [str(e.episode_id) for e in chosen_episodes(args, name)[1]] for name in args.datasets}
    weights = Path(args.weights) if args.weights else None
    signature = {"group": "A", "split": "train", "datasets": args.datasets, "seed": args.seed, "beta": args.beta,
                 "rule": rule_from(args).as_dict(), "controller": "StreamVLN DAgger (stream_policy/controller.py)",
                 "selected_episodes": selected, "episodes_per_dataset": args.episodes,
                 "run_dir": str(Path(args.run_dir).resolve()) if args.run_dir else None,
                 "weights": str(weights.resolve()) if weights else None,
                 "weight_sha256": {p.name: sha256(p) for p in sorted(weights.glob("*.safetensors"))} if weights else {},
                 "stop_bias": args.stop_bias, "keep_all_frames": args.keep_all_frames, "jpeg_quality": args.jpeg_quality,
                 "frames": "frame t is the RGB view before action t; T actions, T frames",
                 "expert_pool_sha256": {n: sha256(EXPERT_ROOT / f"{n}_index.jsonl") for n in args.datasets},
                 "simulator": load_manifest()["simulator"],
                 "source_sha256": {n: sha256(ROOT / n) for n in ("tools/collect_stream_dagger.py", "stream_policy/controller.py",
                                                                  "stream_policy/expert.py", "stream_policy/rollout.py",
                                                                  "stream_policy/server.py", "stream_policy/layout.py",
                                                                  "stream_policy/qwen35.py", "manifest/scene_partition.json")}}
    config = out / "collection_config.json"
    if args.worker_index is None:
        if config.exists() and json.loads(config.read_text()) != signature:
            raise ValueError("collection settings changed; choose a new --out")
        atomic_json(config, signature)
    elif json.loads(config.read_text()) != signature:
        raise ValueError("worker settings differ from the parent's")
    if args.worker_index is not None:
        return worker(args, out, sha256(config))
    if (out / "summary.json").exists():
        print((out / "summary.json").read_text())
        return
    started, seen = time.time(), [-1]

    def progress():
        report = summarize(out, args.datasets)
        if report["completed_episodes"] != seen[0]:
            seen[0] = report["completed_episodes"]
            atomic_json(out / "progress.json", {**report, "seconds_this_run": time.time() - started})
            print(json.dumps(report), flush=True)

    run_workers(python_command(Path(__file__).resolve(), args), args.workers, out, progress)
    report = summarize(out, args.datasets)
    done = {(n, p.parent.name) for n in args.datasets for p in (out / n).glob("*/meta.json") if not p.parent.name.startswith(".")}
    if done != {(n, e) for n, ids in selected.items() for e in ids}:
        raise RuntimeError("completed episodes differ from the selection")
    for name in args.datasets:
        rows = [json.loads((out / name / e / "meta.json").read_text()) for e in sorted(selected[name])]
        (out / f"{name}_index.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("episode_id", "scene", "num_frames", "keep")}
                                                                    | {"success": r["metrics"]["success"], "takeovers": len(r["takeover_steps"])}) + "\n"
                                                         for r in rows))
    report.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), seconds_this_run=time.time() - started,
                  policy_identity=json.loads((out / "workers" / "0.identity.json").read_text()))
    atomic_json(out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
