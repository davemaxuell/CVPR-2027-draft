"""Audit the data a streaming-policy run reads, before launching it (env rolevln-qwen35; CPU only).

Per render root and dataset:
  1. Completeness: for a whole-group render (expert_groupA), the episode folders equal the VLN-CE train
     episodes of that scene group (manifest/scene_partition.json; RxR guide, English); for a subset
     render (expert_groupC_600), every folder is such an episode and the count matches the plan.
  2. Each episode: scene group, split, language, instruction text and start position equal the source
     episode; T actions and T frames; STOP last and only last; replayed actions equal the expert's;
     the index row agrees with meta.json; the usable filter matches the trainer's.
  3. Every frame some window shows (views at turn starts, memory views) decodes to 640x480 RGB; frames
     with almost no contrast are counted, and pure-black ones (Habitat's background through a hole in the
     Matterport mesh: deterministic per pose, so they are kept like any other observation) are listed.
Then the windows exactly as tools/train_stream_policy.py builds them: counts, target actions, supervised
tokens and prompt lengths against max_length (longest conjunction, so an upper bound), the one-pass
sampler (no window twice) and DataLoader throughput with the trainer's workers and augmentation.

With --dagger-root, each StreamVLN DAgger round (tools/collect_stream_dagger.py) is audited as a further
training root: the round is complete, every episode folder matches its index row and is a group-A train
episode written under this round's signature, records are frame-aligned, kept episodes have all their
frames, every frame a window shows decodes, and the trainer's windows over the round are counted.
Rejected episodes aborted before an action may also record the final unexecuted observation; its
null expert label and final pose are checked explicitly and it never contributes training windows.

    python tools/audit_stream_data.py --out trials/trial_1/steps/01_experiment/data_audit.json
    python tools/audit_stream_data.py --dagger-root <round> --out <round>/data_audit.json
"""
import argparse
import gzip
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stream_policy.config import MANIFEST, StreamConfig, simulator_settings  # noqa: E402
from stream_policy.data import MIN_ACTIONS  # noqa: E402
from stream_policy.layout import layout, memory_indices, windows  # noqa: E402

DATA = Path("/data/team_a/dave-workspace/RoleVLN/data")
RENDERS = DATA / "processed/baseline_v0"
EPISODES = {"r2r": "datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz",
            "rxr": "datasets/RxR_VLNCE_v0/{split}/{split}_{role}.json.gz"}
STOP = 0


def source_episodes(name, group_scenes):
    import yaml
    manifest = yaml.safe_load(MANIFEST.read_text())["datasets"]["rxr_ce"]
    roles, languages = (manifest["roles"], set(manifest["languages"])) if name == "rxr" else ([None], None)
    found = {}
    for role in roles:
        with gzip.open(DATA / EPISODES[name].format(split="train", role=role), "rt") as f:
            for ep in json.load(f)["episodes"]:
                if languages is not None and ep["instruction"].get("language") not in languages:
                    continue
                if ep["scene_id"].split("/")[-2] in group_scenes:
                    found[str(ep["episode_id"])] = ep
    return found, languages


def used_frames(num_frames, cfg):
    frames = set()
    for start, stop in windows(num_frames, cfg):
        frames.update(memory_indices(start, cfg))
        frames.update(range(start, stop, cfg.actions_per_turn))
    return sorted(frames)


def decode(job):
    """Decode the listed frames of one episode: (count, failures, low-contrast count, mean brightness, black frames)."""
    folder, frames, size = job
    failures, blank, means, black = [], 0, [], []
    for t in frames:
        try:
            with Image.open(os.path.join(folder, "rgb", f"{t:04d}.jpg")) as image:
                image = image.convert("RGB")
                if image.size != size:
                    failures.append((t, f"size {image.size}"))
                    continue
                small = np.asarray(image.reduce(8), dtype=np.float32)
        except Exception as error:  # noqa: BLE001 (any decode failure is a finding)
            failures.append((t, repr(error)))
            continue
        blank += float(small.std()) < 2.0
        means.append(float(small.mean()))
        if float(small.max()) < 1.0:  # every pixel 0: the renderer's background, seen through a mesh hole
            black.append((folder, t))
    return len(frames), [(folder, *f) for f in failures], blank, means, black


def audit_root(root, group, cfg, sim, whole_group, max_deviation, pool):
    partition = json.loads((ROOT / "manifest/scene_partition.json").read_text())["groups"][group]["scenes"]
    planned = json.loads((root / "render_manifest.json").read_text())["planned"]
    report, jobs = {}, []
    for name in ("r2r", "rxr"):
        source, languages = source_episodes(name, set(partition))
        folders = {p.name for p in (root / name).iterdir() if p.is_dir()}
        hidden = sorted(f for f in folders if f.startswith("."))
        folders -= set(hidden)
        index = {str(r["episode_id"]): r for r in map(json.loads, (root / f"{name}_index.jsonl").read_text().splitlines())}
        problems, excluded_problems, counts, usable, steps = Counter(), Counter(), Counter(), [], 0
        for eid in sorted(folders):
            m = json.loads((root / name / eid / "meta.json").read_text())
            ep, executed, expert = source.get(eid), m["executed_action"], m["expert_action"]
            T = m["num_frames"]
            checks = {
                "not_a_group_episode": ep is None,
                "scene_or_group": m["scene"] not in partition or m["group"] != group or m["split"] != "train",
                "language": languages is not None and m["instruction"]["language"] not in languages,
                "instruction_differs": ep is not None and m["instruction"]["text"] != ep["instruction"]["instruction_text"],
                "start_position_differs": ep is not None and float(np.abs(np.subtract(m["position"][0], ep["start_position"])).max()) > 1e-3,
                "length": not (len(executed) == len(expert) == T == len(m["position"])),
                "stop_not_last_only": executed[-1] != STOP or STOP in executed[:-1],
                "action_ids": any(a not in (0, 1, 2, 3) for a in executed),
                "replay_differs_from_expert": executed != expert,
                "index_row": eid not in index or index[eid]["num_frames"] != T or bool(index[eid]["truncated"]) != bool(m["truncated"]),
                "frame_count": len(os.listdir(root / name / eid / "rgb")) != T,
            }
            failed, drift, short = (m["metrics"]["success"] < 1 or m["truncated"], m["max_location_deviation_m"] >= max_deviation,
                                    T < MIN_ACTIONS)  # the trainer's filter, in the trainer's order
            ok = not (failed or drift or short)
            (problems if ok else excluded_problems).update(k for k, bad in checks.items() if bad)
            counts["failed_or_truncated" if failed else "drift" if drift else "shorter_than_4" if short else "usable"] += 1
            if ok:
                usable.append(eid)
                steps += T
                jobs.append((str(root / name / eid), used_frames(T, cfg), (sim["rgb"]["width"], sim["rgb"]["height"])))
        expected = set(source) if whole_group else None
        report[name] = {"source_group_episodes": len(source), "planned": planned[name]["episodes"], "rendered": len(folders),
                        "leftover_temporary_folders": hidden, "index_rows": len(index),
                        "missing_from_render": sorted(expected - folders)[:20] if expected else None,
                        "missing_count": len(expected - folders) if expected else None,
                        "not_in_group_source": sorted(folders - set(source))[:20],
                        "episode_problems": dict(problems), "problems_in_excluded_episodes": dict(excluded_problems),
                        "filter": dict(counts), "usable_steps": steps,
                        "complete": (folders == expected) if expected else (folders <= set(source) and len(folders) == planned[name]["episodes"])}
    decoded = list(pool.map(decode, jobs, chunksize=8))
    means = np.concatenate([np.asarray(d[3]) for d in decoded if d[3]]) if decoded else np.zeros(1)
    black = [b for d in decoded for b in d[4]]
    frames = {"decoded": sum(d[0] for d in decoded), "failures": sum((d[1] for d in decoded), [])[:20],
              "failure_count": sum(len(d[1]) for d in decoded), "low_contrast": sum(d[2] for d in decoded),
              "black": len(black), "episodes_with_black": len({f for f, _ in black}),
              "black_examples": [f"{Path(f).parent.name}/{Path(f).name}@{t}" for f, t in black[:10]],
              "mean_brightness": {"min": float(means.min()), "p01": float(np.percentile(means, 1)),
                                  "median": float(np.median(means)), "max": float(means.max())}}
    return report, frames


def dagger_record_alignment(m):
    """Return (valid, has_unexecuted_final_observation) for the collector's two record shapes.

    run_episode records the pose and expert label before checking for an unavailable label. On that
    abort, no action is executed: the rejected episode has one final pose and a null label beyond T.
    Kept trajectories must still have exactly T observations and T actions.
    """
    T = m["num_frames"]
    actions_aligned = (all(len(m[k]) == T for k in
                          ("executed_action", "action_source", "policy_scores", "collided_after_action"))
                       and len(m["distances_to_goal"]) == T + 1 and m["metrics"]["steps"] == T)
    observations = ("expert_action", "position", "rotation_wxyz")
    aligned = all(len(m[k]) == T for k in observations)
    final_observation = (
        m["keep"] is False and m["finished"] is False and m["termination"] == "aborted"
        and m["aborted"] == f"expert label unavailable at step {T}"
        and all(len(m[k]) == T + 1 for k in observations)
        and m["expert_action"][-1] is None
        and m["position"][-1] == m["final_position"]
        and m["rotation_wxyz"][-1] == m["final_rotation_wxyz"])
    return actions_aligned and (aligned or final_observation), actions_aligned and final_observation


def audit_dagger(root, cfg, sim, pool):
    """One DAgger round: completeness, per-episode records, kept frames (decoded where windows look)."""
    import hashlib
    partition = set(json.loads((ROOT / "manifest/scene_partition.json").read_text())["groups"]["A"]["scenes"])
    signature = hashlib.sha256((root / "collection_config.json").read_bytes()).hexdigest()
    config = json.loads((root / "collection_config.json").read_text())
    summary = json.loads((root / "summary.json").read_text()) if (root / "summary.json").exists() else {}
    report, jobs = {"complete": summary.get("status") == "complete", "policy_weights": config["weights"]}, []
    for name in config["datasets"]:
        folders = {p.name for p in (root / name).iterdir() if p.is_dir() and not p.name.startswith(".")}
        index = {str(r["episode_id"]): r for r in map(json.loads, (root / f"{name}_index.jsonl").read_text().splitlines())} \
            if (root / f"{name}_index.jsonl").exists() else {}
        problems, counts = Counter(), Counter()
        for eid in sorted(folders):
            m = json.loads((root / name / eid / "meta.json").read_text())
            T, rgb = m["num_frames"], root / name / eid / "rgb"
            frames = len(os.listdir(rgb)) if rgb.exists() else 0
            aligned, final_observation = dagger_record_alignment(m)
            checks = {
                "kind_or_group": m["data_kind"] != "streamvln_dagger" or m["group"] != "A" or m["split"] != "train"
                                 or m["scene"] not in partition or m["dataset"] != name or str(m["episode_id"]) != eid,
                "other_signature": m["collection_signature_sha256"] != signature,
                "index_row": eid not in index or index[eid]["num_frames"] != T or index[eid]["keep"] != m["keep"],
                "length": not aligned,
                "kept_frames_missing": m["keep"] and frames != T,
            }
            problems.update(k for k, bad in checks.items() if bad)
            counts["kept" if m["keep"] else "not_kept"] += 1
            counts["aborted"] += m["aborted"] is not None
            if final_observation:
                counts["aborted_final_observation"] += 1
            if m["keep"] and frames == T:
                jobs.append((str(root / name / eid), used_frames(T, cfg), (sim["rgb"]["width"], sim["rgb"]["height"])))
        report[name] = {"episodes": len(folders), "selected": len(config["selected_episodes"][name]),
                        "all_selected_done": folders == set(config["selected_episodes"][name]),
                        "counts": dict(counts), "problems": dict(problems)}
    decoded = list(pool.map(decode, jobs, chunksize=8))
    report["kept_frames_shown_by_windows"] = {"decoded": sum(d[0] for d in decoded), "failure_count": sum(len(d[1]) for d in decoded),
                                              "failures": sum((d[1] for d in decoded), [])[:20],
                                              "black": sum(len(d[4]) for d in decoded)}
    report["passed"] = bool(report["complete"] and report["kept_frames_shown_by_windows"]["failure_count"] == 0
                            and all(report[n]["all_selected_done"] and not report[n]["problems"] for n in config["datasets"]))
    return report


def audit_windows(roots, cfg, inputs, max_deviation, label):
    from stream_policy.data import TrajectoryWindows
    data = TrajectoryWindows(roots, cfg, inputs, ("r2r", "rxr"), "executed", max_deviation)
    longest = max(cfg.conjunctions, key=len)
    lengths, supervised, memory, worst = [], [], 0, None
    for e, start, stop in data.index:
        ep = data.episodes[e]
        sample = inputs.encode(layout(cfg, ep["instruction"], start, stop, lambda turn: longest), ep["executed"], ep["targets"])
        lengths.append(len(sample["ids"]))
        supervised.append(len(sample["label_pos"]))
        memory += start > 0
        if worst is None or lengths[-1] > worst[0]:
            worst = (lengths[-1], ep["dataset"], ep["episode_id"], start)
    lengths = np.asarray(lengths)
    return data, {"set": label, "windows": len(data), "episodes": len(data.episodes), "windows_with_memory": memory,
                  "steps": sum(len(ep["executed"]) for ep in data.episodes), "summary": data.summary(),
                  "supervised_tokens": int(np.sum(supervised)),
                  "prompt_tokens": {"min": int(lengths.min()), "median": float(np.median(lengths)), "p99": float(np.percentile(lengths, 99)),
                                    "max": int(lengths.max()), "longest_window": list(worst), "max_length": cfg.max_length,
                                    "over_max_length": int((lengths > cfg.max_length).sum())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--train-root", default=str(RENDERS / "expert_groupA"))
    ap.add_argument("--val-root", default=str(RENDERS / "expert_groupC_600"))
    ap.add_argument("--max-deviation", type=float, default=0.5)
    ap.add_argument("--processes", type=int, default=48)
    ap.add_argument("--loader-windows", type=int, default=300, help="windows timed through the trainer's DataLoader")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dagger-root", nargs="*", default=[], help="StreamVLN DAgger rounds to audit as further training roots")
    args = ap.parse_args()
    began = time.time()
    sim = simulator_settings()
    cfg = StreamConfig(forward_step_m=sim["forward_step_m"], turn_angle_deg=sim["turn_angle_deg"])
    from stream_policy.qwen35 import Qwen35Inputs
    inputs = Qwen35Inputs(cfg)
    report = {"config": {k: getattr(cfg, k) for k in ("window_steps", "actions_per_turn", "memory_frames", "max_length")}}
    with ProcessPoolExecutor(args.processes) as pool:
        for key, root, group, whole in (("train", Path(args.train_root), "A", True), ("validation", Path(args.val_root), "C", False)):
            episodes, frames = audit_root(root, group, cfg, sim, whole, args.max_deviation, pool)
            data, windowed = audit_windows([str(root)], cfg, inputs, args.max_deviation, key)
            report[key] = {"root": str(root), "group": group, "render_manifest": json.loads((root / "render_manifest.json").read_text())["args"],
                           "episodes": episodes, "frames_shown_by_windows": frames, "windows": windowed}
            if key == "train":
                train = data
        for root in args.dagger_root:
            entry = audit_dagger(Path(root), cfg, sim, pool)
            entry["windows"] = audit_windows([root], cfg, inputs, args.max_deviation, f"dagger {Path(root).name}")[1]
            report.setdefault("dagger", {})[root] = entry
    from stream_policy.data import MixtureSampler, Sources
    per_update = 128
    updates = len(train) // per_update
    stream = iter(MixtureSampler([len(train)], seed=0))
    draws = [next(stream) for _ in range(updates * per_update)]
    report["one_pass_sampler"] = {"updates": updates, "windows_drawn": len(draws), "distinct_windows": len({i for _, i in draws}),
                                  "draws_in_order": all(d == k for k, (d, _) in enumerate(draws)),
                                  "never_drawn": len(train) - len({i for _, i in draws})}
    import torch
    from torch.utils.data import DataLoader
    from stream_policy.data import TrajectoryWindows
    augmented = TrajectoryWindows([args.train_root], cfg, inputs, augment=True, max_deviation=args.max_deviation)
    loader = DataLoader(Sources([augmented], 0), batch_size=1, sampler=MixtureSampler([len(augmented)], 0), num_workers=args.workers,
                        collate_fn=inputs.collate, prefetch_factor=4)
    batches = iter(loader)
    for _ in range(2 * args.workers):
        next(batches)
    t0 = time.time()
    shapes = Counter()
    for _ in range(args.loader_windows):
        b = next(batches)
        shapes[tuple(b["pixel_values"].shape)] += 1
        assert b["pixel_values"].dtype == torch.bfloat16 and torch.isfinite(b["pixel_values"].float()).all()
    report["loader"] = {"workers": args.workers, "windows": args.loader_windows, "windows_per_s": args.loader_windows / (time.time() - t0),
                        "pixel_value_shapes": {str(k): v for k, v in shapes.items()}, "gpu_consumes_windows_per_s": 1.3}
    t, v = report["train"], report["validation"]
    for part in (t, v):  # the trainer must load exactly the episodes this audit counts as usable
        part["trainer_loads_audited_episodes"] = part["windows"]["episodes"] == sum(
            part["episodes"][n]["filter"].get("usable", 0) for n in ("r2r", "rxr"))
    report["passed"] = bool(
        t["trainer_loads_audited_episodes"] and v["trainer_loads_audited_episodes"] and
        all(t["episodes"][n]["complete"] and not t["episodes"][n]["episode_problems"] for n in ("r2r", "rxr"))
        and all(v["episodes"][n]["complete"] and not v["episodes"][n]["episode_problems"] for n in ("r2r", "rxr"))
        and t["frames_shown_by_windows"]["failure_count"] == 0 and v["frames_shown_by_windows"]["failure_count"] == 0
        and t["windows"]["prompt_tokens"]["over_max_length"] == 0 and v["windows"]["prompt_tokens"]["over_max_length"] == 0
        and report["one_pass_sampler"]["distinct_windows"] == report["one_pass_sampler"]["windows_drawn"]
        and report["loader"]["windows_per_s"] > 5 * report["loader"]["gpu_consumes_windows_per_s"]
        and all(d["passed"] and d["windows"]["windows"] > 0 and d["windows"]["prompt_tokens"]["over_max_length"] == 0
                for d in report.get("dagger", {}).values()))
    report["seconds"] = time.time() - began
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
