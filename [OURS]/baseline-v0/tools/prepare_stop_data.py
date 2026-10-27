"""Versioned training views: consistent 2m labels, pose-aware caps, unchanged RGB/history."""
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.goal_distance import DATA, GoalDistances


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evenly(values, cap):
    values = list(values)
    if len(values) <= cap:
        return values
    return [values[i] for i in np.linspace(0, len(values) - 1, cap).round().astype(int)]


def pose_retained(meta, targets, cap=4):
    buckets = defaultdict(list)
    for t, (position, rotation, target) in enumerate(zip(meta["position"], meta["rotation_wxyz"], targets)):
        q = np.asarray(rotation, dtype=float)
        nonzero = np.flatnonzero(abs(q) > 1e-8)
        if len(nonzero) and q[nonzero[0]] < 0:
            q = -q
        key = (tuple(np.round(position, 3)), tuple(np.round(q, 4)), int(target))
        buckets[key].append(t)
    return sorted(t for values in buckets.values() for t in evenly(values, cap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["expert", "policy"], required=True)
    ap.add_argument("--group", choices=["A", "C"], default="A")
    args = ap.parse_args()
    source, out = Path(args.root).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print((out / "summary.json").read_text(), flush=True)
        return
    allowed = set(json.loads((ROOT / "manifest/scene_partition.json").read_text())["groups"][args.group]["scenes"])
    settings = {"source": str(source), "kind": args.kind, "group": args.group, "stop_radius_m": 2.0,
                "hard_negative_distance_m": [3.0, 5.0], "max_stop_per_episode": 2 if args.kind == "expert" else 8,
                "max_hard_negative_per_episode": 16, "max_repeats_same_pose_rotation_label": 4,
                "position_quantization_m": .001, "quaternion_quantization": .0001,
                "history": "Original frame indices and executed actions; labels only are overridden",
                "original_STOP_outside_2m": "excluded: no known movement correction",
                "goal_sources_sha256": {}, "source_code_sha256": {name: sha(ROOT / name) for name in
                    ("tools/prepare_stop_data.py", "sim/goal_distance.py", "manifest/scene_partition.json")}}
    tasks, goals = [], {}
    for name, filename in (("r2r", "R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"),
                           ("rxr", "RxR_VLNCE_v0/train/train_guide.json.gz")):
        dataset_path = DATA / "datasets" / filename
        settings["goal_sources_sha256"][name] = sha(dataset_path)
        with gzip.open(dataset_path, "rt") as stream:
            rows = json.load(stream)["episodes"]
        goals[name] = {str(r["episode_id"]): r["goals"][0]["position"] for r in rows}
        if args.kind == "expert":
            for line in (source / f"{name}_index.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row["success"] == 1 and not row["truncated"] and row["max_location_deviation_m"] < .5:
                    tasks.append((row["scene"], name, str(row["episode_id"]), source / name / str(row["episode_id"]) / "meta.json"))
        else:
            assert json.loads((source / "summary.json").read_text())["status"] == "complete"
            for path in (source / name).glob("*/meta.json"):
                row = json.loads(path.read_text())
                tasks.append((row["scene"], name, str(row["episode_id"]), path))
    if args.kind == "policy":
        settings["collection_config_sha256"] = sha(source / "collection_config.json")
    if (out / "settings.json").exists():
        assert json.loads((out / "settings.json").read_text()) == settings
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    tasks.sort()
    solver, count, totals = GoalDistances(), 0, Counter()
    labels_all, distances_all = [], []
    temporary = out / "episodes.jsonl.tmp"
    with open(temporary, "w") as stream:
        for scene, name, eid, path in tasks:
            assert scene in allowed, (scene, args.group)
            meta = json.loads(path.read_text())
            executed, original = meta["executed_action"], meta["expert_action"]
            d = solver.distances(scene, meta["position"], goals[name][eid])
            assert len(d) == len(original) == len(executed)
            if args.kind == "expert":
                assert abs(float(d[-1]) - meta["metrics"]["distance_to_goal"]) < 1e-3, (name, eid, float(d[-1]), meta["metrics"])
            else:
                assert np.allclose(d, meta["distances_to_goal"][:-1], atol=1e-3, rtol=0), (name, eid)
            targets = np.array([-1 if a is None else a for a in original], dtype=np.int8)
            targets[~np.isfinite(d)] = -1
            targets[(targets == 0) & (d >= 2)] = -1
            targets[d < 2] = 0
            retained = list(range(len(d))) if args.kind == "expert" else pose_retained(meta, targets)
            stop = evenly([t for t in retained if targets[t] == 0], settings["max_stop_per_episode"])
            hard = evenly([t for t in retained if targets[t] > 0 and (3 <= d[t] <= 5 or (executed[t] == 0 and d[t] >= 3))], 16)
            navigation = [t for t in retained if targets[t] > 0]
            assert all(d[t] < 2 for t in stop)
            assert all(d[t] >= 2 for t in hard + navigation)
            entry = {"meta": str(path), "meta_sha256": sha(path), "dataset": name, "episode_id": eid, "scene": scene,
                     "kind": args.kind, "offset": count, "length": len(d), "stop": stop, "hard": hard, "navigation": navigation}
            stream.write(json.dumps(entry) + "\n")
            count += len(d)
            labels_all.append(targets)
            distances_all.append(d)
            totals.update(episodes=1, original_states=len(d), retained_pose_states=len(retained),
                          stop_examples=len(stop), hard_nonstop_examples=len(hard), navigation_examples=len(navigation),
                          invalid_labels=int((targets < 0).sum()), stop_episodes=int(bool(stop)))
            if totals["episodes"] % 1000 == 0:
                print(json.dumps(dict(totals)), flush=True)
    np.save(out / "targets.npy", np.concatenate(labels_all))
    np.save(out / "goal_distances.npy", np.concatenate(distances_all))
    temporary.replace(out / "episodes.jsonl")
    report = {"status": "complete", "kind": args.kind, "group": args.group, **totals,
              "all_goal_distances_checked_against_recorded_simulator_endpoints": True,
              "navmesh_sha256": solver.mesh_hashes,
              "artifacts_sha256": {name: sha(out / name) for name in ("settings.json", "episodes.jsonl", "targets.npy", "goal_distances.npy")}}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
