"""Index actual group-A transitions and hash every source image; resumable by episode."""
import argparse
import fcntl
import hashlib
import io
import json
import os
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def inspect_episode(job):
    source, path, scenes, signature_hashes, identity = job
    raw = path.read_bytes()
    meta = json.loads(raw)
    assert meta["group"] == "A" and meta["split"] == "train" and meta["scene"] in scenes
    assert meta["dataset"] in ("r2r", "rxr")
    actions = meta["executed_action"]
    assert actions and all(a in (0, 1, 2, 3) for a in actions) and 0 not in actions[:-1]
    n, count = len(actions), meta["num_frames"]
    assert len(meta["collided_after_action"]) == n
    if source == "expert":
        assert meta["metrics"]["success"] == 1 and not meta["truncated"] and meta["max_location_deviation_m"] < 0.5
        assert count == n and actions[-1] == 0
    elif source.startswith("dagger_"):
        assert count == n + 1
        assert meta["collection_signature_sha256"] in signature_hashes
    else:
        assert source == "selected_policy" and count == n + 1
        assert meta["policy_identity_sha256"] == identity
        assert meta["collection_config_sha256"] in signature_hashes
        values = np.asarray(meta["raw_action_scores"], dtype=np.float64)
        assert values.shape == (n, 4) and np.isfinite(values).all() and meta["stop_bias"] == -0.25
        values[:, 0] -= 0.25
        assert np.array_equal(values.argmax(1), actions)
    encoding, ext = ("PNG", "png") if source == "selected_policy" else ("JPEG", "jpg")
    expected = [f"rgb/{t:04d}.{ext}" for t in range(count)]
    assert {p.name for p in (path.parent / "rgb").iterdir()} == {Path(p).name for p in expected}
    branches = meta.get("branches", [])
    expected += [b["target_frame"] for b in branches]
    assert len(expected) == len(set(expected))
    frames = []
    decoded = 0
    for i, relative in enumerate(expected):
        data = (path.parent / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (640, 480) and image.mode == "RGB" and image.format == encoding
            image.verify()
        if i % 31 == 0 or i == len(expected) - 1:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
            decoded += 1
        if source == "selected_policy":
            assert digest == meta["file_sha256"][relative]
        frames.append((relative, digest, len(data)))
    positions = list(meta["position"])
    if len(positions) == n and "final_position" in meta:
        positions.append(meta["final_position"])
    assert len(positions) >= count
    rows = []
    for t, action in enumerate(actions):
        if action == 0:
            continue
        assert t + 1 < count, "Movement target was never recorded"
        displacement = float(np.linalg.norm(np.asarray(positions[t + 1]) - positions[t]))
        rows.append((t, action, 0, f"rgb/{t+1:04d}.{ext}", int(meta["collided_after_action"][t]),
                     int(action == 1 and displacement < 1e-5), displacement))
    branch_keys = set()
    for b in branches:
        t, action = b["t"], b["action"]
        assert t >= 3 and action in (1, 2, 3) and (t, action) not in branch_keys
        branch_keys.add((t, action))
        assert b["root_rgb_sha256"] == meta["raw_rgb_sha256"][t]
        assert b["root_pose"]["position"] == positions[t]
        if action == actions[t]:
            assert b["next_rgb_sha256"] == meta["raw_rgb_sha256"][t + 1]
            assert b["next_pose"]["position"] == positions[t + 1]
            assert b["collided"] == meta["collided_after_action"][t]
            continue  # same root/action already has its actual trajectory target
        rows.append((t, action, 1, b["target_frame"], int(b["collided"]),
                     int(action == 1 and b["displacement_m"] < 1e-5), b["displacement_m"]))
    for t in {t for t, _ in branch_keys}:
        assert {(t, a) for a in (1, 2, 3)} <= branch_keys
    return {"source": source, "path": str(path.parent), "meta_sha256": hashlib.sha256(raw).hexdigest(),
            "meta": meta, "encoding": encoding, "frames": frames, "rows": rows, "decoded": decoded}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "wm_groupA_smolvlm2_refined512_v1"))
    ap.add_argument("--visits", default=str(DATA / "wm_visits_groupA_smolvlm2_refined512_v1"))
    ap.add_argument("--reference", default=str(ROOT / "runs/step4_smolvlm2_refined512/reference"))
    ap.add_argument("--stage", choices=["base", "complete"], default="complete")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out, visits, reference = Path(args.out).resolve(), Path(args.visits).resolve(), Path(args.reference).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / "build.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "catalog.sqlite").exists() and (out / "manifest.pending.json").exists():
        assert sha(out / "catalog.sqlite") == read(out / "manifest.pending.json")["catalog_sha256"]
        (out / "manifest.pending.json").replace(out / "manifest.json")
    if (out / "manifest.json").exists():
        assert read(out / "manifest.json")["status"] == "complete"
        print((out / "manifest.json").read_text(), flush=True)
        return
    identity = read(reference / "reference.json")["policy_identity_sha256"]
    partition = ROOT / "manifest/scene_partition.json"
    scenes = set(read(partition)["groups"]["A"]["scenes"])
    config = {"format": "actual_ce_transitions_v1", "selected_policy_identity_sha256": identity,
              "reference": str(reference), "reference_manifest_sha256": sha(reference / "file_sha256.json"),
              "partition_sha256": sha(partition), "visits": str(visits),
              "matched_rule": "same root/action IDs with t>=3, DINO last 3 frames, CDiT last 4 frames",
              "early_rule": "t<3 stored separately for optional DINO-only use", "STOP_in_dynamics": False,
              "duplicate_rule": "At branched roots, use actual trajectory outcome for the executed movement; add only alternate branch actions",
              "source_sha256": {p: sha(ROOT / p) for p in ("tools/build_wm_corpus.py", "wm/data.py")}}
    if (out / "build_config.json").exists():
        assert read(out / "build_config.json") == config, "Corpus settings or builder changed"
    else:
        write(out / "build_config.json", config)
    db = sqlite3.connect(out / "catalog.sqlite.partial")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS sources(name TEXT PRIMARY KEY, provenance_json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS episodes(id INTEGER PRIMARY KEY, source TEXT, dataset TEXT, episode_id TEXT,
      scene TEXT, directory TEXT, meta_sha256 TEXT, encoding TEXT, num_actions INTEGER, num_frames INTEGER,
      decoded_frames INTEGER, UNIQUE(source,dataset,episode_id));
    CREATE TABLE IF NOT EXISTS frames(episode_pk INTEGER, relative_path TEXT, sha256 TEXT, bytes INTEGER,
      PRIMARY KEY(episode_pk,relative_path)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS transitions(id INTEGER PRIMARY KEY, episode_pk INTEGER, t INTEGER,
      action INTEGER CHECK(action IN (1,2,3)), is_branch INTEGER, target_frame TEXT, collided INTEGER,
      blocked INTEGER, displacement_m REAL, UNIQUE(episode_pk,t,action));
    """)
    roots = [("expert", DATA / "expert_groupA"), ("dagger_round1", DATA / "dagger_groupA_round1_smolvlm2"),
             ("dagger_round2", DATA / "dagger_groupA_round2_smolvlm2_stop2m")]
    if args.stage == "complete":
        assert read(visits / "summary.json")["status"] == "complete"
        assert read(visits / "summary.json")["episodes"] == 256
        roots.append(("selected_policy", visits))
    for source, directory in roots:
        signatures = set()
        if source == "expert":
            provenance_paths = [directory / "render_manifest.json"] + [directory / f"{name}_index.jsonl" for name in ("r2r", "rxr")]
            paths = []
            for name in ("r2r", "rxr"):
                for row in map(json.loads, (directory / f"{name}_index.jsonl").read_text().splitlines()):
                    if row["success"] == 1 and not row["truncated"] and row["max_location_deviation_m"] < 0.5:
                        paths.append(directory / name / str(row["episode_id"]) / "meta.json")
            assert len(paths) == 21333
            origin = "Validated original expert replays; no learned policy"
        else:
            provenance_paths = sorted(directory.glob("collection_config*.json")) + [directory / "summary.json"]
            for p in provenance_paths:
                if p.name != "collection_config_history.json" and p.name.startswith("collection_config"):
                    signatures.add(sha(p))
            paths = sorted(p for p in directory.glob("*/*/meta.json") if not p.parent.name.startswith("."))
            origin = read(directory / "collection_config.json")
        provenance = {"directory": str(directory), "files_sha256": {p.name: sha(p) for p in provenance_paths},
                      "origin": origin, "encoding": "PNG lossless" if source == "selected_policy" else "JPEG quality 95 (historical recorded observations)",
                      "expected_episodes": len(paths)}
        encoded = json.dumps(provenance, sort_keys=True)
        existing = db.execute("SELECT provenance_json FROM sources WHERE name=?", (source,)).fetchone()
        if existing:
            assert existing[0] == encoded, ("Source provenance changed", source)
        else:
            db.execute("INSERT INTO sources VALUES (?,?)", (source, encoded)); db.commit()
        known = dict(db.execute("SELECT directory,meta_sha256 FROM episodes WHERE source=?", (source,)))
        pending = []
        for path in sorted(paths):
            if str(path.parent) in known:
                assert sha(path) == known[str(path.parent)]
            else:
                pending.append((source, path, scenes, signatures, identity))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # A bounded queue keeps decoded metadata/image manifests out of memory.
            for offset in range(0, len(pending), args.workers * 4):
                for item in pool.map(inspect_episode, pending[offset:offset + args.workers * 4]):
                    m = item["meta"]
                    with db:
                        cursor = db.execute("INSERT INTO episodes(source,dataset,episode_id,scene,directory,meta_sha256,encoding,num_actions,num_frames,decoded_frames) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (source, m["dataset"], str(m["episode_id"]), m["scene"], item["path"], item["meta_sha256"], item["encoding"], len(m["executed_action"]), m["num_frames"], item["decoded"]))
                        eid = cursor.lastrowid
                        db.executemany("INSERT INTO frames VALUES (?,?,?,?)", [(eid, *row) for row in item["frames"]])
                        db.executemany("INSERT INTO transitions(episode_pk,t,action,is_branch,target_frame,collided,blocked,displacement_m) VALUES (?,?,?,?,?,?,?,?)", [(eid, *row) for row in item["rows"]])
                done = len(known) + min(offset + args.workers * 4, len(pending))
                progress = {"status": "indexing", "source": source, "completed_source_episodes": done, "expected_source_episodes": len(paths)}
                write(out / "progress.json", progress)
                if done % 256 < args.workers * 4 or done == len(paths):
                    print(json.dumps(progress), flush=True)
        assert db.execute("SELECT count(*) FROM episodes WHERE source=?", (source,)).fetchone()[0] == len(paths)
    if args.stage == "base":
        write(out / "base_ready.json", {"status": "complete", "episodes": db.execute("SELECT count(*) FROM episodes").fetchone()[0]})
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.close()
        return
    db.executescript("""
    DROP TABLE IF EXISTS matched_samples;
    DROP TABLE IF EXISTS early_samples;
    CREATE TABLE matched_samples(slot INTEGER PRIMARY KEY, transition_id INTEGER UNIQUE);
    INSERT INTO matched_samples SELECT ROW_NUMBER() OVER(ORDER BY id)-1,id FROM transitions WHERE t>=3;
    CREATE TABLE early_samples(slot INTEGER PRIMARY KEY, transition_id INTEGER UNIQUE);
    INSERT INTO early_samples SELECT ROW_NUMBER() OVER(ORDER BY id)-1,id FROM transitions WHERE t<3;
    """)
    counts = {"episodes": db.execute("SELECT count(*) FROM episodes").fetchone()[0],
              "frames": db.execute("SELECT count(*) FROM frames").fetchone()[0],
              "image_bytes": db.execute("SELECT sum(bytes) FROM frames").fetchone()[0],
              "decoded_frames_checked": db.execute("SELECT sum(decoded_frames) FROM episodes").fetchone()[0],
              "transitions": db.execute("SELECT count(*) FROM transitions").fetchone()[0],
              "matched": db.execute("SELECT count(*) FROM matched_samples").fetchone()[0],
              "dino_early_extra": db.execute("SELECT count(*) FROM early_samples").fetchone()[0]}
    by_source = {}
    for source, _ in roots:
        by_source[source] = {"episodes": db.execute("SELECT count(*) FROM episodes WHERE source=?", (source,)).fetchone()[0],
                             "scenes": db.execute("SELECT count(DISTINCT scene) FROM episodes WHERE source=?", (source,)).fetchone()[0]}
        by_source[source]["actions"] = {str(a): {"transitions": n, "collisions": collisions, "blocked": blocked, "alternate_branches": branches}
            for a, n, collisions, blocked, branches in db.execute("SELECT t.action,count(*),sum(t.collided),sum(t.blocked),sum(t.is_branch) FROM transitions t JOIN episodes e ON e.id=t.episode_pk WHERE e.source=? GROUP BY t.action", (source,))}
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db.execute("SELECT count(*) FROM transitions t LEFT JOIN frames f ON f.episode_pk=t.episode_pk AND f.relative_path=t.target_frame WHERE f.sha256 IS NULL").fetchone()[0] == 0
    assert counts["transitions"] == counts["matched"] + counts["dino_early_extra"]
    provenance = {name: json.loads(encoded) for name, encoded in db.execute("SELECT * FROM sources")}
    db.commit(); db.execute("PRAGMA wal_checkpoint(TRUNCATE)"); db.execute("PRAGMA journal_mode=DELETE"); db.close()
    manifest = {**config, "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                "catalog_sha256": sha(out / "catalog.sqlite.partial"), "counts": counts, "by_source": by_source,
                "sources": provenance, "image_integrity": "Every frame SHA256, image header/size/format verified; every 31st frame plus episode final fully decoded; loader rechecks hashes on access",
                "data_dependency": "Catalog references original absolute frame paths; preserve all four source corpora",
                "intended_use": "Identical matched sample IDs for transfer/fresh pairs and DINO/CDiT; no group-B/C/E fitting data"}
    write(out / "manifest.pending.json", manifest)
    (out / "catalog.sqlite.partial").rename(out / "catalog.sqlite")
    (out / "manifest.pending.json").replace(out / "manifest.json")
    write(out / "progress.json", {"status": "complete", "counts": counts})
    print(json.dumps({"status": "complete", "counts": counts, "by_source": by_source}), flush=True)


if __name__ == "__main__":
    main()
