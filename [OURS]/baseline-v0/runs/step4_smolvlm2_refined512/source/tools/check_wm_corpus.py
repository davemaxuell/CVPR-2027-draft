"""Audit the completed transition corpus, paired-family loader, and frozen reference."""
import argparse
import hashlib
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wm.data import TransitionDataset


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/wm_groupA_smolvlm2_refined512_v1")
    ap.add_argument("--run", default=str(ROOT / "runs/step4_smolvlm2_refined512"))
    args = ap.parse_args()
    corpus, run = Path(args.corpus), Path(args.run)
    manifest = read(corpus / "manifest.json")
    assert manifest["status"] == "complete" and sha(corpus / "catalog.sqlite") == manifest["catalog_sha256"]
    reference = run / "reference"
    metadata = read(reference / "reference.json")
    assert metadata["policy_identity_sha256"] == manifest["selected_policy_identity_sha256"]
    assert metadata["decoder"]["stop_logit_bias"] == -0.25
    assert not metadata["selection"]["original_stopping_safeguard_passed"]
    assert metadata["selection"]["authority"] == "explicit_user_selection_2026-10-26"
    for relative, expected in read(reference / "file_sha256.json").items():
        assert sha(reference / relative) == expected
    control = Path(metadata["selection"]["control_reference"])
    assert sha(control / "sha256.json") == metadata["selection"]["control_bundle_manifest_sha256"]
    for relative, expected in read(control / "sha256.json").items():
        assert sha(control / relative) == expected
    assert sha(ROOT / "runs/step3b_smolvlm2_stop/reference_decision.json") == sha(reference / "evidence/reference_decision.json")
    for relative, expected in read(run / "source_sha256.json").items():
        assert sha(ROOT / relative) == expected
        assert sha(run / "source" / relative) == expected
    runtime = read(run / "check_runtime.json")
    assert runtime["status"] == "passed" and runtime["policy_identity_sha256"] == metadata["policy_identity_sha256"]
    assert runtime["trainable_parameters"] == 0 and runtime["evaluation_mode"]
    for relative, expected in runtime["source_sha256"].items():
        assert sha(ROOT / relative) == expected
    db = sqlite3.connect(f"file:{corpus / 'catalog.sqlite'}?mode=ro&immutable=1", uri=True)
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    scenes = set(read(ROOT / "manifest/scene_partition.json")["groups"]["A"]["scenes"])
    assert {r[0] for r in db.execute("SELECT DISTINCT scene FROM episodes")} <= scenes
    assert dict(db.execute("SELECT source,count(*) FROM episodes GROUP BY source")) == {"expert": 21333, "dagger_round1": 256, "dagger_round2": 128, "selected_policy": 256}
    assert db.execute("SELECT count(*) FROM transitions WHERE action NOT IN(1,2,3)").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM matched_samples s JOIN transitions t ON t.id=s.transition_id WHERE t.t<3").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM early_samples s JOIN transitions t ON t.id=s.transition_id WHERE t.t>=3 OR t.is_branch=1").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM transitions t LEFT JOIN frames f ON f.episode_pk=t.episode_pk AND f.relative_path=t.target_frame WHERE f.sha256 IS NULL").fetchone()[0] == 0
    blocked = db.execute("SELECT count(*) FROM transitions WHERE action=1 AND blocked=1").fetchone()[0]
    assert blocked > 0
    dino = TransitionDataset(corpus, 3, expected_policy_identity=metadata["policy_identity_sha256"])
    cdit = TransitionDataset(corpus, 4, expected_policy_identity=metadata["policy_identity_sha256"])
    assert len(dino) == len(cdit) == manifest["counts"]["matched"]
    slots = set(random.Random(4).sample(range(len(dino)), min(256, len(dino))))
    slots.update((0, len(dino) - 1))
    for source in manifest["by_source"]:
        for clause in ("t.action=1", "t.action=2", "t.action=3", "t.blocked=1", "t.is_branch=1"):
            row = db.execute(f"SELECT min(s.slot) FROM matched_samples s JOIN transitions t ON t.id=s.transition_id JOIN episodes e ON e.id=t.episode_pk WHERE e.source=? AND {clause}", (source,)).fetchone()
            if row[0] is not None:
                slots.add(row[0])
    for slot in sorted(slots):
        a, b = dino[slot], cdit[slot]
        assert a["sample_id"] == b["sample_id"] and a["action"] == b["action"]
        assert len(a["observations"]) == 3 and len(b["observations"]) == 4
        assert len(a["executed_history_actions"]) == 2 and len(b["executed_history_actions"]) == 3
        assert a["executed_history_actions"] == b["executed_history_actions"][-2:]
        assert np.array_equal(a["target_observation"], b["target_observation"])
        for left, right in zip(a["observations"], b["observations"][-3:]):
            assert np.array_equal(left, right)
        assert a["movement_action_index"] == a["action"] - 1
        assert "position" not in a and "displacement" not in a
    extra = TransitionDataset(corpus, 3, subset="dino_early_extra")
    for slot in sorted({0, min(1, len(extra) - 1), min(2, len(extra) - 1), len(extra) - 1}):
        sample = extra[slot]
        assert sample["t"] < 3 and len(sample["observations"]) == sample["t"] + 1
    try:
        TransitionDataset(corpus, 4, subset="dino_early_extra")
    except AssertionError:
        pass
    else:
        raise AssertionError("CDiT accepted fewer than four frames")
    try:
        TransitionDataset(corpus, expected_policy_identity="wrong policy identity")
    except AssertionError:
        pass
    else:
        raise AssertionError("Corpus accepted a different selected policy")
    report = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "counts": manifest["counts"], "paired_family_samples_checked": len(slots),
              "same_root_action_IDs_and_actual_targets": True, "group_A_only": True, "STOP_excluded": True,
              "blocked_forward_transitions_retained": blocked, "all_frames_hashed_and_headers_verified": True,
              "saved_reference_and_bias_verified": True, "control_preserved": True,
              "failed_stopping_safeguard_preserved_in_selection_record": True,
              "native_runtime_checks": runtime, "catalog_sha256": manifest["catalog_sha256"],
              "policy_identity_sha256": metadata["policy_identity_sha256"]}
    (run / "final_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "native_runtime_checks"}, indent=2))


if __name__ == "__main__":
    main()
