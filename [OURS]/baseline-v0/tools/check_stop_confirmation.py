"""Independently audit the completed, locked STOP study from saved trajectories."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="runs/step3b_smolvlm2_stop")
    args = ap.parse_args()
    study = (ROOT / args.study).resolve()
    result = read(study / "result.json")
    report = result["final_confirmation"]
    assert result["status"] == report["status"] == "complete"
    assert result["improvement_passed_locked_criteria"] == all(report["checks"].values())
    directory = study / ("confirmation_" + report["confirmation_set"])
    assert read(directory / "report.json") == report
    pinned = read(study / "study_source_sha256.json")
    for path, expected in pinned.items():
        assert sha(ROOT / path) == expected, ("source drift", path)
    locked = read(directory / "locked_candidate.json")
    selection = study / ("confirmation_" + report["confirmation_set"] + ".json")
    assert sha(selection) == locked["selection_manifest_sha256"]
    assert sha(study / "plan.json") == locked["plan_sha256"]
    manifest = read(selection)
    expected_ids = {(dataset, str(eid)) for dataset, ids in manifest["selected_episodes"].items() for eid in ids}
    assert len(expected_ids) == 224
    output = {}
    for name in ("candidate", "step2_3072", "step3_1024"):
        config = read(directory / name / "eval_config.json")
        assert not config["movement_only"]
        assert config["selection_manifest_sha256"] == sha(selection)
        assert sha(Path(config["checkpoint"]) / "adapter_model.safetensors") == config["adapter_sha256"]
        if name == "candidate":
            assert config["adapter_sha256"] == locked["adapter_sha256"]
            assert config["stop_bias"] == locked["stop_bias"]
        else:
            expected_checkpoint = ROOT / ("runs/step2_smolvlm2/train/checkpoints/step_003072" if name == "step2_3072" else "runs/step3_smolvlm2/train/checkpoints/step_001024")
            assert Path(config["checkpoint"]).resolve() == expected_checkpoint
            assert config["stop_bias"] == 0
        rows = [read(path) for path in sorted((directory / name / "episodes").glob("*.json"))]
        assert len(rows) == 224
        assert {(r["dataset"], str(r["episode_id"])) for r in rows} == expected_ids
        cells = Counter((r["dataset"], r["scene"]) for r in rows)
        assert len(cells) == 14 and set(cells.values()) == {16}
        counts = Counter()
        spl_values = []
        for row in rows:
            actions = np.asarray(row["executed_actions"])
            positions = np.asarray(row["positions"], dtype=np.float64)
            distances = np.asarray(row["distances_to_goal"], dtype=np.float64)
            collisions = np.asarray(row["collided_after_action"], dtype=bool)
            scores = np.asarray(row["action_scores"], dtype=np.float64)
            steps = len(actions)
            assert positions.shape == (steps + 1, 3) and len(distances) == steps + 1
            assert scores.shape == (steps, 4) and len(collisions) == steps
            assert all(np.isfinite(a).all() for a in (positions, distances, scores))
            scores[:, 0] += config["stop_bias"]
            assert np.array_equal(np.argmax(scores, axis=1), actions)
            assert not np.any(actions[:-1] == 0)
            stopped = bool(actions[-1] == 0)
            radius = config["simulator"]["success_distance_m"]
            success = stopped and distances[-1] < radius
            reached = distances.min() < radius
            wrong_stop = stopped and not success
            path_length = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
            spl = float(success) * distances[0] / max(distances[0], path_length, 1e-12)
            for key, value in {"success": success, "oracle_success": reached, "premature_stop": wrong_stop,
                               "reached_but_failed": reached and not success, "stopped": stopped,
                               "steps": steps, "distance_to_goal": distances[-1], "spl": spl}.items():
                assert abs(row[key] - value) < 1e-6, (name, row["episode_id"], key, row[key], value)
            assert stopped or steps == config["simulator"]["max_episode_steps"][row["dataset"]]
            forward = actions == 1
            assert row["forward_attempts"] == int(forward.sum())
            assert row["forward_collisions"] == int((forward & collisions).sum())
            counts.update(episodes=1, steps=steps, successes=int(success), goal_visits=int(reached),
                          incorrect_stops=int(wrong_stop), timeouts=int(not stopped),
                          reached_then_incorrect_stop=int(reached and wrong_stop),
                          reached_then_timeout=int(reached and not stopped),
                          never_reached_then_stop=int(not reached and stopped),
                          never_reached_then_timeout=int(not reached and not stopped),
                          forward_attempts=int(forward.sum()), forward_collisions=int((forward & collisions).sum()))
            spl_values.append(spl)
        metrics = report["metrics"][name]
        assert metrics["overall"]["successes"] == counts["successes"]
        for key, value in {"macro_success": counts["successes"] / 224,
                           "macro_oracle_success": counts["goal_visits"] / 224,
                           "macro_premature_stop": counts["incorrect_stops"] / 224,
                           "macro_spl": float(np.mean(spl_values))}.items():
            assert abs(metrics[key] - value) < 1e-6, (name, key)
        output[name] = {**dict(counts), "spl": float(np.mean(spl_values)),
                        "forward_collision_rate": counts["forward_collisions"] / max(1, counts["forward_attempts"])}
    original = ROOT / "runs/step3_smolvlm2/frozen_reference"
    unchanged = 0
    for path, expected in read(original / "inference.json")["source_sha256"].items():
        assert sha(ROOT / path) == expected, ("original source drift", path)
        unchanged += 1
    audit = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
             "pinned_sources_verified": len(pinned), "original_sources_unchanged": unchanged,
             "locked_confirmation_verified": True, "independently_recomputed": output,
             "result_sha256": sha(study / "result.json"), "report_sha256": sha(directory / "report.json")}
    (study / "check_final.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
