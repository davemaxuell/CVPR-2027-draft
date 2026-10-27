"""Verify metadata-only recovery preserved completed data and failed-episode images."""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
VISITS = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/wm_visits_groupA_smolvlm2_refined512_v1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    completed = json.loads((RUN / "episodes_before_nonfinite_fix.json").read_text())
    for relative, expected in completed.items():
        assert sha(VISITS / relative) == expected, relative
    episode = VISITS / "rxr/4838"
    frame_hashes = json.loads((RUN / "rxr_4838_frames_before_nonfinite_fix.json").read_text())
    for relative, expected in frame_hashes.items():
        assert sha(episode / relative) == expected, relative
    row = json.loads((episode / "meta.json").read_text())
    assert row["num_actions"] == 700 and row["num_frames"] == 701
    assert row["stop_bias"] == -0.25 and row["nonfinite_goal_diagnostics"]
    assert row["policy_identity_sha256"] == json.loads((RUN / "reference/reference.json").read_text())["policy_identity_sha256"]
    for path, value in row["nonfinite_goal_diagnostics"].items():
        assert value in ("inf", "-inf", "nan")
        section, index = path.split("/")
        assert row[section][int(index) if section == "distances_to_goal" else index] is None
    report = {"status": "passed", "completed_episode_metadata_unchanged": len(completed),
              "failed_episode_images_identical_after_rerun": len(frame_hashes),
              "policy_identity_and_STOP_bias_unchanged": True,
              "diagnostic_values_explicitly_unavailable": len(row["nonfinite_goal_diagnostics"]),
              "actions": row["num_actions"], "metrics": row["metrics"]}
    (RUN / "check_serialization_recovery.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
