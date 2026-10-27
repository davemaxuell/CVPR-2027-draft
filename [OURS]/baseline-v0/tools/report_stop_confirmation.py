"""Apply the prospectively locked pass criteria to matched held-out episodes."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.stop_study_metrics import paired_comparison, summarize
from tools.calibrate_stop import rows_at


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="runs/step3b_smolvlm2_stop")
    ap.add_argument("--set", choices=["A", "B"], required=True)
    args = ap.parse_args()
    study = Path(args.study).resolve()
    root = study / f"confirmation_{args.set}"
    locked = read(root / "locked_candidate.json")
    manifest = read(study / f"confirmation_{args.set}.json")
    expected = {(name, str(eid)) for name, ids in manifest["selected_episodes"].items() for eid in ids}
    rows, metrics = {}, {}
    for name in ("candidate", "step2_3072", "step3_1024"):
        assert read(root / name / "summary.json")["status"] == "complete"
        config = read(root / name / "eval_config.json")
        assert not config["movement_only"] and config["selected_episodes"] == manifest["selected_episodes"]
        if name == "candidate":
            assert config["adapter_sha256"] == locked["adapter_sha256"] and config["stop_bias"] == locked["stop_bias"]
        else:
            assert config["stop_bias"] == 0
        rows[name] = rows_at(root / name)
        assert len(rows[name]) == len(expected) == 224
        assert {(r["dataset"], r["episode_id"]) for r in rows[name]} == expected
        cells = Counter((r["scene"], r["dataset"]) for r in rows[name])
        assert len(cells) == 14 and set(cells.values()) == {16}, "Scene bootstrap requires balanced cells"
        metrics[name] = summarize(rows[name])
    if (root / "step3_2048" / "summary.json").exists():
        rows["step3_2048"] = rows_at(root / "step3_2048")
        metrics["step3_2048"] = summarize(rows["step3_2048"])
    paired = {name: paired_comparison(rows["candidate"], value) for name, value in rows.items() if name != "candidate"}
    candidate, step2, step3 = (metrics[name] for name in ("candidate", "step2_3072", "step3_1024"))
    criteria = read(study / "plan.json")["confirmation"]["pass"]
    checks = {"SR_gain_at_least_5pp_vs_each_control": all(paired[name]["success"]["difference"] + 1e-12 >= criteria["absolute_macro_SR_gain_vs_each_control_at_least"] for name in ("step2_3072", "step3_1024")),
              "SR_paired_scene_CI_excludes_zero_vs_each_control": all(paired[name]["success"]["scene_bootstrap_95ci"][0] > 0 for name in ("step2_3072", "step3_1024")),
              "SPL_at_least_step2": candidate["macro_spl"] >= step2["macro_spl"],
              "premature_stops_at_most_step2": candidate["macro_premature_stop"] <= step2["macro_premature_stop"],
              "nDTW_drop_vs_step3_at_most_0.05": candidate["macro_ndtw"] >= step3["macro_ndtw"] - criteria["mean_nDTW_drop_vs_unmodified_step3_1024_at_most"]}
    report = {"status": "complete", "confirmation_set": args.set, "candidate": locked, "passed": all(checks.values()),
              "checks": checks, "metrics": metrics, "paired_comparisons": paired,
              "scope": "224 distinct held-out trajectories in seven group-C buildings; not final group-E evaluation",
              "limitations": ["Seven-scene uncertainty", "15-degree RxR actions differ from official 30-degree protocol",
                              "Calibration and checkpoint selection used the earlier 64 development episodes only"]}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "passed": report["passed"], "checks": checks,
                      "success": {k: v["macro_success"] for k, v in metrics.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
