"""Tune only on locked C64; exact STOP-only replay on real carrier trajectories."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.stop_diagnostics import ndtw_trace, replay_stop
from policy.stop_study_metrics import summarize


def read(path):
    return json.loads(Path(path).read_text())


def rows_at(directory):
    return [read(p) for p in sorted((Path(directory) / "episodes").glob("*.json"))]


def augment_original(rows):
    from sim.config import load_expert
    for name in ("r2r", "rxr"):
        gt = load_expert(name, "train")
        for row in rows:
            if row["dataset"] != name:
                continue
            trace = ndtw_trace(np.asarray(row["positions"], dtype=np.float64), np.asarray(gt[row["episode_id"]]["locations"], dtype=np.float64))
            actions = np.asarray(row["executed_actions"])
            row.update(ndtw=float(trace[-1]), sdtw=float(trace[-1]) * row["success"],
                       premature_stop=float(row["stopped"] and not row["success"]),
                       reached_but_failed=float(row["oracle_success"] and not row["success"]),
                       never_reached_timeout=float(not row["oracle_success"] and not row["stopped"]),
                       forward_attempts=int((actions == 1).sum()),
                       forward_collisions=int(((actions == 1) & np.asarray(row["collided_after_action"])).sum()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="runs/step3b_smolvlm2_stop")
    ap.add_argument("--evaluations", nargs="+", help="Carrier evaluation directories; default diagnostic 1024 and 2048")
    ap.add_argument("--out", help="Report directory; default study/calibration")
    args = ap.parse_args()
    study = Path(args.study).resolve()
    out = Path(args.out).resolve() if args.out else study / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    plan = read(study / "plan.json")
    paths = list(map(Path, args.evaluations)) if args.evaluations else [study / "diagnostic" / f"step_{s:06d}" for s in (1024, 2048)]
    baseline_path = study / "step2_development_metrics.json"
    if baseline_path.exists():
        baseline = read(baseline_path)
    else:
        baseline_rows = augment_original(rows_at(ROOT / "runs/step2_smolvlm2/evaluations/step_003072"))
        baseline = summarize(baseline_rows)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    expected = read(study / "development.json")["selected_episodes"]
    expected_ids = {(name, eid) for name, ids in expected.items() for eid in ids}
    comparisons, native, carrier_rows = [], {}, {}
    for path in paths:
        assert read(path / "summary.json")["status"] == "complete"
        config = read(path / "eval_config.json")
        assert config["movement_only"] and config["selected_episodes"] == expected
        rows = rows_at(path)
        assert len(rows) == 64 and {(r["dataset"], r["episode_id"]) for r in rows} == expected_ids
        carrier_rows[str(path)] = rows
        native[str(path)] = summarize([replay_stop(row, 0) for row in rows])
        for bias in plan["calibration"]["bias_grid"]:
            metrics = summarize([replay_stop(row, bias) for row in rows])
            comparisons.append({"evaluation": str(path), "checkpoint": config["checkpoint"],
                                "adapter_sha256": config["adapter_sha256"], "stop_bias": bias, "metrics": metrics})
    def rank(c):
        m = c["metrics"]
        return (m["macro_success"], m["macro_spl"], -m["macro_premature_stop"], -c["stop_bias"])
    feasible = [c for c in comparisons if c["metrics"]["macro_premature_stop"] <= baseline["macro_premature_stop"]
                and c["metrics"]["macro_spl"] >= baseline["macro_spl"]]
    best = max(feasible or comparisons, key=rank)
    sufficient = bool(feasible) and best["metrics"]["macro_success"] >= plan["calibration"]["sufficient_on_development"]["macro_success_at_least"]
    selected_rows = [replay_stop(row, best["stop_bias"]) for row in carrier_rows[best["evaluation"]]]
    compact = [{key: row[key] for key in ("dataset", "episode_id", "scene", "success", "spl", "oracle_success",
                "ndtw", "sdtw", "premature_stop", "reached_but_failed", "never_reached_timeout", "steps", "distance_to_goal")}
               for row in selected_rows]
    report = {"status": "complete", "development_only": True, "exact_replay_on_real_movement_only_carriers": True,
              "baseline_step2": baseline, "native_counterfactuals": native, "selected": best,
              "sufficient_on_development": sufficient, "feasible_candidates": len(feasible),
              "candidate_rules": len(comparisons), "selection": plan["calibration"]["selection"],
              "note": "STOP probability is not assumed calibrated. No confirmation outcomes were used."}
    (out / "sweep.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    (out / "selected_episode_metrics.json").write_text(json.dumps(compact, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
