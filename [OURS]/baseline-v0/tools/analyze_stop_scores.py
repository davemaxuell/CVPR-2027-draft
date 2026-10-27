"""Describe STOP discrimination on native prefixes; never tune on confirmation."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.stop_diagnostics import replay_stop
from tools.calibrate_stop import rows_at


def weighted_auc(score, positive, weight):
    order = np.argsort(score)
    score, positive, weight = score[order], positive[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
    p = np.add.reduceat(weight * positive, starts)
    n = np.add.reduceat(weight * ~positive, starts)
    if not p.sum() or not n.sum():
        return None
    return float(np.sum(p * (np.cumsum(n) - .5 * n)) / (p.sum() * n.sum()))


def main():
    study = ROOT / "runs/step3b_smolvlm2_stop"
    result = {}
    for checkpoint in (1024, 2048):
        directory = study / "diagnostic" / f"step_{checkpoint:06d}"
        assert json.loads((directory / "summary.json").read_text())["status"] == "complete"
        margins, probabilities, distances, weights = [], [], [], []
        episodes = rows_at(directory)
        for row in episodes:
            native = replay_stop(row, 0.)
            size = native["steps"]
            score = np.asarray(row["action_scores"][:size], dtype=np.float64)
            margin = score[:, 0] - score[:, 1:].max(1)
            exponential = np.exp(score - score.max(1, keepdims=True))
            margins.extend(margin)
            probabilities.extend(exponential[:, 0] / exponential.sum(1))
            distances.extend(row["distances_to_goal"][:size])
            weights.extend([1. / size] * size)
        d = np.asarray(distances)
        w = np.asarray(weights)
        scores = {"STOP_margin": np.asarray(margins), "four_action_STOP_softmax": np.asarray(probabilities)}
        report = {"episodes": len(episodes), "states": len(d), "inside_2m_states": int((d < 2).sum()),
                  "inside_3m_states": int((d < 3).sum()), "scope": "Only actual native-policy prefixes through the first native STOP"}
        for name, score in scores.items():
            certain = (d < 2) | (d >= 3)
            report[name] = {"state_weighted_AUC_inside_3m": weighted_auc(score, d < 3, np.ones(len(d))),
                            "episode_balanced_AUC_inside_3m": weighted_auc(score, d < 3, w),
                            "episode_balanced_AUC_inside_2m_vs_outside_3m": weighted_auc(score[certain], d[certain] < 2, w[certain]),
                            "inside_2m_quantiles_10_50_90": np.quantile(score[d < 2], [.1, .5, .9]).tolist(),
                            "outside_3m_quantiles_10_50_90": np.quantile(score[d >= 3], [.1, .5, .9]).tolist()}
        result[str(checkpoint)] = report
    output = {"status": "complete", "checkpoints": result,
              "interpretation": "AUC is a per-state diagnostic, not an episode success estimate. One early false STOP can terminate an otherwise useful trajectory.",
              "not_used_for_selection": True}
    (study / "stop_discrimination.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
