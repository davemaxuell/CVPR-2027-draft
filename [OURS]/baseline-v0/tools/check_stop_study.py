"""Checks for counterfactual STOP timing, real continuations, metrics and splits."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.stop_diagnostics import decode, ndtw_trace, replay_stop


def independent_ndtw(positions, reference):
    unique = [positions[0]]
    for position in positions[1:]:
        if not np.array_equal(position, unique[-1]):
            unique.append(position)
    table = np.full((len(unique) + 1, len(reference) + 1), np.inf)
    table[0, 0] = 0
    for i, position in enumerate(unique, 1):
        for j, goal in enumerate(reference, 1):
            table[i, j] = np.linalg.norm(position - goal) + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return float(np.exp(-table[-1, -1] / (3 * len(reference))))


def main():
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=float)
    reference = np.array([[x, 0, 0] for x in range(7)], dtype=float)
    trace = ndtw_trace(positions, reference)
    row = {"movement_only": True, "action_scores": [[0, 5, 1, 0], [3, 2, 1, 0], [1, 2, 1, 0], [5, 2, 1, 0]],
           "executed_actions": [1, 1, 1, 1], "positions": positions.tolist(),
           "distances_to_goal": [6, 5, 4, 2, 1], "collided_after_action": [False] * 4,
           "ndtw_trace": trace.tolist(), "success": 0, "steps": 4, "stopped": False}
    native = replay_stop(row, 0)
    assert native["stop_step"] == 1 and native["steps"] == 2 and native["distance_to_goal"] == 5
    assert native["success"] == 0 and native["premature_stop"] == 1
    delayed = replay_stop(row, -2)
    assert delayed["stop_step"] == 3 and delayed["steps"] == 4 and delayed["distance_to_goal"] == 2
    assert delayed["success"] == 1 and delayed["spl"] == 1 and delayed["ndtw"] == trace[3]
    assert replay_stop(row, 6)["stop_step"] == 0
    native_only = {**row, "movement_only": False, "action_scores": row["action_scores"][:2], "executed_actions": [1, 0]}
    try:
        replay_stop(native_only, -1)
        raise AssertionError("Invented a continuation after native STOP")
    except ValueError:
        pass
    rng = np.random.default_rng(17)
    for _ in range(10):
        points = rng.normal(size=(10, 3))
        points[3:5] = points[2]
        ref = rng.normal(size=(7, 3))
        actual = ndtw_trace(points, ref)
        expected = [independent_ndtw(points[:i + 1], ref) for i in range(len(points))]
        assert np.allclose(actual, expected, atol=1e-12, rtol=0)
    assert decode([2, 2, 1, 0]) == 0
    assert decode([2, 2, 1, 0], -.5) == 1
    run = ROOT / "runs/step3b_smolvlm2_stop"
    a, b = [json.loads((run / f"confirmation_{name}.json").read_text()) for name in ("A", "B")]
    for name in ("r2r", "rxr"):
        assert len(a["selected_episodes"][name]) == len(b["selected_episodes"][name]) == 112
        aa = {r["trajectory_id"] for r in a["records"] if r["dataset"] == name}
        bb = {r["trajectory_id"] for r in b["records"] if r["dataset"] == name}
        assert not aa & bb and len(aa) == len(bb) == 112
        old = set(a["exclusions"][name]["trajectory_ids"])
        assert not (aa | bb) & old
    result = {"status": "passed", "pre_action_STOP_timing": True, "native_STOP_suppression_uses_real_continuations": True,
              "cannot_invent_missing_continuation": True, "exact_DTW_matches_independent_implementation": True,
              "confirmation_sets_disjoint_by_trajectory_and_from_prior_validation": True}
    (run / "check_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
