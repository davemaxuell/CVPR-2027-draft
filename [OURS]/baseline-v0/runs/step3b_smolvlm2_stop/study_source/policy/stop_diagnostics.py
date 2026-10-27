"""STOP calibration and exact DTW diagnostics; no privileged policy inputs.

DTW follows VLN-CE's ground-truth locations, Euclidean local distance,
consecutive-identical-position suppression and 3 m normalization. We use
exact dynamic programming rather than FastDTW approximation.
https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/measures.py
"""
import numpy as np
from numba import njit


def decode(scores, stop_bias=0.0):
    """STOP-only logit bias; movement ranking remains unchanged."""
    if not np.isfinite(stop_bias):
        raise ValueError("STOP bias must be finite")
    values = np.asarray(scores, dtype=np.float64).copy()
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Expected four finite action scores")
    values[0] += stop_bias
    return int(values.argmax())


@njit(cache=True)
def ndtw_trace(positions, reference, success_distance=3.0):
    """nDTW after every observed position, including the initial state."""
    n, m = len(positions), len(reference)
    if n == 0 or m == 0:
        raise ValueError("DTW paths must be nonempty")
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    output = np.empty(n)
    for i in range(n):
        if i > 0 and np.all(positions[i] == positions[i - 1]):
            output[i] = output[i - 1]
            continue
        current = np.full(m + 1, np.inf)
        for j in range(1, m + 1):
            distance = np.sqrt(np.sum((positions[i] - reference[j - 1]) ** 2))
            current[j] = distance + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
        output[i] = np.exp(-current[m] / (m * success_distance))
    return output


def replay_stop(row, stop_bias):
    """Exact STOP counterfactual on a logged movement-only carrier trajectory.

    For a native greedy trajectory only nonnegative biases are supported: a
    rejected native STOP has no recorded continuation. A movement-only carrier
    supplies those continuations through actual model/simulator execution.
    """
    if not np.isfinite(stop_bias):
        raise ValueError("Nonfinite STOP bias")
    scores = np.asarray(row["action_scores"], dtype=np.float64)
    actions = np.asarray(row["executed_actions"], dtype=int)
    carrier = row.get("movement_only", False)
    expected = 1 + scores[:, 1:].argmax(1) if carrier else scores.argmax(1)
    if len(scores) != len(actions) or not np.array_equal(expected, actions):
        raise ValueError("Recorded actions disagree with the declared carrier")
    if not carrier and stop_bias < 0:
        raise ValueError("Suppressing a native STOP requires real carrier continuations")
    changed = scores.copy()
    changed[:, 0] += stop_bias
    stops = np.flatnonzero(changed.argmax(1) == 0)
    if not len(stops):
        return dict(row)
    t = int(stops[0])
    positions = np.asarray(row["positions"], dtype=np.float64)
    distance = float(row["distances_to_goal"][t])
    shortest = float(row["distances_to_goal"][0])
    travelled = float(np.linalg.norm(np.diff(positions[:t + 1], axis=0), axis=1).sum())
    success = float(distance < 3.0)
    oracle = float(min(row["distances_to_goal"][:t + 1]) < 3.0)
    prefix = actions[:t]
    collisions = np.asarray(row["collided_after_action"][:t], dtype=bool)
    return {**row, "success": success, "spl": success * shortest / max(shortest, travelled, 1e-12),
            "distance_to_goal": distance, "oracle_success": oracle, "steps": t + 1,
            "collisions": int(collisions.sum()), "stopped": True, "termination": "STOP",
            "ndtw": float(row["ndtw_trace"][t]), "sdtw": success * float(row["ndtw_trace"][t]),
            "premature_stop": float(not success), "reached_but_failed": float(oracle and not success),
            "never_reached_timeout": 0.0, "forward_attempts": int((prefix == 1).sum()),
            "forward_collisions": int(((prefix == 1) & collisions).sum()), "stop_step": t,
            "stop_bias": float(stop_bias)}
