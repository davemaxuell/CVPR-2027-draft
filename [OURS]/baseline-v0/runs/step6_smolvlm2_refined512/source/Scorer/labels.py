"""Utility from observed branch outcomes. Never called on imagined metrics.

Success and nDTW must come from the configured CE evaluator on the FULL path,
including the real prefix. This helper is not an implementation of CE metrics.
"""

import math


def utility_target(success: int, ndtw: float, remaining_actions: int,
                   remaining_collisions: int, max_episode_actions: int) -> float:
    if success not in (0, 1) or not math.isfinite(ndtw) or not 0 <= ndtw <= 1:
        raise ValueError("Expected binary success and nDTW between zero and one")
    counts = (remaining_actions, remaining_collisions, max_episode_actions)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in counts):
        raise ValueError("Action and collision counts must be integers")
    if not (1 <= remaining_actions <= max_episode_actions and 0 <= remaining_collisions <= remaining_actions):
        raise ValueError("Invalid branch counts or maximum episode length")
    return (0.7 * success + 0.3 * ndtw
            - 0.01 * remaining_actions / max_episode_actions
            - 0.02 * remaining_collisions / max_episode_actions)
