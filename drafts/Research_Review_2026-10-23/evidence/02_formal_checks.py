"""Bounded tabular checks of A2 deductions; not CE or learned-model evidence."""
import itertools
import json
import random
from pathlib import Path

rng = random.Random(20261023)
depth = 4
minimum_gain = float('inf')
checked_advantages = 0
for trial in range(500):
    leaves = {p: 0.7 * rng.randrange(2) + 0.3 * rng.random()
              for p in itertools.product(range(2), repeat=depth)}
    collisions = {p: rng.randrange(2) for n in range(1, depth + 1)
                  for p in itertools.product(range(2), repeat=n)}
    q = {}
    value_ref = {}
    value_new = {}
    for n in reversed(range(depth + 1)):
        for h in itertools.product(range(2), repeat=n):
            if n == depth:
                value_ref[h] = value_new[h] = leaves[h]
                continue
            q[h] = [-0.01 / depth - 0.02 * collisions[h + (a,)] / depth
                    + value_ref[h + (a,)] for a in range(2)]
            value_ref[h] = q[h][0]
            chosen = max(range(2), key=lambda a: q[h][a])
            value_new[h] = (-0.01 / depth
                            - 0.02 * collisions[h + (chosen,)] / depth
                            + value_new[h + (chosen,)])
            assert value_new[h] + 1e-12 >= value_ref[h]
            checked_advantages += 1
    minimum_gain = min(minimum_gain, value_new[()] - value_ref[()])

# Exact score-error bound at every override for a two-action root.
epsilon = 0.1
delta = 0.2
override_checks = 0
for true_a, true_b, err_a, err_b in itertools.product(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [-epsilon, 0.0, epsilon], [-epsilon, 0.0, epsilon]):
    gain_hat = true_b + err_b - true_a - err_a
    if gain_hat > delta:
        assert true_b - true_a >= gain_hat - 2 * epsilon - 1e-12
        assert true_b >= true_a - 1e-12
        override_checks += 1

result = {
    'status': 'passed',
    'scope': 'finite deterministic binary trees with path-dependent synthetic terminal scores; not nDTW or CE',
    'seed': 20261023,
    'trees': 500,
    'horizon': depth,
    'history_policy_improvement_checks': checked_advantages,
    'minimum_root_gain': minimum_gain,
    'margin_error_bound_override_checks': override_checks,
    'perfect_stochastic_sample_counterexample': {
        'safe_expected_utility': 0.6,
        'risky_expected_utility': 0.5,
        'one_sample_selection_expected_utility': 0.5 * 0.5 + 0.5 * 0.6,
        'gap_vs_safe': 0.5 * 0.5 + 0.5 * 0.6 - 0.6,
    },
    'occupancy_counterexample': {
        'base_value': 0.6,
        'root_Q_ref': {'base': 0.6, 'override': 0.8},
        'off_support_Q_ref': {'base': 0.8, 'override': 0.0},
        'off_support_estimates': {'base': 0.1, 'override': 0.9},
        'reranker_closed_loop_value': 0.0,
    },
}
path = Path(__file__).with_name('02_formal_checks.json')
path.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
