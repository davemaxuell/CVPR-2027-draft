"""Small original counterexamples; no navigation model or benchmark is executed."""
from collections import deque
from pathlib import Path
import json
import platform


def shortest_actions(graph, start, remaining):
    """Enumerate shortest first actions in a tiny product graph (node, obligations)."""
    queue = deque([(start, tuple(remaining), None, 0)])
    visited = {}
    solutions = []
    best = None
    while queue:
        node, todo, first, depth = queue.popleft()
        key = (node, todo, first)
        if key in visited and visited[key] <= depth:
            continue
        visited[key] = depth
        if not todo:
            if best is None:
                best = depth
            if depth == best:
                solutions.append(first)
            continue
        if best is not None and depth >= best:
            continue
        for nxt in graph[node]:
            rest = todo[1:] if nxt == todo[0] else todo
            queue.append((nxt, rest, nxt if first is None else first, depth + 1))
    return sorted(set(solutions)), best


def first_order(paths, gamma):
    out = [0.0, 0.0]
    for path in paths:
        for step, event in enumerate(path):
            out[event] += gamma ** step / len(paths)
    return out


def ordered_event_probability(paths, left, right):
    return sum(any(path[i] == left and path[j] == right
                   for i in range(len(path)) for j in range(i + 1, len(path)))
               for path in paths) / len(paths)


graph = {"junction": ["A", "goal"], "A": ["junction"], "goal": ["junction"]}
before = shortest_actions(graph, "junction", ["A", "goal"])
after = shortest_actions(graph, "junction", ["goal"])
# Same pose and instruction. History differs by whether event A was completed.
assert before[0] == ["A"] and after[0] == ["goal"]

same_event_paths = [[0, 0], [1, 1]]
alternating_paths = [[0, 1], [1, 0]]
rows = []
for gamma in [0.0, 0.3, 0.5, 0.7, 1.0]:
    u = first_order(same_event_paths, gamma)
    v = first_order(alternating_paths, gamma)
    assert u == v
    rows.append({"gamma": gamma, "mixture_AA_BB": u, "mixture_AB_BA": v})
j1 = ordered_event_probability(same_event_paths, 0, 1)
j2 = ordered_event_probability(alternating_paths, 0, 1)
assert (j1, j2) == (0.0, 0.5)

# A deliberately simplified completion contract, solely to expose non-identifiability.
failure_cases = [{"correct_identity": d, "executable_approach": r,
                  "success_under_toy_contract": int(d and r)}
                 for d in [0, 1] for r in [0, 1] if not (d and r)]
assert any(r["correct_identity"] == 1 for r in failure_cases)

result = {
    "python": platform.python_version(),
    "scope": "Original finite graph/arithmetic checks. No fetched model code, trained model, simulator or navigation benchmark.",
    "P1_history_dependence": {
        "graph": graph, "instruction": "Visit A, then reach goal", "current_node": "junction",
        "A_uncompleted": {"optimal_first_actions": before[0], "minimum_steps": before[1]},
        "A_completed": {"optimal_first_actions": after[0], "minimum_steps": after[1]},
        "establishes": "One current location is insufficient for this precisely specified history-dependent task.",
        "does_not_establish": "Prevalence in R2R/RxR, failures of any learned policy, or novelty of automata/product-graph planning."
    },
    "P3_first_order_aliasing": {
        "paths": {"mixture_AA_BB": same_event_paths, "mixture_AB_BA": alternating_paths},
        "first_order_results": rows,
        "probability_A_before_B": {"mixture_AA_BB": j1, "mixture_AB_BA": j2},
        "establishes": "These distributions share every timewise event marginal and therefore every additive discounted event mean, but differ on an ordered joint event.",
        "does_not_establish": "Actual NavQ target collision frequency, universal failure of its complete architecture, or learnability of unseen hidden facts."
    },
    "P4_failure_nonidentifiability": {
        "toy_contract": "success = correct_identity AND executable_approach",
        "failed_cases": failure_cases,
        "establishes": "A failed episode is compatible with both correct and incorrect identity under this contract.",
        "does_not_establish": "That every benchmark obeys this simplified contract or a measured PSG-Nav memory-poisoning rate."
    }
}
Path(__file__).with_name("mechanism_check_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
