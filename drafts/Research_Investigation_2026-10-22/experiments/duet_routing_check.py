"""Execute only the reviewed FloydGraph class from pinned official DUET source.

This is a synthetic four-node graph check, not a navigation benchmark, trained
model evaluation, or estimate of how often DUET actually selects a backtrack.
"""
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repositories/01/duet/map_nav_src/models/graph_utils.py"
code = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(code)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FloydGraph")
# The class, including each executed method, was read before this test. Do not
# import the project or execute its module-level code, setup, or training scripts.
module = ast.Module(body=[node], type_ignores=[])
namespace = {"defaultdict": defaultdict}
exec(compile(module, str(SOURCE), "exec"), namespace)
graph = namespace["FloydGraph"]()
for left, right in [("start", "junction"), ("junction", "dead_end"), ("junction", "frontier")]:
    graph.add_edge(left, right, 1.0)
for visited in ["start", "junction", "dead_end"]:
    graph.update(visited)
frontier_route = graph.path("dead_end", "frontier")
return_route = graph.path("dead_end", "start")
assert frontier_route == ["junction", "frontier"]
assert return_route == ["junction", "start"]
assert graph.visited("junction") and graph.visited("start")
result = {
    "evidence_type": "EXECUTED TEST",
    "date": "2026-10-22",
    "source_repository": "https://github.com/cshizhe/VLN-DUET",
    "source_commit": "93e8b233164bc079a6db48b8a0a78d123ec8de41",
    "source_path": "map_nav_src/models/graph_utils.py",
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "executed_component": "FloydGraph class only, AST-extracted from inspected pinned source",
    "setup": "Four-node unit-edge tree; start, junction and dead_end marked visited; frontier unvisited; current node dead_end",
    "frontier_route": frontier_route,
    "return_to_visited_start_route": return_route,
    "finding": "The official routing class returns paths through visited nodes and can return to a visited destination.",
    "scope": "Synthetic routing check only. Selection/masking behavior is separately established by source inspection of agent.py and vilmodel.py. No trained model, simulator, benchmark, frequency, or performance claim.",
}
out = ROOT / "experiments/duet_routing_check_results.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
