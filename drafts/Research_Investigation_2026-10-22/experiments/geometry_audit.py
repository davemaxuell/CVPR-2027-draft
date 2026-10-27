"""Static R2R reference-prefix metric audit; no navigator/simulator is run.

Fetch public connectivity JSON from an immutable Matterport3DSimulator commit.
The graph construction and SPL formula are independently implemented after
inspection of the pinned DUET loader/evaluator. No fetched Python is imported.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parents[1]
OUT = ROOT / "experiments/geometry"
CONNECTIVITY = OUT / "connectivity"
COMMIT = "589d091b111333f9e9f9d6cfd021b2eb68435925"
DUET_COMMIT = "93e8b233164bc079a6db48b8a0a78d123ec8de41"
ANNOTATIONS = WORKSPACE / "261020/Presentation/_source_Benchmarks_Baseline/cvpr_review_response_2026-10-21"
RADIUS = 3.0


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(scan):
    target = CONNECTIVITY / f"{scan}_connectivity.json"
    url = f"https://raw.githubusercontent.com/peteanderson80/Matterport3DSimulator/{COMMIT}/connectivity/{target.name}"
    if not target.exists():
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        response.json()  # Parse before persisting an unexpected response.
        target.write_bytes(response.content)
    raw = target.read_bytes()
    return scan, json.loads(raw), {"scan": scan, "url": url, "sha256": sha(raw), "bytes": len(raw)}


def graph(data):
    result = {r["image_id"]: {} for r in data if r["included"]}
    for i, r in enumerate(data):
        if not r["included"]:
            continue
        for j, connected in enumerate(r["unobstructed"]):
            if connected and data[j]["included"]:
                if not data[j]["unobstructed"][i]:
                    raise ValueError("Connectivity must be symmetric")
                distance = math.sqrt(sum((r["pose"][k] - data[j]["pose"][k]) ** 2 for k in (3, 7, 11)))
                result[r["image_id"]][data[j]["image_id"]] = distance
    return result


def shortest(g, start):
    distance = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for neighbor, edge in g[node].items():
            proposed = cost + edge
            if proposed < distance.get(neighbor, math.inf):
                distance[neighbor] = proposed
                heapq.heappush(queue, (proposed, neighbor))
    return distance


def main():
    CONNECTIVITY.mkdir(parents=True, exist_ok=True)
    splits, annotation_manifest = {}, []
    for split in ("train", "val_unseen"):
        path = ANNOTATIONS / f"R2R_{split}.json"
        raw = path.read_bytes()
        splits[split] = json.loads(raw)
        annotation_manifest.append({"split": split, "path": str(path), "sha256": sha(raw), "paths": len(splits[split])})
    scans = sorted({p["scan"] for rows in splits.values() for p in rows})
    with ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(fetch, scans))
    graphs = {scan: graph(data) for scan, data, _ in downloaded}
    cache = {}
    rows, invalid = [], []
    for split, paths in splits.items():
        for p in paths:
            scan, route = p["scan"], p["path"]
            g = graphs[scan]
            if any(b not in g.get(a, {}) for a, b in zip(route, route[1:])):
                invalid.append({"split": split, "path_id": p["path_id"], "reason": "Reference includes a missing direct graph edge"})
                continue
            goal_key = (scan, route[-1])
            if goal_key not in cache:
                cache[goal_key] = shortest(g, route[-1])
            goal_dist = cache[goal_key]
            prefix_length = [0.0]
            for a, b in zip(route, route[1:]):
                prefix_length.append(prefix_length[-1] + g[a][b])
            full_length = prefix_length[-1]
            candidates = [i for i in range(len(route) - 1) if goal_dist.get(route[i], math.inf) < RADIUS]
            index = min(candidates) if candidates else None
            executed = prefix_length[index] if index is not None else None
            spl = full_length / max(executed, full_length, 0.01) if index is not None else None
            rows.append({
                "split": split, "scan": scan, "path_id": p["path_id"],
                "reference_nodes": len(route), "reference_length_m": full_length,
                "proper_successful_prefix_exists": index is not None,
                "earliest_successful_prefix_nodes": index + 1 if index is not None else None,
                "omitted_reference_edges": len(route) - 1 - index if index is not None else 0,
                "omitted_reference_length_m": full_length - executed if index is not None else 0.0,
                "prefix_goal_distance_m": goal_dist[route[index]] if index is not None else None,
                "prefix_spl_under_reviewed_formula": spl,
            })
    summary = {}
    for split in splits:
        subset = [r for r in rows if r["split"] == split]
        passing = [r for r in subset if r["proper_successful_prefix_exists"]]
        summary[split] = {
            "valid_reference_paths": len(subset), "proper_successful_prefix_paths": len(passing),
            "fraction_of_reference_paths": len(passing) / len(subset),
            "all_passing_prefixes_have_spl_one": all(abs(r["prefix_spl_under_reviewed_formula"] - 1) < 1e-12 for r in passing),
            "median_omitted_reference_length_m_among_passing": statistics.median(r["omitted_reference_length_m"] for r in passing),
            "max_omitted_reference_edges": max(r["omitted_reference_edges"] for r in passing),
        }
    result = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_type": "EXECUTED STATIC DATASET GEOMETRY AND METRIC CHECK",
        "navigation_benchmark_executed": False,
        "connectivity_repo": "https://github.com/peteanderson80/Matterport3DSimulator",
        "connectivity_commit": COMMIT,
        "connectivity_scenes": len(scans),
        "reference_metric_source": f"https://github.com/cshizhe/VLN-DUET/blob/{DUET_COMMIT}/map_nav_src/r2r/env.py#L314",
        "graph_source": f"https://github.com/cshizhe/VLN-DUET/blob/{DUET_COMMIT}/map_nav_src/utils/data.py#L25",
        "success_condition": "Graph shortest-path endpoint distance strictly below 3 metres",
        "spl_formula": "success * reference_length / max(executed_length, reference_length, 0.01)",
        "unit": "One reference path; multiple natural-language instructions per path are not counted as independent geometries",
        "annotation_sources": annotation_manifest,
        "summary": summary,
        "invalid_paths": invalid,
        "interpretation": "The reviewed success/SPL criterion permits some proper reference prefixes to receive full credit. It therefore cannot logically certify visiting every reference waypoint.",
        "limitations": [
            "This is expected tolerance behavior, not an evaluator bug.",
            "No model selected these prefixes; no navigator, simulator, checkpoint, image encoder or original project script was run.",
            "Language entailment was not annotated. Omitting a waypoint is not automatically an instruction violation.",
            "These are supplied local R2R reference annotations, not SRDF-generated data; no empirical SRDF failure rate follows.",
            "Local annotations were not verified byte-identical to an upstream annotation version.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "connectivity_manifest.json").write_text(json.dumps([m for _, _, m in downloaded], indent=2), encoding="utf-8")
    with (OUT / "path_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": summary, "invalid_paths": len(invalid), "scenes": len(scans)}, indent=2))


if __name__ == "__main__":
    main()
