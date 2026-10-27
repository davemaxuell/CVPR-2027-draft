"""Pinned CE terminal nDTW with an exact, linear-memory DTW backend.

Execute the upstream NDTW.update_metric method on the COMPLETE actual path.
The backend implements the configured exact DTW option; no FastDTW is used.
Consecutive identical positions and normalization remain upstream behavior.
"""
import ast
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from numba import njit

from .common import RUN, read, sha


@njit(cache=True)
def exact_distance(a, b):
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0.0
    for point in a:
        current = np.full(len(b) + 1, np.inf)
        for j in range(1, len(b) + 1):
            distance = np.sqrt(np.sum((point - b[j - 1]) ** 2))
            current[j] = distance + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[-1]


def dtw_backend(a, b, dist):
    # The upstream method supplies Euclidean distance; the compiled backend uses it exactly.
    return (float(exact_distance(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))),)


@lru_cache(None)
def upstream_update():
    provenance = read(RUN / 'assets/evaluator_source.json')
    path = RUN / 'assets' / provenance['file']
    assert sha(path) == provenance['sha256']
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NDTW')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update_metric')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    scope = {'np': np, 'euclidean_distance': lambda a, b: np.linalg.norm(np.asarray(b) - np.asarray(a))}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    return scope['update_metric']


def terminal_ndtw(positions, reference, success_distance=3.0):
    points, target = np.asarray(positions, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    assert points.ndim == target.ndim == 2 and points.shape[1] == target.shape[1] == 3
    assert len(points) and len(target) and np.isfinite(points).all() and np.isfinite(target).all()
    unique = [points[0].tolist()]
    for point in points[1:]:
        if point.tolist() != unique[-1]:
            unique.append(point.tolist())
    context = SimpleNamespace(
        locations=unique[:-1], gt_locations=target.tolist(), dtw_func=dtw_backend,
        _config=SimpleNamespace(SUCCESS_DISTANCE=success_distance),
        _sim=SimpleNamespace(get_agent_state=lambda: SimpleNamespace(position=np.array(unique[-1]))))
    upstream_update()(context)
    assert np.isfinite(context._metric) and 0 <= context._metric <= 1
    return float(context._metric)
