"""Independently audit all eligible R2R pair compositions with stdlib Dijkstra."""
from collections import defaultdict
import heapq
import json
import math


def audit(rows, source, connectivity):
    inputs = json.loads(source.read_text())
    by_id = {row['path_id']: row for row in inputs}
    groups = defaultdict(list)
    for row in inputs:
        groups[row['scan']].append(row)
    expected, graphs = {}, {}
    for scan, group in groups.items():
        nodes = json.loads((connectivity / (scan + '_connectivity.json')).read_text())
        positions = {r['image_id']: [r['pose'][i] for i in (3, 7, 11)] for r in nodes if r['included']}
        graph = {}
        for row in nodes:
            if row['image_id'] not in positions:
                continue
            graph[row['image_id']] = {
                nodes[i]['image_id']: math.dist(positions[row['image_id']], positions[nodes[i]['image_id']])
                for i, linked in enumerate(row['unobstructed']) if linked and nodes[i]['included']}
        graphs[scan] = graph
        distances = {}
        for start in {r['path'][-1] for r in group}:
            best, pending = {start: 0.0}, [(0.0, start)]
            while pending:
                distance, node = heapq.heappop(pending)
                if distance != best[node]:
                    continue
                for neighbor, weight in graph[node].items():
                    candidate = distance + weight
                    if candidate <= 3.0 and candidate < best.get(neighbor, math.inf):
                        best[neighbor] = candidate
                        heapq.heappush(pending, (candidate, neighbor))
            distances[start] = best
        for first in group:
            for second in group:
                distance = distances[first['path'][-1]].get(second['path'][0])
                if distance is not None:
                    expected[first['path_id'], second['path_id']] = distance
    actual = set()
    for index, row in enumerate(rows):
        key = row['first_path_id'], row['second_path_id']
        assert key in expected and key not in actual, ('Unexpected or duplicate pair', key)
        actual.add(key)
        first, second = by_id[key[0]], by_id[key[1]]
        assert row['path_id'] == index and row['scan'] == first['scan'] == second['scan'], key
        path = row['path']
        first_prefix, second_suffix = first['path'][:-1], second['path'][1:]
        assert path[:len(first_prefix)] == first_prefix and path[-len(second_suffix):] == second_suffix, key
        connector = path[len(first_prefix):len(path) - len(second_suffix)]
        assert connector[0] == first['path'][-1] and connector[-1] == second['path'][0], key
        graph = graphs[row['scan']]
        length = sum(graph[a][b] for a, b in zip(connector, connector[1:]))
        assert math.isclose(length, expected[key], abs_tol=1e-8), key
        assert row['instructions'] == [x + y for x in first['instructions'] for y in second['instructions']], key
        assert row['heading'] == first['heading'], key
        assert math.isclose(row['distance'], first['distance'] + length + second['distance'], abs_tol=1e-8), key
    assert actual == set(expected), ('Missing eligible pairs', len(set(expected) - actual))
    return dict(passed=True, paths=len(rows), instructions=sum(len(r['instructions']) for r in rows),
                eligible_pairs=len(expected), algorithm='Independent standard-library Dijkstra and full instruction/path composition audit',
                distance_threshold=3.0)
