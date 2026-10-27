"""Continuous-environment (VLN-CE) runs for the explorer: ETPNav episodes logged by
[prior works]/ETPNav/tools/log_etpnav_steps.py (R2R-CE, RxR-CE).

The explorer works on Matterport navigation graphs. A continuous episode gets its own small graph (CEGraph)
with the same interface, whose "viewpoints" are
  n<k>  ETPNav's map nodes: where decision k was taken (the agent's actual position)
  g<k>  ghost waypoints: predicted, never visited (position as last estimated)
  p<k>  samples of the agent's continuous track (every 0.5 m and at every arrival)
  r<k>  the reference path: the R2R / RxR viewpoints snapped to the navmesh; the last one is the goal
Positions are Matterport coordinates (habitat x, -z, y + camera height), so building meshes, floors and
rooms line up with the discrete runs. Distances to reference points are habitat geodesics logged with the
episode, so parse_trajectories.analyse_episode (frames, sub-tasks, decision review, diagnostics) applies
unchanged. Headline metrics are ETPNav's own, with the VLN-CE definitions (nDTW over the dense track,
SPL against the geodesic start-goal distance).

Episode keys are the discrete instruction ids where they exist (R2R <path_id>_<j>, RxR instruction_id), so
FGR2R / RxR sub-tasks, tags and comparisons with the discrete runs line up.
"""
import json
import math
import re
from pathlib import Path

import parse_trajectories as pt

CE_LOGS = pt.ROOT.parent / 'ETPNav' / 'data' / 'logs' / 'explorer'    # <model>_<task>/episodes_<split>.jsonl
ROWS_CACHE = Path(__file__).resolve().parent / 'cache' / 'ce_rows'
ROWS_VERSION = 4         # bump when rows gain fields
# Habitat cube faces (explorer_env.CUBE_FACES) seen through views.js's skybox math: texture = -R^T world.
CUBE_ROT = [-1, 0, 0, 0, 0, -1, 0, -1, 0]
STUCK_M = 0.25           # a move that changes the agent's position by less than this went nowhere
ARRIVAL_M = 0.05         # the final position counts as the stop node itself within this distance
RXR_DIR = pt.RXR_DIR
R2R_ANNOTATIONS = pt.DUET_DATA / 'R2R' / 'annotations'


def discover():
    """{run key: (model, split, episodes file, meta)} for every logged CE run, e.g. 'etpnav/r2r_ce_val_unseen'."""
    found = {}
    for f in sorted(CE_LOGS.glob('*/episodes_*.jsonl')):
        m = re.match(r'(.+)_(r2r|rxr)$', f.parent.name)
        if not m or not f.stat().st_size:
            continue
        model, task, split = m.group(1), m.group(2), f.stem[len('episodes_'):]
        meta = f.parent / f'meta_{split}.json'
        found[f'{model}/{task}_ce_{split}'] = (model, f'{task}_ce_{split}', f,
                                               json.loads(meta.read_text()) if meta.is_file() else {})
    return found


def _mp3d(p, cam):
    return (p[0], -p[2], p[1] + cam)


class CEGraph:
    """One episode's points with the ScanGraph interface parse_trajectories needs (see module doc)."""

    def __init__(self, rec, scan_graph, floors):
        cam = rec['camera_height']
        self.scan, self.heights = rec['scene'], {}
        self.pos, self.kind, self.geo, self.rot = {}, {}, {}, {}
        n_refs = len(rec['reference_path'])
        clean = lambda row: [x if x is not None else math.inf for x in row[:n_refs]]

        def add(vp, p, kind, geo):
            self.pos[vp], self.kind[vp], self.heights[vp] = _mp3d(p, cam), kind, cam
            self.geo[vp] = clean(geo)
        for vp, p in rec['nodes'].items():
            add(vp, p, 'node', rec['geo']['node'][vp])
            if rec.get('views'):
                self.rot[vp] = CUBE_ROT
        for vp, p in rec['ghosts'].items():
            add(vp, p, 'ghost', rec['geo']['ghost'][vp])
        for j, k in enumerate(rec['samples']):
            add(f'p{j}', rec['track'][k], 'sample', rec['geo']['sample'][j])
        self.refs = [f'r{j}' for j in range(n_refs)]
        for j, p in enumerate(rec['reference_path']):
            add(self.refs[j], p, 'ref', rec['ref_geo'][j])
        self.ref_index = {r: j for j, r in enumerate(self.refs)}
        # ETPNav's map: a move adds the edge front node -> arrival node; a waypoint predicted on top of a
        # visited node links the two; ghosts hang off the nodes that predicted them.
        self.adj = {}
        link = lambda a, b: (self.adj.setdefault(a, {}).__setitem__(b, self.euclid(a, b)),
                             self.adj.setdefault(b, {}).__setitem__(a, self.euclid(a, b)))
        steps = rec['steps']
        for t, s in enumerate(steps):
            if s.get('choice') and f'n{t + 1}' in self.pos:
                link(s['front'], f'n{t + 1}')
            for _, merged in s.get('cands', []):
                if merged and merged.startswith('n') and merged != s['vp'] and merged in self.pos:
                    link(s['vp'], merged)
        self.node_adj = {a: dict(b) for a, b in self.adj.items()}
        for g, fronts in rec['ghost_fronts'].items():
            for f in fronts:
                if f in self.pos and g in self.pos:
                    link(g, f)
        # Floors: the scan's floor slabs (so per-floor meshes match); rooms: the nearest Matterport viewpoint
        # on the same floor, for map nodes and reference points only (enough for room labels and stop_place).
        self.floors = floors
        self.floor_of = {vp: min(range(len(floors)), key=lambda f: abs(floors[f]['z'] - (p[2] - cam)))
                         if floors else 0 for vp, p in self.pos.items()}
        self.region_of, self.rid_of = {}, {}
        sg = scan_graph
        for vp in self.pos:
            if self.kind[vp] not in ('node', 'ref'):
                continue
            x, y, z = self.pos[vp]
            fl = self.floor_of[vp]
            best = min(((math.hypot(q[0] - x, q[1] - y), v) for v, q in sg.pos.items()
                        if sg.floor_of.get(v, 0) == fl), default=None)
            if best and best[0] < 4.0:
                self.region_of[vp], self.rid_of[vp] = sg.region_of.get(best[1]), sg.rid_of.get(best[1])
        self._sp = {}

    def dist(self, a, b):
        if a == b:
            return 0.0
        j = self.ref_index.get(b)
        if j is not None:
            return self.geo[a][j]
        j = self.ref_index.get(a)
        if j is not None:
            return self.geo[b][j]
        return self.euclid(a, b)    # only between non-reference points (back-tracking lengths)

    def euclid(self, a, b):
        return math.dist(self.pos[a], self.pos[b])

    def heading(self, a, b):
        dx, dy = self.pos[b][0] - self.pos[a][0], self.pos[b][1] - self.pos[a][1]
        return math.atan2(dx, dy) % (2 * math.pi)

    def path(self, a, b):
        return [b]

    def node_dist(self, a, b):
        """Shortest walk a -> b over ETPNav's map (map nodes only)."""
        dist, heap = {a: 0.0}, [(0.0, a)]
        import heapq
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                return d
            if d > dist[u]:
                continue
            for v, w in self.node_adj.get(u, {}).items():
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(heap, (d + w, v))
        return self.euclid(a, b)

    def to_json(self, sky=None):
        edges = sorted({tuple(sorted((a, b))) for a in self.adj for b in self.adj[a]})
        return {
            'scan': self.scan, 'ce': True,
            'nodes': {vp: {'pos': [round(c, 3) for c in p], 'h': round(self.heights[vp], 3),
                           'floor': self.floor_of.get(vp, 0), 'region': self.region_of.get(vp),
                           'rid': self.rid_of.get(vp), 'rot': self.rot.get(vp, []), 'kind': self.kind[vp],
                           **({'sky': f'{sky}/{vp}'} if sky and vp in self.rot else {})}
                      for vp, p in self.pos.items()},
            'edges': edges,
            'floors': self.floors,
        }


def ce_decisions(rec, graph):
    """ETPNav's steps as explorer decisions: route = track samples walked, ending at the arrival node.

    Each decision also carries what is specific to the continuous setting:
      cands  the waypoint predictor's output at this node: [[x, y, z] on the floor (Matterport), merged into]
             where merged into is the map node or ghost the prediction was folded into (None: dropped)
      exec   how the low-level controller carried it out: metres walked (the logged track only has points
             where the position changed), and for a move the route length to the aimed waypoint (over the
             map, then straight), how far the agent got from where it decided, and how far from the waypoint
             it ended
    """
    steps, seg, track = rec['steps'], rec['seg_end'], rec['track']
    sample_at = {k: f'p{j}' for j, k in enumerate(rec['samples'])}
    floor = lambda p: [round(p[0], 3), round(-p[2], 3), round(p[1], 3)]
    decisions, passed = [], [steps[0]['vp']]
    for t, s in enumerate(steps):
        lo, hi = seg[t], seg[t + 1]
        walked = [sample_at[k] for k in range(lo, hi) if k in sample_at]
        if s.get('choice') is None:
            route = walked
            if walked and graph.euclid(walked[-1], s['stop_at']) < ARRIVAL_M:
                route = walked[:-1] + [s['stop_at']]
            d = {'at': s['vp'], 'action': 'stop', 'target': route[-1] if route else s['vp'], 'route': route}
        else:
            arrival = f'n{t + 1}'
            route = (walked[:-1] if walked else []) + [arrival]
            d = {'at': s['vp'], 'action': 'local' if s['front'] == s['vp'] else 'remote', 'target': arrival,
                 'route': route, 'revisit': False, 'choice': s['choice']}
            if t + 1 < len(steps):
                d['heading'] = steps[t + 1]['heading']
        ex = {'walked_m': round(sum(math.dist(track[k - 1], track[k]) for k in range(lo, hi)), 3)}
        if s.get('choice') and s['choice'] in graph.pos and d['target'] in graph.pos:
            ex.update(aim_m=round(graph.node_dist(s['vp'], s['front']) + graph.euclid(s['front'], s['choice']), 3),
                      miss_m=round(graph.euclid(d['target'], s['choice']), 3),
                      moved_m=round(graph.euclid(s['vp'], d['target']), 3))
        d['exec'] = ex
        d['cands'] = [[floor(p), merged] for p, merged in s.get('cands', [])]
        d['passed'] = passed
        passed = [s['vp']] + route
        decisions.append(d)
    return decisions


def r2r_instr_ids(split):
    """(path_id, normalised instruction) -> R2R instruction id <path_id>_<j>."""
    f = R2R_ANNOTATIONS / f'R2R_{split}_enc.json'
    out = {}
    if f.is_file():
        for it in json.load(open(f)):
            for j, text in enumerate(it['instructions']):
                out[(str(it['path_id']), ' '.join(text.lower().split()))] = f"{it['path_id']}_{j}"
    return out


class CERun:
    """A logged ETPNav run with the interface visualizer_app expects of parse_trajectories.Run."""

    ce = True
    route_reference = False

    def __init__(self, name, model, split, episodes_file, meta, graphs):
        self.name, self.model, self.split = name, model, split
        self.file = Path(episodes_file)
        self.views_dir = self.file.parent / f"views_{self.file.stem[len('episodes_'):]}"
        self.task = split.split('_ce_')[0]                    # r2r | rxr
        self.base_split = split.split('_ce_')[1]              # val_unseen ...
        self.pair_split = self.base_split if self.task == 'r2r' else f'rxr_{self.base_split}'
        self.route_reference = self.task == 'rxr'
        self.pred_file = str(self.file)
        self.anno_file = str(meta.get('annotations') or
                             f"{'R2R' if self.task == 'r2r' else 'RxR'}_VLNCE ({self.base_split})")
        self.steplog_file = str(self.file)
        self.steplog = True
        self.max_action_len = meta.get('max_action_len', 15 if self.task == 'r2r' else 25)
        self.note = f"{meta.get('note') or 'continuous (VLN-CE)'}" + \
            (f" · {', '.join(meta['languages'])}" if meta.get('languages') else '')
        self.settings = dict(meta, env='continuous')
        self.graphs = graphs
        self.subtasks = self._subtasks()
        self._load()

    def _subtasks(self):
        if self.task == 'r2r':
            f = pt.FGR2R_DIR / f'FGR2R_{self.base_split}.json'
        else:
            f = pt.RXR_SUBTASKS_DIR / f'rxr_{self.base_split}.json'
        return pt.load_subtasks(f) if f.is_file() else {}

    def _load(self):
        """Index the episodes file (byte offsets by instruction id); rows come from the cache when the file
        has not changed."""
        self.size = self.file.stat().st_size
        ids = r2r_instr_ids(self.base_split) if self.task == 'r2r' else {}
        self.offsets, self.episode_ids = {}, {}
        with open(self.file, 'rb') as f:
            pos = 0
            for line in f:
                if line.strip():
                    ep_id = re.search(rb'"episode_id":"([^"]+)"', line).group(1).decode()
                    iid = self._instr_id(line, ids, ep_id)
                    self.offsets[iid] = pos
                    self.episode_ids[iid] = ep_id
                pos += len(line)
        self.ids = list(self.offsets)
        self.preds = {i: True for i in self.ids}
        self.gt = {}
        self._rows = self._points = None

    def _instr_id(self, line, ids, ep_id):
        if self.task == 'rxr':
            m = re.search(rb'"instruction_id":"([^"]*)"', line)
            return m.group(1).decode() if m and m.group(1) else f'ce{ep_id}'
        rec = json.loads(line)
        return ids.get((rec['trajectory_id'], ' '.join(rec['instruction'].lower().split())), f'ce{ep_id}')

    def stale(self):
        return self.file.stat().st_size != self.size

    def record(self, instr_id):
        with open(self.file, 'rb') as f:
            f.seek(self.offsets[instr_id])
            return json.loads(f.readline())

    def graph(self, rec):
        sg = self.graphs[rec['scene']]
        return CEGraph(rec, sg, sg.to_json()['floors'])

    def episode(self, instr_id, full=True):
        rec = self.record(instr_id)
        g = self.graph(rec)
        decisions = ce_decisions(rec, g)
        gt = {'scan': rec['scene'], 'path': g.refs, 'instruction': rec['instruction'],
              'heading': rec['steps'][0]['heading']}
        pred = {'instr_id': instr_id, 'trajectory': None,
                'details': {vp: {'stop_prob': p} for vp, p in (rec['steps'][-1].get('stop_scores') or {}).items()}}
        steplog = {'steps': rec['steps'], 'tokens': rec['tokens']}
        ep = pt.analyse_episode(pred, gt, g, rec['max_steps'], steplog, full=full,
                                subtasks=self.subtasks.get(instr_id), route_reference=self.route_reference,
                                decisions=decisions, ref_near=lambda vp, ref: g.dist(vp, ref) < pt.SUBGOAL_MARGIN)
        self._apply_ce_metrics(ep, rec, g, decisions)
        if full:
            sky = f'/api/ceview/{self.model}/{self.split}/{rec["episode_id"]}'
            ep['graph'] = g.to_json(sky if rec.get('views') else None)
            ep['dist_to_goal'] = {vp: round(g.dist(vp, ep['goal']), 2) for vp in g.pos}
            ep['goal_zone'] = [vp for vp, d in ep['dist_to_goal'].items() if d < pt.ERROR_MARGIN]
            ep['ce'] = {'episode_id': rec['episode_id'], 'language': rec.get('language'),
                        'track': [[round(c, 3) for c in _mp3d(p, 0.0)] for p in rec['track']],
                        'gt_dense': [[round(c, 3) for c in _mp3d(p, 0.0)] for p in rec['gt_locations']],
                        'camera_height': rec['camera_height']}
        else:
            ep['diag']['steplog'] = True
        return ep

    def _apply_ce_metrics(self, ep, rec, g, decisions):
        """ETPNav's own metrics replace the graph-based ones; CE-specific diagnostics."""
        em, m, diag = rec['metrics'], ep['metrics'], ep['diag']
        m.update(ne=em['distance_to_goal'], sr=em['success'], osr=em['oracle_success'], spl=em['spl'],
                 ndtw=em['ndtw'], sdtw=em['sdtw'], tl=round(em['path_length'], 3),
                 sp_len=round(rec['track_goal_dist'][0], 3), collisions=em['collisions'],
                 low_level_steps=em['steps_taken'])
        if bool(em['success']) != (ep['outcome'] == 'success'):
            if em['success']:
                ep['outcome'] = ep['failure_mode'] = 'success'
            else:
                ep['outcome'] = 'step_limit' if len(decisions) >= rec['max_steps'] else 'wrong_stop'
                ep['failure_mode'] = 'passed goal: did not stop there' if em['oracle_success'] else \
                    'never near goal: stopped elsewhere'
        moves = [d for d in decisions if d['action'] != 'stop']
        remote = [d for d in decisions if d['action'] == 'remote']
        steps = rec['steps']
        back = sum(g.node_dist(d['at'], steps[d_t]['front']) for d_t, d in enumerate(decisions) if d['action'] == 'remote')
        misses = [d['exec']['miss_m'] for d in moves if 'miss_m' in d['exec']]
        diag.update(first_off_gt_hop=None, backtrack_hops=len(remote), backtrack_m=round(back, 3),
                    stuck=sum(1 for d in moves if g.euclid(d['at'], d['target']) < STUCK_M),
                    ghosts=len(rec['ghosts']),
                    # controller: how far from the aimed waypoint each move ended (worst move, mean)
                    ctrl_miss_max=round(max(misses), 3) if misses else None,
                    ctrl_miss_mean=round(sum(misses) / len(misses), 3) if misses else None,
                    actions_per_decision=round(em['steps_taken'] / max(1, len(decisions)), 1),
                    # ETPNav's 'collisions' is collisions per logged track point; turn it back into a count
                    collision_count=round(em['collisions'] * len(rec['track'])))
        # stopping: ETPNav stops at the visited node with the highest stop score; was a node inside the
        # success area scored lower?
        last = steps[-1]
        scores = last.get('stop_scores') or {}
        zone = [s for vp, s in scores.items() if vp in g.pos and g.dist(vp, g.refs[-1]) < pt.ERROR_MARGIN]
        diag.update(stop_score=scores.get(last.get('stop_at')),
                    zone_stop_score=max(zone) if zone else None)

    def rows(self):
        if self._rows is None:
            cache = ROWS_CACHE / f"{self.name.replace('/', '__')}.json"
            if cache.is_file():
                c = json.loads(cache.read_text())
                if c.get('size') == self.size and c.get('version') == ROWS_VERSION and c.get('code') == self._code():
                    self._rows, self._points = c['rows'], c['points']
                    return self._rows
            eps = [self.episode(i, full=False) for i in self.ids]
            self._points = {ep['instr_id']: ep['review'] for ep in eps if 'review' in ep}
            self._rows = [pt.summary_row(ep) for ep in eps]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({'size': self.size, 'version': ROWS_VERSION, 'code': self._code(),
                                         'rows': self._rows, 'points': self._points}))
        return self._rows

    def points(self):
        self.rows()
        return self._points

    @staticmethod
    def _code():
        """The analysis code the cached rows came from (parse_trajectories and this module)."""
        return [pt.file_stamp(pt.__file__), pt.file_stamp(__file__)]

    def view_file(self, episode_id, node, face):
        return self.views_dir / episode_id / f'{node}_{face}.jpg'
