"""Interactive VLN navigation and error visualizer for DUET / ScaleVLN predictions.

  python3 analysis/vln_viz/visualizer_app.py                 # released B/16 + H/14, every R2R split
  python3 analysis/vln_viz/visualizer_app.py --preds duet=path/to/detail_val_unseen.json \
          --steplog duet=path/to/steps_val_unseen.jsonl --port 8765

Then open http://127.0.0.1:8765 (from a laptop: ssh -L 8765:127.0.0.1:8765 <server>).

What-if branches (whatif.py) run on the GPU only while the viewer's "What-if · GPU" switch is on; it is off
whenever the server starts. --whatif-gpu picks the GPU, --no-whatif removes the switch.

Continuous-environment runs (VLN-CE: ETPNav on R2R-CE / RxR-CE, logged by [prior works]/ETPNav/tools/
log_etpnav_steps.py) are picked up from ETPNav/data/logs/explorer automatically; see ce_runs.py.

Standard library only (no Streamlit/Plotly install into the pinned scalevln env); any python3 >= 3.8 works.
Episodes, metrics and diagnostics come from parse_trajectories.py; the browser draws the top-down
map, playback, DUET frontier scores, instruction attention and a first-person view from the
Matterport skyboxes.
"""
import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_trajectories as pt  # noqa: E402
import ce_runs  # noqa: E402
import whatif  # noqa: E402

STATIC = Path(__file__).resolve().parent / 'static'
MESHES = Path(__file__).resolve().parent / 'cache' / 'meshes'
MESHES_HD = MESHES.parent / 'meshes_hd'      # one high-detail mesh per floor (build_meshes.py --detail)
NAVMAPS = MESHES.parent / 'navmaps'          # navigable area per floor (build_navmaps.py, habitat navmesh)
EXPRS = pt.DUET_DATA / 'R2R' / 'exprs_map'
SAFE_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
NOTES = Path(__file__).resolve().parent / 'episode_notes.json'


_navmaps = {}


def navmap(scan):
    """Bounds and floors of the scan's navigable-area maps (build_navmaps.py), or None."""
    if scan not in _navmaps:
        f = NAVMAPS / f'{scan}.json'
        _navmaps[scan] = json.loads(f.read_text()) if f.is_file() else None
    return _navmaps[scan]


class Notes:
    """Tags and a free-text note per episode of a run, kept in episode_notes.json:
    {run name: {instr_id: {'tags': [...], 'note': str}}}."""

    def __init__(self, path=NOTES):
        self.path, self.lock = path, threading.Lock()
        self.data = json.loads(path.read_text()) if path.is_file() else {}

    def for_run(self, run):
        tags = {}
        for entries in self.data.values():
            for e in entries.values():
                for t in e.get('tags', []):
                    tags[t] = tags.get(t, 0) + 1
        return {'notes': self.data.get(run, {}), 'vocabulary': sorted(tags, key=lambda t: (-tags[t], t.lower()))}

    def save(self, run, instr_id, tags, note):
        tags = list(dict.fromkeys(str(t).strip()[:40] for t in tags if str(t).strip()))[:20]
        note = str(note or '')[:4000]
        with self.lock:
            entries = self.data.setdefault(run, {})
            if tags or note.strip():
                entries[instr_id] = {'tags': tags, 'note': note}
            else:
                entries.pop(instr_id, None)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            os.replace(tmp, self.path)
        return entries.get(instr_id, {'tags': [], 'note': ''})


def open_browser(url):
    # A remote, headless shell cannot open a window on the user's laptop.
    port = urlparse(url).port
    tunnel = f'For SSH, run ssh -L {port}:127.0.0.1:{port} <server> on your laptop, then open the URL there.'
    if sys.platform.startswith('linux') and not any(os.environ.get(k) for k in ('DISPLAY', 'WAYLAND_DISPLAY', 'BROWSER')):
        print(f'No desktop browser in this session. Open {url}\n{tunnel}', flush=True)
        return
    try:
        if webbrowser.open(url, new=2):
            return
    except webbrowser.Error:
        pass
    print(f'Could not open a browser automatically. Open {url}\n{tunnel}', flush=True)


def discover(split):
    """Released/eval runs under exprs_map/eval: {key: (model, split, predictions, step log or None, settings)}.

    `split` is one split name or 'all' (every detail_<split>.json found). Settings come from the step log's
    meta_<split>.json (log_duet_steps.py): the decision budget and a note such as "zero-shot".
    """
    found = {}
    for f in sorted(EXPRS.glob(f"eval/*/preds/detail_{'*' if split == 'all' else split}.json")):
        model, sp = f.parents[1].name, f.stem[len('detail_'):]
        log = EXPRS / 'steplogs' / model / f'steps_{sp}.jsonl'
        meta = EXPRS / 'steplogs' / model / f'meta_{sp}.json'
        settings = json.loads(meta.read_text()) if meta.is_file() else {}
        found[f'{model}/{sp}'] = (model, sp, f, log if log.is_file() else None, settings)
    return found


def parse_pairs(items, flag):
    out = {}
    for item in items or []:
        name, sep, path = item.partition('=')
        if not sep:
            name, path = Path(item).stem, item
        if not Path(path).is_file():
            sys.exit(f'{flag} {item}: file not found')
        out[name] = Path(path)
    return out


class App:
    def __init__(self, args):
        self.args = args
        preds = parse_pairs(args.preds, '--preds')
        steplogs = parse_pairs(args.steplog, '--steplog')
        if preds:
            runs = {name: (name, None, path, steplogs.get(name)) for name, path in preds.items()}
        else:
            runs = discover(args.split)
            if not runs:
                sys.exit(f'No detail_{args.split}.json under {EXPRS}/eval; pass --preds')
        self.graphs = pt.GraphCache(args.conn_dir, args.scans_dir)
        self.annos, self.subtasks = {}, {}
        self.runs = {}
        self.refresh_lock = threading.Lock()
        for key, spec in runs.items():
            self.runs[key] = self.load_run(key, *spec)
        if not preds and not args.no_ce:
            for key, spec in ce_runs.discover().items():
                self.runs[key] = self.load_ce_run(key, *spec)
        self.scans_dir = Path(args.scans_dir)
        self.mesh_lock = threading.Lock()
        self.mesh_building = {}                  # output file -> lock, so different meshes build in parallel
        self.detail_faces = args.detail_faces
        self.notes = Notes()
        self.whatif = whatif.WhatIf(args.whatif_gpu, allowed=not args.no_whatif)

    def load_run(self, key, model, split, path, log, settings=None):
        args, settings = self.args, settings or {}
        anno = Path(args.anno) if args.anno else pt.default_annotation_for(path)
        if anno not in self.annos:
            self.annos[anno] = pt.load_annotations(anno, args.bboxes)
            f = pt.default_subtasks_for(anno)
            self.subtasks[anno] = pt.load_subtasks(f) if f else {}
        run = pt.Run(key, path, anno, args.conn_dir, args.scans_dir, log,
                     settings.get('max_action_len', args.max_action_len), graphs=self.graphs,
                     annotations=self.annos[anno], subtasks=self.subtasks[anno], model=model, split=split)
        run.note, run.settings = settings.get('note', ''), settings
        s = pt.aggregate(run.rows())
        print(f"  {key:<28} {s['n']:5d} episodes  SR {s.get('sr', '-')}  SPL {s.get('spl', '-')}"
              f"  step log: {'yes' if run.steplog else 'no'}  ({path})", flush=True)
        return run

    def load_ce_run(self, key, model, split, path, meta):
        run = ce_runs.CERun(key, model, split, path, meta, self.graphs)
        s = pt.aggregate(run.rows())
        print(f"  {key:<28} {s['n']:5d} episodes  SR {s.get('sr', '-')}  SPL {s.get('spl', '-')}"
              f"  continuous (VLN-CE)  ({path})", flush=True)
        return run

    def refresh(self):
        """Load runs whose predictions appeared under exprs_map/eval since startup (on page load), so a
        finished log_duet_steps.py run shows up without restarting the server. Continuous runs still being
        logged are reloaded when their episodes file has grown."""
        if self.args.preds:
            return
        with self.refresh_lock:
            new = {k: (self.load_run, spec) for k, spec in discover(self.args.split).items() if k not in self.runs}
            if not self.args.no_ce:
                new.update({k: (self.load_ce_run, spec) for k, spec in ce_runs.discover().items()
                            if k not in self.runs or (getattr(self.runs[k], 'ce', False) and self.runs[k].stale())})
            if new:
                loaded = {}
                for key, (load, spec) in new.items():
                    try:
                        loaded[key] = load(key, *spec)
                    except Exception as e:   # e.g. a file still being written; retried on the next page load
                        print(f'  {key}: not loaded yet ({type(e).__name__}: {e})', flush=True)
                self.runs = {**self.runs, **loaded}   # swap, never mutate: other threads iterate self.runs

    def mesh(self, scan, floor=None):
        """Decimated mesh (gzip) of the whole building, or the high-detail mesh of one floor. Built on first
        request if build_meshes.py has not cached it (a few seconds for the building, up to ~45 s per floor)."""
        f = MESHES / f'{scan}.bin.gz' if floor is None else MESHES_HD / f'{scan}_f{floor}.bin.gz'
        if not f.is_file():
            if floor is not None:
                z = {fl['id']: fl['z'] for fl in self.graphs[scan].to_json()['floors']}.get(floor)
                if z is None:
                    raise KeyError(f'floor {floor}')
            with self.mesh_lock:
                lock = self.mesh_building.setdefault(f, threading.Lock())
            with lock:
                if not f.is_file():
                    import build_meshes   # needs numpy
                    if floor is None:
                        build_meshes.build(scan, self.scans_dir, MESHES)
                    else:
                        build_meshes.build_floor(scan, floor, z, self.scans_dir, MESHES_HD, self.detail_faces)
        return f.read_bytes()

    def meta(self):
        self.refresh()
        return {'runs': [{'name': n, 'model': r.model, 'split': r.split, 'predictions': r.pred_file,
                          'annotations': r.anno_file,
                          'steplog': r.steplog_file, 'max_action_len': r.max_action_len, 'note': r.note,
                          'reference': 'route' if r.route_reference else 'goal',
                          'env': 'continuous' if getattr(r, 'ce', False) else 'discrete',
                          'pair_split': getattr(r, 'pair_split', None),
                          'summary': pt.aggregate(r.rows())} for n, r in self.runs.items()],
                'error_margin': pt.ERROR_MARGIN}

    def episode(self, name, instr_id):
        run = self.runs[name]
        ep = run.episode(instr_id)
        ep['others'] = {}
        for other, r in self.runs.items():
            if other == name or instr_id not in r.ids:
                continue
            if r.split == run.split:
                o = r.episode(instr_id, full=False)
                ep['others'][other] = {'path': o['path'], 'outcome': o['outcome'], 'metrics': o['metrics'],
                                       'failure_mode': o['failure_mode']}
            elif r.split == getattr(run, 'pair_split', None) or getattr(r, 'pair_split', None) == run.split:
                ep['others'][other] = self.cross_route(r, instr_id)     # same instruction, other setting
        ep['reference'] = 'route' if run.route_reference else 'goal'
        ep['whatif'] = self.whatif.for_episode(run, instr_id)
        if getattr(run, 'ce', False):         # graph, goal zone and distances come with the episode
            ep['graph']['navmap'] = navmap(ep['scan'])
            return ep
        g = run.graphs[ep['scan']]
        if ep.get('goal'):
            # REVERIE: success means stopping where the target object is visible, not within 3 m of the path end
            ep['goal_zone'] = ep['goal_vps'] if ep.get('goal_vps') is not None else \
                [vp for vp in g.pos if g.dist(vp, ep['goal']) < pt.ERROR_MARGIN]
            ep['dist_to_goal'] = {vp: round(g.dist(vp, ep['goal']), 2) for vp in g.pos}
        return ep


    def cross_route(self, r, instr_id):
        """Another run of the same instruction in the other setting (discrete graph vs continuous): its route
        with its own positions, which the viewer adds to this episode's points as 'compare' nodes."""
        if getattr(r, 'ce', False):
            o = r.episode(instr_id)
            pts = o['graph']['nodes']
            nodes = {vp: {'pos': pts[vp]['pos'], 'h': pts[vp]['h'], 'floor': pts[vp]['floor']} for vp in o['path']}
        else:
            o = r.episode(instr_id, full=False)
            g = r.graphs[o['scan']]
            nodes = {vp: {'pos': [round(c, 3) for c in g.pos[vp]], 'h': round(g.heights[vp], 3),
                          'floor': g.floor_of.get(vp, 0)} for vp in o['path']}
        ids = {vp: f'cmp:{r.name}:{k}' for k, vp in enumerate(dict.fromkeys(o['path']))}
        return {'path': [ids[vp] for vp in o['path']], 'outcome': o['outcome'], 'metrics': o['metrics'],
                'failure_mode': o['failure_mode'], 'cross': True,
                'nodes': {ids[vp]: dict(n, kind='compare', rot=[]) for vp, n in nodes.items()}}


class Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, fmt, *args):
        pass

    def send(self, body, ctype='application/json', status=200, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(',', ':'), default=float).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status, msg):
        self.send({'error': msg}, status=status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        app = self.app
        try:
            if url.path in ('/', '/index.html'):
                return self.send((STATIC / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path.startswith('/static/'):
                f = (STATIC / url.path[len('/static/'):]).resolve()
                if STATIC not in f.parents or not f.is_file():
                    return self.fail(404, 'not found')
                ctype = 'text/javascript' if f.suffix == '.js' else mimetypes.guess_type(f.name)[0] or 'application/octet-stream'
                return self.send(f.read_bytes(), ctype, headers={'Cache-Control': 'no-cache'})
            if url.path == '/api/meta':
                return self.send(app.meta())
            if url.path == '/api/rows':
                return self.send(app.runs[q['run']].rows())
            if url.path == '/api/episode':
                return self.send(app.episode(q['run'], q['id']))
            if url.path == '/api/points':
                return self.send(app.runs[q['run']].points())
            if url.path == '/api/notes':
                return self.send(app.notes.for_run(q['run']))
            if url.path == '/api/whatif/status':
                return self.send(app.whatif.status())
            if url.path == '/api/whatif/results':
                return self.send(app.whatif.summary(app.runs[q['run']]))
            if url.path == '/api/scan':
                if not SAFE_ID.match(q.get('scan', '')):
                    return self.fail(400, 'bad scan id')
                return self.send(dict(app.graphs[q['scan']].to_json(), navmap=navmap(q['scan'])))
            m = re.match(r'^/api/mesh/([A-Za-z0-9_-]{1,64})(?:/f(\d{1,3}))?$', url.path)
            if m:
                floor = None if m[2] is None else int(m[2])
                cached = MESHES / f'{m[1]}.bin.gz' if floor is None else MESHES_HD / f'{m[1]}_f{floor}.bin.gz'
                if not (app.scans_dir / m[1] / 'house_segmentations' / f'{m[1]}.ply').is_file() and not cached.is_file():
                    return self.fail(404, 'no house mesh for this scan')
                try:
                    body = app.mesh(m[1], floor)
                except ImportError:
                    return self.fail(404, 'mesh not built; run build_meshes.py (needs numpy)')
                return self.send(body, 'application/octet-stream',
                                 headers={'Content-Encoding': 'gzip', 'Cache-Control': 'max-age=86400'})
            m = re.match(r'^/api/ceview/([A-Za-z0-9_-]{1,64})/([A-Za-z0-9_-]{1,64})/([A-Za-z0-9_-]{1,64})/(n\d{1,4})_([0-5])\.jpg$',
                         url.path)
            if m:
                run = app.runs.get(f'{m[1]}/{m[2]}')
                f = run.view_file(m[3], m[4], m[5]) if getattr(run, 'ce', False) else None
                if f is None or not f.is_file():
                    return self.fail(404, 'no view image')
                return self.send(f.read_bytes(), 'image/jpeg', headers={'Cache-Control': 'max-age=86400'})
            m = re.match(r'^/api/navmap/([A-Za-z0-9_-]{1,64})_f(\d{1,3})\.png$', url.path)
            if m:
                f = NAVMAPS / f'{m[1]}_f{m[2]}.png'
                if not f.is_file():
                    return self.fail(404, 'no navigable-area map; run build_navmaps.py')
                return self.send(f.read_bytes(), 'image/png', headers={'Cache-Control': 'max-age=86400'})
            m = re.match(r'^/api/skybox/([A-Za-z0-9_-]+)/([0-9a-f]{32})/([0-5])\.jpg$', url.path)
            if m:
                f = app.scans_dir / m[1] / 'matterport_skybox_images' / f'{m[2]}_skybox{m[3]}_sami.jpg'
                if not f.is_file():
                    return self.fail(404, 'no skybox image')
                return self.send(f.read_bytes(), 'image/jpeg', headers={'Cache-Control': 'max-age=86400'})
            return self.fail(404, 'not found')
        except KeyError as e:
            return self.fail(404, f'unknown {e}')

    def do_POST(self):
        path = urlparse(self.path).path
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        app, run = self.app, self.app.runs.get(req.get('run'))
        try:
            if path == '/api/whatif/switch':
                return self.send(app.whatif.switch(bool(req.get('on'))))
            if path == '/api/whatif/cancel':
                return self.send(app.whatif.cancel())
            if run is None:
                return self.fail(404, 'unknown run')
            if path == '/api/whatif/run':
                items = [(str(i), int(t)) for i, t in req.get('items') or []]
                if req.get('dry_run'):
                    branches, cached, skipped = app.whatif.plan(run, items)
                    return self.send({'branches': len(branches), 'cached': cached, 'skipped': skipped})
                return self.send(app.whatif.submit(run, items, str(req.get('label') or 'what-if')[:120]))
            if path == '/api/notes':
                if str(req.get('id')) not in run.preds:
                    return self.fail(404, 'unknown episode')
                return self.send(app.notes.save(run.name, str(req['id']), req.get('tags') or [], req.get('note')))
        except PermissionError as e:
            return self.fail(409, str(e))
        except ValueError as e:
            return self.fail(400, str(e))
        if path != '/api/export':
            return self.fail(404, 'not found')
        if run is None:
            return self.fail(404, 'unknown run')
        ids = set(req.get('ids') or [])
        notes = app.notes.for_run(run.name)['notes']
        rows = [dict(r, tags='; '.join(notes.get(r['instr_id'], {}).get('tags', [])),
                     note=notes.get(r['instr_id'], {}).get('note', '')) for r in run.rows() if r['instr_id'] in ids]
        fmt = req.get('format', 'csv')
        if fmt not in ('csv', 'json', 'jsonl'):
            return self.fail(400, 'format must be csv, json or jsonl')
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = pt.write_export(run, rows, Path(tmp) / f'export.{fmt}', fmt, full=bool(req.get('full')))
            body = out.read_bytes()
        name = f"{run.name}_{req.get('label') or 'episodes'}_{len(rows)}.{fmt}"
        self.send(body, 'text/csv' if fmt == 'csv' else 'application/json',
                  headers={'Content-Disposition': f'attachment; filename="{re.sub(r"[^A-Za-z0-9_.-]", "_", name)}"'})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preds', nargs='+', metavar='NAME=FILE',
                   help='prediction files; default: every exprs_map/eval/*/preds/detail_<split>.json')
    p.add_argument('--steplog', nargs='+', metavar='NAME=FILE', help='step logs from log_duet_steps.py')
    p.add_argument('--split', default='all', help="split(s) to auto-discover: one name or 'all' (default)")
    p.add_argument('--anno', help='annotation file; default: inferred per prediction file')
    p.add_argument('--bboxes', help='REVERIE BBoxes.json (default: next to the annotations)')
    p.add_argument('--conn-dir', default=str(pt.DUET_DATA / 'R2R' / 'connectivity_mp3d'))
    p.add_argument('--scans-dir', default=str(pt.SCANS_DIR), help='Matterport scans with skyboxes and .house files')
    p.add_argument('--max-action-len', type=int, default=pt.DUET_MAX_ACTION_LEN)
    p.add_argument('--detail-faces', type=int, default=2_000_000,
                   help='triangle budget of the high-detail floor meshes built on request (see build_meshes.py)')
    p.add_argument('--whatif-gpu', default='0', help='GPU (CUDA_VISIBLE_DEVICES) for what-if branches (default 0)')
    p.add_argument('--no-whatif', action='store_true', help='remove the What-if · GPU switch: never use the GPU')
    p.add_argument('--no-ce', action='store_true', help='skip the continuous-environment (VLN-CE) runs')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--open-browser', action='store_true', help='open the visualizer in the default browser')
    p.add_argument('--autoplay', action='store_true', help='show all outcomes and start playing the first episode')
    args = p.parse_args()
    if not 0 <= args.port <= 65535:
        p.error('--port must be between 0 and 65535')

    print('Loading runs ...')
    Handler.app = App(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    url = f'http://{host}:{server.server_port}/'
    if args.autoplay:
        run = next(iter(Handler.app.runs.values()))
        if run.ids:
            url += '#' + urlencode({'run': run.name, 'id': run.ids[0], 'autoplay': '1'})
    print(f'VLN visualizer on {url}  (Ctrl-C to stop)', flush=True)
    if args.open_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
