"""Top-down maps of the navigable area (Habitat navmesh), one per scan floor, drawn under the 2D view.

Continuous (VLN-CE) agents can only stand on the navmesh, so these maps show where an agent could have
gone; discrete runs get them too, since the scans are the same. Needs habitat_sim (CPU only, no rendering),
so run it with the etpnav env's Python:

  /home/bufsgpu/yes/envs/etpnav/bin/python analysis/vln_viz/build_navmaps.py            # every scan
  /home/bufsgpu/yes/envs/etpnav/bin/python analysis/vln_viz/build_navmaps.py --scans 17DRP5sb8fy

Writes cache/navmaps/<scan>_f<id>.png (greyscale: 255 = navigable, north up, MPP metres per pixel) and
cache/navmaps/<scan>.json ({'bounds': [x0, y0, x1, y1] in Matterport x/y, 'mpp', 'floors': [ids]}).
Triangles go to the nearest floor of the scan's navigation graph by height; navmesh islands smaller than
ISLAND_M across (table tops, beds) are left out, since the agent cannot get onto them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parse_trajectories as pt  # noqa: E402

import habitat_sim  # noqa: E402

OUT = HERE / 'cache' / 'navmaps'
SCENES = pt.ROOT.parent.parent / 'data' / 'scene_datasets' / 'mp3d'
MPP = 0.04            # metres per pixel
SUPER = 3             # drawn at MPP / SUPER, then averaged down (smooth edges)
ISLAND_M = 1.0        # minimum island radius kept
MARGIN = 0.5


def build(scan, graphs):
    navmesh = SCENES / scan / f'{scan}.navmesh'
    if not navmesh.is_file():
        return f'{scan}: no navmesh'
    pf = habitat_sim.nav.PathFinder()
    pf.load_nav_mesh(str(navmesh))
    v = np.asarray(pf.build_navmesh_vertices(), dtype=np.float64).reshape(-1, 3, 3)   # habitat x, y (up), z
    centroids = v.mean(1)
    keep = np.array([pf.island_radius(c.astype(np.float32)) >= ISLAND_M for c in centroids])
    v, centroids = v[keep], centroids[keep]
    if not len(v):
        return f'{scan}: empty navmesh'
    floors = graphs[scan].to_json()['floors']
    fz = np.array([f['z'] for f in floors])
    which = np.abs(centroids[:, 1:2] - fz[None, :]).argmin(1)
    near = np.abs(centroids[:, 1] - fz[which]) < 2.5
    xy = np.stack([v[..., 0], -v[..., 2]], -1)                       # Matterport x, y
    x0, y0 = xy[..., 0].min() - MARGIN, xy[..., 1].min() - MARGIN
    x1, y1 = xy[..., 0].max() + MARGIN, xy[..., 1].max() + MARGIN
    w, h = int(np.ceil((x1 - x0) / MPP)), int(np.ceil((y1 - y0) / MPP))
    done = []
    for k, f in enumerate(floors):
        tris = xy[(which == k) & near]
        if not len(tris):
            continue
        img = Image.new('L', (w * SUPER, h * SUPER), 0)
        draw = ImageDraw.Draw(img)
        px = (tris[..., 0] - x0) / MPP * SUPER
        py = (y1 - tris[..., 1]) / MPP * SUPER
        for t in range(len(tris)):
            draw.polygon([(px[t, i], py[t, i]) for i in range(3)], fill=255)
        img.resize((w, h), Image.BOX).save(OUT / f"{scan}_f{f['id']}.png", optimize=True)
        done.append(f['id'])
    (OUT / f'{scan}.json').write_text(json.dumps({'bounds': [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                                                  'mpp': MPP, 'floors': done}))
    return f'{scan}: {len(v)} triangles, floors {done}, {w}x{h} px'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scans', nargs='*', help='scan ids (default: every scan with a navmesh)')
    ap.add_argument('--force', action='store_true', help='rebuild existing maps')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    graphs = pt.GraphCache(pt.DUET_DATA / 'R2R' / 'connectivity_mp3d', pt.SCANS_DIR)
    scans = a.scans or sorted(p.name for p in SCENES.iterdir() if (p / f'{p.name}.navmesh').is_file())
    for scan in scans:
        if not a.force and (OUT / f'{scan}.json').is_file():
            continue
        print(build(scan, graphs), flush=True)


if __name__ == '__main__':
    main()
