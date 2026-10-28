#!/usr/bin/env python3
"""Pack what the public repo must not hold (Matterport-derived renders, traces and image indexes) for a private
GitHub release, and list it in migration/private_assets.json.

    # old server
    python3 -I migration/private_archives.py                 # pack into ../CVPR-2027-private-assets
    python3 -I migration/private_archives.py --dry-run       # sizes only
    python3 -I migration/release_assets.py --manifest migration/private_assets.json upload
    # new machine, from the repo root (token with Contents: read on the private repo)
    python3 -I migration/release_assets.py --manifest migration/private_assets.json download
    python3 -I migration/release_assets.py --manifest migration/private_assets.json verify

Keep that repo private: these files come from Matterport scans. Each folder becomes one tar stream cut into PART
pieces in one pass, plus a <name>.tar.sha256 list of every file in it (verify checks them after unpacking);
single files go up as they are. The out dir holds the complete set (single files hardlinked), so it can also be
copied to a disk and used with download --from-dir. Folders whose files did not change are not repacked.
Standard library only.
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_repo import PART, SRC, flat_name, link_assets, rel  # noqa: E402

REPO = 'davemaxuell/CVPR-2027-private'
TAG = 'private-v1'
MANIFEST = SRC / 'migration/private_assets.json'
OUT_DIR = SRC.parent / 'CVPR-2027-private-assets'
PROC = 'data/processed/baseline_v0'
# (archive name, folder relative to SRC)
FOLDERS = [
    ('scorer_labels_smolvlm2_refined512_v1', f'{PROC}/scorer_labels_smolvlm2_refined512_v1'),     # Step 6
    ('dagger_groupA_round1_smolvlm2', f'{PROC}/dagger_groupA_round1_smolvlm2'),
    ('dagger_groupA_round2_smolvlm2_stop2m', f'{PROC}/dagger_groupA_round2_smolvlm2_stop2m'),
    ('wm_visits_groupA_smolvlm2_refined512_v1', f'{PROC}/wm_visits_groupA_smolvlm2_refined512_v1'),  # Step 4
    ('wm_groupA_smolvlm2_refined512_v1', f'{PROC}/wm_groupA_smolvlm2_refined512_v1'),
    ('expert_groupC_600', f'{PROC}/expert_groupC_600'),                                            # SFT val renders
    ('etpnav_explorer', '[prior works]/ETPNav/data/logs/explorer'),                                # cube-face renders
]
FILES = [                                                                                          # NWM image index
    '[OURS]/baseline-v0/runs/step5_smolvlm2_refined512/data_index/images.sqlite',
    '[OURS]/baseline-v0/runs/step5_smolvlm2_refined512/data_index/train_samples.npy',
]
BLOCK = 8 << 20


def tree(src):
    """Sorted (path, size, mtime) of every file under src."""
    out = []
    for d, dirs, files in os.walk(src):
        dirs.sort()
        for n in sorted(files):
            p = Path(d) / n
            if not p.is_symlink():
                st = p.stat()
                out.append((p.relative_to(src.parent).as_posix(), st.st_size, int(st.st_mtime)))
    return out


def signature(files):
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


class Parts:
    """Write sink that cuts one stream into PART-sized files and hashes each piece and the whole."""

    def __init__(self, out_dir, base):
        self.out_dir, self.base, self.done, self.f = out_dir, base, [], None
        self.whole, self.total = hashlib.sha256(), 0

    def write(self, b):
        mv = memoryview(b)
        while mv:
            if self.f is None or self.n == PART:
                self._next()
            k = min(len(mv), PART - self.n)
            self.f.write(mv[:k])
            self.h.update(mv[:k])
            self.whole.update(mv[:k])
            self.n += k
            self.total += k
            mv = mv[k:]
        return len(b)

    def _next(self):
        self._finish()
        self.path = self.out_dir / f'{self.base}.part{len(self.done):02d}.tmp'
        self.f, self.h, self.n = open(self.path, 'wb'), hashlib.sha256(), 0

    def _finish(self):
        if self.f:
            self.f.close()
            self.done.append((self.path, self.n, self.h.hexdigest()))
            self.f = None

    def close(self):
        self._finish()
        for old in self.out_dir.glob(f'{self.base}.part*'):
            if not old.name.endswith('.tmp'):
                old.unlink()
        parts = []
        for tmp, size, digest in self.done:
            final = tmp.with_name(tmp.name[:-len('.tmp')])
            os.replace(tmp, final)
            parts.append((final, size, digest))
        return parts


class Hashing:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        return b


def pack(name, folder, files, out_dir):
    src = SRC / folder
    base = f'{name}.tar'
    sink, sums = Parts(out_dir, base), []
    with tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT, bufsize=BLOCK) as tar:
        seen = set()
        for path, size, _ in files:
            for parent in reversed(PurePosixPath(path).parents[:-1]):     # directories first, once
                if str(parent) not in seen:
                    seen.add(str(parent))
                    tar.add(str(src.parent / parent), arcname=str(parent), recursive=False)
            info = tar.gettarinfo(str(src.parent / path), arcname=path)
            if info.size != size:
                raise SystemExit(f'{path} changed while packing')
            with open(src.parent / path, 'rb') as f:
                reader = Hashing(f)
                tar.addfile(info, reader)
            sums.append(f'{reader.h.hexdigest()}  {path}\n')
    parts = sink.close()
    sums_path = out_dir / f'{base}.sha256'
    sums_path.write_text(''.join(sums))
    extract_to = str(PurePosixPath(folder).parent)
    entries = [{'name': p.name, 'path': None, 'size': size, 'kind': 'archive-part', 'archive': base,
                'extract_to': extract_to, 'archive_size': sink.total, 'archive_sha256': sink.whole.hexdigest(),
                'sha256': digest, 'source': str(p)} for p, size, digest in parts]
    entries.append({'name': sums_path.name, 'path': None, 'size': sums_path.stat().st_size, 'kind': 'archive-sums',
                    'archive': base, 'extract_to': extract_to, 'files': len(sums),
                    'sha256': hashlib.sha256(sums_path.read_bytes()).hexdigest(), 'source': str(sums_path)})
    return entries


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(BLOCK):
            h.update(b)
    return h.hexdigest()


def reusable(old, base, sig):
    """The previous run's entries for this archive if its files are unchanged and its parts are still there."""
    es = [e for e in old if e.get('archive') == base]
    ok = es and all(e.get('tree') == sig and Path(e['source']).is_file() and
                    Path(e['source']).stat().st_size == e['size'] for e in es)
    return es if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    old = json.loads(MANIFEST.read_text())['assets'] if MANIFEST.is_file() else []
    a.out_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, folder in FOLDERS:
        files = tree(SRC / folder)
        size = sum(s for _, s, _ in files)
        sig = signature(files)
        prev = reusable(old, f'{name}.tar', sig)
        state = 'unchanged' if prev else 'packing'
        print(f'{name}: {len(files):,} files, {size / 2**30:.2f} GiB ({state})', flush=True)
        if a.dry_run:
            continue
        entries = prev or pack(name, folder, files, a.out_dir)
        for e in entries:
            e['tree'] = sig
        assets += entries
    for f in FILES:
        p = SRC / f
        st = p.stat()
        print(f'{f}: {st.st_size / 2**20:.0f} MiB', flush=True)
        if a.dry_run:
            continue
        cached = next((e for e in old if e.get('path') == f and e['size'] == st.st_size and e['mtime'] == st.st_mtime),
                      None)
        assets.append({'name': 'file--' + flat_name(rel(p)), 'path': f, 'size': st.st_size, 'mtime': st.st_mtime,
                       'kind': 'file', 'sha256': cached['sha256'] if cached else file_sha256(p), 'source': str(p)})
    if a.dry_run:
        return
    link_assets([e for e in assets if e['kind'] == 'file'], a.out_dir)    # out_dir = the complete set (--from-dir)
    names = [e['name'] for e in assets]
    assert len(names) == len(set(names)), 'duplicate asset names'
    assert all(e['size'] < 2 << 30 for e in assets), 'asset over 2 GiB'
    MANIFEST.write_text(json.dumps({
        'repo': REPO, 'tag': TAG, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'note': 'PRIVATE release: Matterport-derived files that must not go into the public repo. Archive parts are '
                'unpacked at extract_to; <name>.tar.sha256 lists every file for verify. '
                'migration/release_assets.py --manifest migration/private_assets.json upload | download | verify.',
        'assets': assets}, indent=1) + '\n')
    up = sum(e['size'] for e in assets)
    print(f'wrote {MANIFEST.relative_to(SRC)}: {len(assets)} assets, {up / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
