#!/usr/bin/env python3
"""Mirror RoleVLN into the GitHub working copy and list the checkpoints that go up as release assets.

    python3 -I migration/export_repo.py                       # sync into ../CVPR-2027-draft
    python3 -I migration/export_repo.py --dry-run             # report only
    python3 -I migration/export_repo.py --build-archives      # also pack the explorer logs (tier 2)
    python3 -I migration/export_repo.py --stage               # ... and `git add` the result (commit/push yourself)

What goes where (see MIGRATION.md):
  git      code, docs, results, logs, and the selected/frozen checkpoints (TIER1, every file < 95 MiB)
  release  every other checkpoint of ours + the explorer run logs, listed in migration/release_assets.json
  nothing  datasets, public weights, regenerable caches and frames (DATA_MIGRATION.md says how to get them)

The source tree is never modified. Nested upstream clones are copied as plain files (their .git is skipped);
clean clones under drafts/.../repositories are recorded (URL + commit) instead of copied.
Standard library only.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SRC = Path(__file__).resolve().parents[1]
DEST = SRC.parent / 'CVPR-2027-draft'
ASSETS_DIR = SRC.parent / 'CVPR-2027-draft-assets'
REPO = 'davemaxuell/CVPR-2027-draft'
TAG = 'assets-v1'

MIB = 1 << 20
GIT_LIMIT = 50 * MIB          # larger files never go into git ...
TIER1_LIMIT = 95 * MIB        # ... except TIER1 checkpoints (GitHub rejects > 100 MB)
PART = 1900 * MIB             # release assets must be < 2 GiB

OURS_RUNS = '[OURS]/baseline-v0/runs'
CKPT_SUFFIXES = ('.pt', '.pth', '.safetensors', '.ckpt', '.pth.tar')

# Selected / frozen checkpoints committed to git (paths under OURS_RUNS; a trailing / means the whole folder).
TIER1 = [
    'step2_smolvlm2/train/checkpoints/step_003072/adapter_model.safetensors',     # Step 2 selected SFT policy
    'step3_smolvlm2/frozen_reference/',                                            # Step 3 frozen control
    'step3b_smolvlm2_stop/train/checkpoints/step_000512/adapter_model.safetensors',  # refined 512 (source)
    'step4_smolvlm2_refined512/reference/',                                        # selected policy (STOP bias -0.25)
    'step5_smolvlm2_refined512/frozen/',                                           # both DINO world models + encoder
    'step5_smolvlm2_refined512/fresh/selected_model.pt',
    'step5_smolvlm2_refined512/transfer/selected_model.pt',
] + [f'step8_smolvlm2_refined512_dino_pair_v1/{v}/checkpoints/step_{s:06d}.pt'     # Step 9 shortlisted scorers
     for v, steps in (('null', (128, 1408, 1536)), ('pooled', (512, 1536, 2048))) for s in steps]

# Upstream checkpoints that only sit in our runs folder: re-download instead (MIGRATION.md).
UPSTREAM_CKPTS = ['step5_smolvlm2_refined512/assets/outputs/*/checkpoints/*']

# Explorer outputs packed into release archives: (asset prefix, folder relative to SRC, compress).
ARCHIVES = [
    ('scalevln_exprs_map', '[prior works]/ScaleVLN/VLN-DUET/datasets/R2R/exprs_map', True),
    ('etpnav_explorer_logs', '[prior works]/ETPNav/data/logs/explorer', False),
]

TOP_FILES = {'README.md', 'MIGRATION.md', 'DATA_MIGRATION.md', 'TRAINING_STATE.md'}
TOP_DIRS = {'migration', 'data', 'drafts', '[OURS]', '[prior works]'}

SKIP_NAMES = {'.git', '__pycache__', '.ipynb_checkpoints', '.mypy_cache', '.pytest_cache', '.venvs', '.venv', 'venv'}
SKIP_GLOBS = ['*.pyc', '*.pid', '*.swp', '.nfs*']

# Whole folders left out (relative to SRC).
PRUNE = {
    'data/.downloads', 'data/datasets/.downloads', 'data/img_features', 'data/mattersim', 'data/scene_datasets',
    'data/processed', 'data/logs', 'data/hime_task_data', 'data/connectivity',
    '[prior works]/ScaleVLN/VLN-DUET/datasets', '[prior works]/ScaleVLN/VLN-DUET-RVR/datasets',
    '[prior works]/ScaleVLN/runs', '[prior works]/ScaleVLN/Matterport3DSimulator/build',
    '[prior works]/ScaleVLN/analysis/vln_viz/cache/meshes', '[prior works]/ScaleVLN/analysis/vln_viz/cache/meshes_hd',
    '[prior works]/ScaleVLN/analysis/vln_viz/cache/rows', '[prior works]/ScaleVLN/analysis/vln_viz/cache/ce_rows',
    '[prior works]/ETPNav/data',
}
# Files kept inside pruned folders.
KEEP_IN_PRUNED = [
    '[prior works]/ScaleVLN/VLN-DUET/datasets/.gitkeep',
    '[prior works]/ScaleVLN/VLN-DUET-RVR/datasets/R2R',          # relative symlinks
    '[prior works]/ScaleVLN/VLN-DUET-RVR/datasets/pretrained',
    '[prior works]/ScaleVLN/runs/readiness.json',
    '[prior works]/ScaleVLN/runs/verification.json',
    '[prior works]/ScaleVLN/runs/data_audit.json',
    '[prior works]/ETPNav/data/scene_datasets',                  # symlinks, made relative
    '[prior works]/ETPNav/data/datasets/RxR_VLNCE_v0',
    '[prior works]/ETPNav/data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/process_instrs_to_bert_idx.py',
    '[prior works]/ETPNav/data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/vocab.txt',
    '[prior works]/ETPNav/data/datasets/RxR_VLNCE_v0_enc_xlmr/xlm_tokenize.py',
    '[prior works]/ETPNav/data/logs/eval_results/**',
    '[prior works]/ETPNav/data/logs/*.log',
    '[prior works]/ETPNav/data/logs/running_log/**',
]
# data/datasets keeps only small provenance files at data/datasets/<name>/<file> plus its layout symlinks.
DATASET_DOC_GLOBS = ['README*.md', 'LICENSE*', 'source_manifest.json', 'generation_manifest.json',
                     'required_scenes.json', 'envdrop_missing_episode_ids.json', 'haps_extraction_report.json']
SKIP_FILES = {'data/splits/rxr_original_download_manifest.jsonl'}    # 93 MB download provenance
SECRET_FILES = {'drafts/Research_Investigation_2026-10-22/repositories/05/CAST/cast/baselines/convoi/gamma_api.txt'}  # upstream-committed API key
FRAME_SUFFIXES = ('.jpg', '.jpeg')                                     # simulator frame dumps in runs/
# Reference clones under drafts/ keep their code and text, not their simulator assets or data blobs.
DRAFT_REPO_BINARY = ('.png', '.jpg', '.jpeg', '.gif', '.msh', '.obj', '.stl', '.dae', '.mtl', '.glb', '.ply',
                     '.mp4', '.npz', '.npy', '.pkl', '.h5', '.hdf5', '.pt', '.pth', '.bin', '.zip', '.tar', '.gz')

DRAFT_REPOS = 'drafts/Research_Investigation_2026-10-22/repositories'
GENERATED = {'migration/release_assets.json', 'migration/export_report.json', 'migration/hardcoded_paths.txt',
             f'{DRAFT_REPOS}/REPOSITORIES.json', '.gitignore'}


def rel(p):
    return PurePosixPath(p.relative_to(SRC).as_posix())


def under(r, prefix):
    s, p = str(r), prefix.rstrip('/')
    return s == p or s.startswith(p + '/')


def is_ckpt(name):
    return name.endswith(CKPT_SUFFIXES)


def tier1(r):
    if not under(r, OURS_RUNS):
        return False
    sub = str(r)[len(OURS_RUNS) + 1:]
    return any(sub.startswith(t) if t.endswith('/') else sub == t for t in TIER1)


def asset_name(r):
    return str(r)[len(OURS_RUNS) + 1:].replace('/', '--')


def git(cwd, *args, check=True):
    return subprocess.run(['git', *args], cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def draft_repo_record(path):
    """URL, commit and cleanliness of a cloned reference repo under drafts/."""
    try:
        url = git(path, 'remote', 'get-url', 'origin', check=False)
        head = git(path, 'rev-parse', 'HEAD')
        dirty = bool(git(path, 'status', '--porcelain', '--untracked-files=no'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return {'path': str(rel(path)), 'url': url, 'commit': head, 'dirty': dirty}


class Plan:
    def __init__(self):
        self.git = {}          # rel -> source Path (file) or link target str (symlink)
        self.links = {}
        self.assets = []       # checkpoint release assets
        self.skipped = {}      # reason -> [count, bytes]
        self.too_big = []      # unclassified large files (reported, never silently dropped)
        self.draft_repos = []

    def skip(self, reason, size=0):
        c = self.skipped.setdefault(reason, [0, 0])
        c[0] += 1
        c[1] += size


def keep_in_pruned(r):
    return any(fnmatch.fnmatch(str(r), g) or str(r) == g for g in KEEP_IN_PRUNED)


def link_target(p, r):
    """Symlink target for the repo: absolute targets inside SRC become relative, others are kept as they are."""
    t = os.readlink(p)
    if os.path.isabs(t):
        try:
            inside = Path(t).relative_to(SRC)
        except ValueError:
            return t
        return os.path.relpath(SRC / inside, (SRC / r).parent)
    return t


def classify_file(plan, p, r, st):
    name, size = p.name, st.st_size
    if str(r) in SKIP_FILES:
        return plan.skip('download provenance (93 MB)', size)
    if str(r) in SECRET_FILES:
        return plan.skip('contains a credential (upstream file)', size)
    if under(r, 'data/datasets'):
        if len(r.parts) == 4 and any(fnmatch.fnmatch(name, g) for g in DATASET_DOC_GLOBS) and size < 2 * MIB:
            plan.git[r] = p
        else:
            plan.skip('dataset payload', size)
        return
    if under(r, OURS_RUNS) and (is_ckpt(name) or name.endswith('.bin')):
        sub = str(r)[len(OURS_RUNS) + 1:]
        if any(fnmatch.fnmatch(sub, g) for g in UPSTREAM_CKPTS):
            return plan.skip('upstream DINO-WM checkpoints (re-download)', size)
        if tier1(r):
            if size > TIER1_LIMIT:
                raise SystemExit(f'TIER1 file over the git limit: {r} ({size / MIB:.0f} MiB)')
            plan.git[r] = p
        else:
            plan.assets.append({'name': asset_name(r), 'path': str(r), 'size': size, 'mtime': st.st_mtime,
                                'kind': 'checkpoint'})
        return
    if under(r, DRAFT_REPOS) and (name.lower().endswith(DRAFT_REPO_BINARY) or size > 5 * MIB):
        return plan.skip('assets/binaries inside reference clones under drafts/', size)
    if under(r, OURS_RUNS) and name.lower().endswith(FRAME_SUFFIXES):
        return plan.skip('simulator frame dumps in runs/ (jpg)', size)
    if size > GIT_LIMIT:
        plan.too_big.append((str(r), size))
        return plan.skip('over 50 MiB, not a checkpoint', size)
    plan.git[r] = p


def walk(plan):
    for top in sorted(SRC.iterdir()):
        r = rel(top)
        if top.is_dir() and not top.is_symlink():
            if top.name not in TOP_DIRS:
                plan.skip(f'top-level folder not exported: {top.name}')
                continue
        elif top.name not in TOP_FILES:
            plan.skip(f'top-level file not exported: {top.name}')
            continue
        if top.is_file():
            plan.git[r] = top
            continue
        stack = [top]
        while stack:
            d = stack.pop()
            dr = rel(d)
            if str(dr).startswith(DRAFT_REPOS + '/') and len(dr.parts) == 5 and (d / '.git').exists():
                rec = draft_repo_record(d)
                if rec and not rec['dirty']:
                    plan.draft_repos.append(rec)
                    plan.skip('clean reference clones under drafts/ (recorded, re-clone)')
                    continue
                if rec:
                    plan.draft_repos.append(rec)
            pruned = any(under(dr, x) for x in PRUNE)
            for e in sorted(os.scandir(d), key=lambda e: e.name):
                p = Path(e.path)
                r = rel(p)
                if e.name in SKIP_NAMES or any(fnmatch.fnmatch(e.name, g) for g in SKIP_GLOBS):
                    continue
                inside_prune = pruned or any(under(r, x) for x in PRUNE)
                if e.is_symlink():
                    if inside_prune and not keep_in_pruned(r):
                        continue
                    if under(r, 'data/datasets') and len(r.parts) > 4:
                        continue
                    plan.links[r] = link_target(p, r)
                elif e.is_dir():
                    if under(r, 'data/datasets') and len(r.parts) >= 4:
                        plan.skip('dataset payload')
                        continue
                    if inside_prune and not any(k.startswith(str(r) + '/') for k in KEEP_IN_PRUNED):
                        if not keep_in_pruned(r):
                            plan.skip('excluded folders (datasets, public weights, caches, smoke runs)')
                            continue
                    stack.append(p)
                elif e.is_file():
                    st = e.stat()
                    if inside_prune and not keep_in_pruned(r):
                        plan.skip('excluded folders (datasets, public weights, caches, smoke runs)', st.st_size)
                        continue
                    classify_file(plan, p, r, st)


def sha256(path, cache):
    st = path.stat()
    key = str(path)
    hit = cache.get(key)
    if hit and hit['size'] == st.st_size and hit['mtime'] == st.st_mtime:
        return hit['sha256']
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * MIB), b''):
            h.update(block)
    cache[key] = {'size': st.st_size, 'mtime': st.st_mtime, 'sha256': h.hexdigest()}
    return cache[key]['sha256']


def build_archives(out_dir):
    """Pack the explorer outputs and split them into < 2 GiB parts; returns their manifest entries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for prefix, folder, compress in ARCHIVES:
        src = SRC / folder
        tar_path = out_dir / f'{prefix}.tar{".gz" if compress else ""}'
        for old in out_dir.glob(f'{prefix}.tar*'):
            old.unlink()
        print(f'packing {folder} -> {tar_path.name}', flush=True)
        with tarfile.open(tar_path, 'w:gz' if compress else 'w') as tar:
            tar.add(src, arcname=src.name)
        size = tar_path.stat().st_size
        parts = []
        with open(tar_path, 'rb') as f:
            i = 0
            while True:
                block = f.read(PART)
                if not block:
                    break
                part = out_dir / f'{tar_path.name}.part{i:02d}'
                part.write_bytes(block)
                parts.append(part)
                i += 1
        tar_path.unlink()
        for part in parts:
            entries.append({'name': part.name, 'path': None, 'size': part.stat().st_size, 'kind': 'archive-part',
                            'archive': tar_path.name, 'extract_to': str(PurePosixPath(folder).parent),
                            'archive_size': size})
    return entries


def link_assets(assets, out_dir):
    """Hardlink each checkpoint asset into out_dir under its asset name (no extra disk), so that folder holds the
    complete tier-2 set: upload it, rsync it, or pass it to release_assets.py download --from-dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in assets:
        src, dst = Path(e['source']), out_dir / e['name']
        if dst.exists() and os.path.samefile(src, dst):
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError as err:
            print(f'cannot hardlink {src} ({err}); release_assets.py upload reads the source path instead')


def mirror(plan, dest, dry):
    wanted = set(map(str, plan.git)) | set(map(str, plan.links)) | GENERATED
    copied = removed = 0
    if not dry:
        dest.mkdir(parents=True, exist_ok=True)
    for r, p in plan.git.items():
        d = dest / r
        st = p.stat()
        if d.is_file() and not d.is_symlink():
            dst = d.stat()
            if dst.st_size == st.st_size and int(dst.st_mtime) == int(st.st_mtime):
                continue
        copied += 1
        if not dry:
            if d.is_symlink() or d.is_dir():
                shutil.rmtree(d) if d.is_dir() and not d.is_symlink() else d.unlink()
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d)
    for r, target in plan.links.items():
        d = dest / r
        if d.is_symlink() and os.readlink(d) == target:
            continue
        copied += 1
        if not dry:
            if d.is_symlink() or d.is_file():
                d.unlink()
            elif d.is_dir():
                shutil.rmtree(d)
            d.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(target, d)
    if dest.exists():
        for root, dirs, files in os.walk(dest, topdown=True):
            rr = Path(root).relative_to(dest).as_posix()
            if rr == '.':
                dirs[:] = [x for x in dirs if x != '.git']
            for name in files + [x for x in dirs if (Path(root) / x).is_symlink()]:
                r = name if rr == '.' else f'{rr}/{name}'
                if r not in wanted:
                    removed += 1
                    if not dry:
                        (Path(root) / name).unlink()
    return copied, removed


def hardcoded_paths(dest):
    """Code and config files (not run records or logs) that hardcode this server's paths."""
    pats = ['/data/team_a/dave-workspace/RoleVLN', '/data/shared/hf_cache', '/home/bufsgpu']
    code = ('.py', '.sh', '.yaml', '.yml', '.cfg', '.toml', '.ini', '.json')
    out = subprocess.run(['grep', '-rlIZ', '--exclude-dir=.git', '-e', pats[0], '-e', pats[1], '-e', pats[2], '.'],
                         cwd=dest, capture_output=True, text=True).stdout.split('\0')
    files = [f[2:] for f in out if f.endswith(code)]
    return sorted(f for f in files if not f.startswith(('drafts/', 'data/splits/', 'migration/', OURS_RUNS + '/'))
                  and '/runs/' not in f and '/logs/' not in f and '/.backups/' not in f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', type=Path, default=DEST)
    ap.add_argument('--assets-dir', type=Path, default=ASSETS_DIR, help='where --build-archives writes its parts')
    ap.add_argument('--build-archives', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--stage', action='store_true', help='git add -A -f in --dest (it holds exactly the export)')
    a = ap.parse_args()
    dest = a.dest.resolve()
    if dest == SRC or SRC in dest.parents or dest in SRC.parents:
        raise SystemExit(f'--dest must be outside the source tree: {dest}')

    plan = Plan()
    walk(plan)
    git_bytes = sum(p.stat().st_size for p in plan.git.values())
    copied, removed = mirror(plan, a.dest, a.dry_run)

    manifest_path = a.dest / 'migration/release_assets.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    cache = {e['source']: {'size': e['size'], 'mtime': e.get('mtime'), 'sha256': e['sha256']}
             for e in old.get('assets', []) if e.get('source') and e.get('sha256')}
    archive_entries = [e for e in old.get('assets', []) if e['kind'] == 'archive-part']
    if a.build_archives and not a.dry_run:
        archive_entries = build_archives(a.assets_dir)
        for e in archive_entries:
            e['source'] = str(a.assets_dir / e['name'])
    assets = []
    if not a.dry_run:
        for e in plan.assets:
            src = SRC / e['path']
            e['source'] = str(src)
            e['sha256'] = sha256(src, cache)
            assets.append(e)
        for e in archive_entries:
            src = Path(e['source'])
            if src.is_file():
                e['sha256'] = sha256(src, cache)
                e['mtime'] = src.stat().st_mtime
            assets.append(e)
        link_assets([e for e in assets if e['kind'] == 'checkpoint'], a.assets_dir)
        names = [e['name'] for e in assets]
        assert len(names) == len(set(names)), 'duplicate release asset names'

        manifest = {'repo': REPO, 'tag': TAG, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                    'note': 'Tier-2 files: our checkpoints not in git, and the explorer logs. '
                            'migration/release_assets.py upload | download | verify.',
                    'assets': assets}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
        if plan.draft_repos:
            (a.dest / DRAFT_REPOS).mkdir(parents=True, exist_ok=True)
            (a.dest / DRAFT_REPOS / 'REPOSITORIES.json').write_text(json.dumps(
                {'note': 'Reference clones: copied without .git and without their binary assets (images, meshes, '
                         'weights, files > 5 MiB). Clean clones are not copied at all. To get a full upstream copy, '
                         'clone url at commit; dirty=true means the local snapshot deleted or changed upstream files.',
                 'repositories': sorted(plan.draft_repos, key=lambda x: x['path'])}, indent=1) + '\n')
        shutil.copy2(SRC / 'migration/repo.gitignore', a.dest / '.gitignore')
        (a.dest / 'migration/hardcoded_paths.txt').write_text('\n'.join(hardcoded_paths(a.dest)) + '\n')

    big = sorted(((str(r), p.stat().st_size) for r, p in plan.git.items()), key=lambda x: -x[1])[:15]
    report = {
        'updated_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SRC), 'dest': str(a.dest),
        'git_files': len(plan.git), 'git_symlinks': len(plan.links), 'git_bytes': git_bytes,
        'tier1_checkpoints': sorted(str(r) for r in plan.git if tier1(r) and is_ckpt(r.name)),
        'release_checkpoints': len(plan.assets), 'release_checkpoint_bytes': sum(e['size'] for e in plan.assets),
        'largest_git_files': big,
        'skipped': {k: {'files': v[0], 'bytes': v[1]} for k, v in sorted(plan.skipped.items())},
        'over_50MiB_not_exported': plan.too_big,
        'draft_repos_recorded': len([x for x in plan.draft_repos if not x['dirty']]),
        'draft_repos_dirty_copied': [x['path'] for x in plan.draft_repos if x['dirty']],
    }
    if not a.dry_run:
        (a.dest / 'migration/export_report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('largest_git_files',)}, indent=1))
    print('largest files in git:')
    for r, s in big:
        print(f'  {s / MIB:7.1f} MiB  {r}')
    print(f'{"would copy" if a.dry_run else "copied"} {copied}, {"would remove" if a.dry_run else "removed"} {removed}')
    if a.stage and not a.dry_run:
        # -f: the working copy holds exactly the export, but nested upstream .gitignore files and our own
        # .gitignore (which keeps future checkpoints out) would otherwise hide TIER1 checkpoints and run records.
        git(a.dest, 'add', '-A', '-f')
        print(git(a.dest, 'status', '--short') and f'staged: {len(git(a.dest, "status", "--short").splitlines())} changes'
              or 'nothing to commit')


if __name__ == '__main__':
    sys.exit(main())
