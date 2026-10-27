#!/usr/bin/env python3
"""Tier-2 files (our checkpoints that are not in git, the third-party checkpoints, and two explorer step logs over
100 MB) as GitHub release assets.

    # old server: upload everything in migration/release_assets.json (token needs Contents: read and write)
    python3 -I migration/release_assets.py upload
    # new machine, from the repo root (public repo: press Enter at the token prompt; private: Contents: read)
    python3 -I migration/release_assets.py download                        # everything (~23 GiB)
    python3 -I migration/release_assets.py download --kind checkpoint      # only checkpoints
    python3 -I migration/release_assets.py download --match 'step5_*'      # by asset name
    python3 -I migration/release_assets.py download --from-dir /mnt/disk/CVPR-2027-draft-assets   # no GitHub
    python3 -I migration/release_assets.py verify

The token comes from GITHUB_TOKEN or GH_TOKEN, else a hidden prompt; it is never written anywhere. Downloads
from a public repo work without one (empty prompt), using the public release download links.
Uploads skip assets already on the release with the right size. Downloads skip files whose sha256 already
matches and put every file at its original path (archive parts, if any, are unpacked where they came from; files
over GitHub's 2 GiB limit went up as .partNN pieces and are joined back).
Standard library only.
"""
import argparse
import fnmatch
import getpass
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'migration/release_assets.json'
ASSETS_DIR = ROOT.parent / 'CVPR-2027-draft-assets'      # where export_repo.py stages the files on the old server
DOWNLOADS = ROOT / '.release_downloads'                   # archive parts on the new machine (git-ignored)
API = 'https://api.github.com'
UPLOADS = 'https://uploads.github.com'
MIB = 1 << 20


def token(optional=False):
    t = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if t:
        return t
    hint = ' (Enter for none: public repo)' if optional else ''
    t = getpass.getpass(f'GitHub token{hint}, input hidden: ').strip() if sys.stdin.isatty() else ''
    if not t and not optional:
        raise SystemExit('a token is needed: set GITHUB_TOKEN')
    return t


class GitHub:
    def __init__(self, repo, tok, api=API, uploads=UPLOADS):
        self.repo, self.tok, self.api, self.uploads = repo, tok, api.rstrip('/'), uploads.rstrip('/')

    def request(self, method, url, body=None, headers=None, timeout=60):
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header('Accept', 'application/vnd.github+json')
        req.add_header('X-GitHub-Api-Version', '2022-11-28')
        if self.tok:
            req.add_unredirected_header('Authorization', f'Bearer {self.tok}')   # not sent on to the CDN redirect
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        return urllib.request.urlopen(req, timeout=timeout)

    def json(self, method, path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        hdr = {'Content-Type': 'application/json'} if body else {}
        with self.request(method, f'{self.api}{path}', body, hdr) as r:
            data = r.read()
        return json.loads(data) if data else None

    def release(self, tag, create=False):
        try:
            return self.json('GET', f'/repos/{self.repo}/releases/tags/{tag}')
        except urllib.error.HTTPError as e:
            if e.code != 404 or not create:
                raise
        return self.json('POST', f'/repos/{self.repo}/releases', {
            'tag_name': tag, 'name': tag, 'draft': False, 'prerelease': False,
            'body': 'Tier-2 files: checkpoints not in git and explorer run logs. '
                    'See MIGRATION.md and migration/release_assets.json.'})

    def assets(self, release_id):
        out, page = [], 1
        while True:
            batch = self.json('GET', f'/repos/{self.repo}/releases/{release_id}/assets?per_page=100&page={page}')
            out += batch
            if len(batch) < 100:
                return out
            page += 1

    def delete_asset(self, asset_id):
        self.json('DELETE', f'/repos/{self.repo}/releases/assets/{asset_id}')

    def upload(self, release_id, name, path):
        size = path.stat().st_size
        url = f'{self.uploads}/repos/{self.repo}/releases/{release_id}/assets?name={urllib.parse.quote(name)}'
        with open(path, 'rb') as f:
            body = Progress(f, size, name)
            with self.request('POST', url, body, {'Content-Type': 'application/octet-stream',
                                                  'Content-Length': str(size)}, timeout=600) as r:
                return json.loads(r.read())

    def download(self, asset, dest, name, size):
        if self.tok:
            url = f'{self.api}/repos/{self.repo}/releases/assets/{asset["id"]}'
        else:   # public repo: the release download link is not rate-limited like the API
            url = asset['browser_download_url']
        with self.request('GET', url, headers={'Accept': 'application/octet-stream'}, timeout=600) as r:
            copy_stream(r, dest, name, size)


class Progress:
    """File wrapper that prints transfer progress as http.client reads it."""
    def __init__(self, f, size, name):
        self.f, self.size, self.name, self.done, self.t0, self.last = f, size, name, 0, time.time(), 0

    def read(self, n=-1):
        block = self.f.read(8 * MIB if n is None or n < 0 else n)
        self.done += len(block)
        report(self.name, self.done, self.size, self.t0, self)
        return block


def report(name, done, size, t0, state):
    now = time.time()
    if now - state.last >= 15 or done >= size:
        state.last = now
        rate = done / max(now - t0, 1e-6) / MIB
        print(f'  {name}: {done / MIB:,.0f}/{size / MIB:,.0f} MiB  {rate:.1f} MiB/s', flush=True)


def copy_stream(src, dest, name, size):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.partial')
    state = type('S', (), {'last': 0})()
    t0, done = time.time(), 0
    with open(tmp, 'wb') as out:
        while True:
            block = src.read(8 * MIB)
            if not block:
                break
            out.write(block)
            done += len(block)
            report(name, done, size, t0, state)
    return tmp


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * MIB), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest(args):
    m = json.loads(MANIFEST.read_text())
    assets = m['assets']
    if args.kind:
        assets = [a for a in assets if a['kind'] == args.kind]
    if args.match:
        assets = [a for a in assets if fnmatch.fnmatch(a['name'], args.match)]
    names = {a['name'] for a in assets}
    assets += [a for a in m['assets'] if a.get('of') in names and a['name'] not in names]   # a split file's parts
    return m, assets


def present(a):
    t = local_target(a)
    return t.is_file() and t.stat().st_size == a['size'] and sha256(t) == a['sha256']


def local_target(a):
    return ROOT / a['path'] if a.get('path') else DOWNLOADS / a['name']     # checkpoints and files: own path


def upload_source(a, assets_dir):
    for p in (assets_dir / a['name'], Path(a.get('source') or '/nonexistent'),
              ROOT / a['path'] if a.get('path') else None):
        if p and p.is_file() and p.stat().st_size == a['size']:
            return p
    return None


def with_retries(fn, what, tries=4):
    for i in range(tries):
        try:
            return fn()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404, 422):
                raise
            if i == tries - 1:
                raise
            wait = 10 * 2 ** i
            print(f'  {what}: {e}; retrying in {wait}s', flush=True)
            time.sleep(wait)


def cmd_upload(args):
    m, assets = load_manifest(args)
    assets = [a for a in assets if not a.get('split')]       # these go up as their file-part pieces
    missing = [a['name'] for a in assets if not upload_source(a, args.assets_dir)]
    if missing:
        raise SystemExit(f'{len(missing)} assets have no local file of the right size, e.g. {missing[:3]}; '
                         'rerun migration/export_repo.py (and --build-archives for the logs)')
    total = sum(a['size'] for a in assets)
    print(f'{len(assets)} assets, {total / 2**30:.2f} GiB -> {m["repo"]} release {m["tag"]}')
    if args.dry_run:
        return
    gh = GitHub(m['repo'], token(), args.api, args.uploads)
    rel = gh.release(m['tag'], create=True)
    remote = {x['name']: x for x in gh.assets(rel['id'])}
    for i, a in enumerate(assets, 1):
        have = remote.get(a['name'])
        if have and have['size'] == a['size'] and have.get('state') == 'uploaded':
            continue
        if have:
            gh.delete_asset(have['id'])
        src = upload_source(a, args.assets_dir)
        print(f'[{i}/{len(assets)}] {a["name"]} ({a["size"] / MIB:,.0f} MiB)', flush=True)

        def attempt():
            try:
                return gh.upload(rel['id'], a['name'], src)
            except (urllib.error.URLError, OSError):
                stale = {x['name']: x for x in gh.assets(rel['id'])}.get(a['name'])
                if stale:                                   # a half-finished asset blocks the name
                    gh.delete_asset(stale['id'])
                raise
        with_retries(attempt, a['name'])
    print('upload complete')


def extract_archives(entries, keep_parts):
    by_archive = {}
    for a in entries:
        if a['kind'] == 'archive-part':
            by_archive.setdefault(a['archive'], []).append(a)
    for name, parts in sorted(by_archive.items()):
        parts.sort(key=lambda a: a['name'])
        files = [DOWNLOADS / p['name'] for p in parts]
        if not all(f.is_file() for f in files):
            print(f'{name}: parts missing, not unpacked')
            continue
        dest = ROOT / parts[0]['extract_to']
        dest.mkdir(parents=True, exist_ok=True)
        print(f'unpacking {name} into {parts[0]["extract_to"]}/', flush=True)
        with Concat(files) as stream, tarfile.open(fileobj=stream, mode='r|*') as tar:
            for member in tar:
                target = (dest / member.name).resolve()
                if dest.resolve() not in target.parents and target != dest.resolve():
                    raise SystemExit(f'unsafe path in {name}: {member.name}')
                if member.issym() or member.islnk():
                    continue
                tar.extract(member, dest, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
        if not keep_parts:
            for f in files:
                f.unlink()


def join_parts(entries, keep_parts):
    """Rebuild each split file at its own path from its downloaded .partNN pieces; returns the failures."""
    failed = []
    for a in entries:
        if not a.get('split') or present(a):
            continue
        parts = sorted((p for p in entries if p.get('of') == a['name']), key=lambda p: p['name'])
        files = [DOWNLOADS / p['name'] for p in parts]
        if not files or not all(f.is_file() for f in files):
            failed.append(f'{a["name"]}: parts missing, not joined')
            continue
        t = local_target(a)
        print(f'joining {len(files)} parts -> {a["path"]}', flush=True)
        with Concat(files) as stream:
            tmp = copy_stream(stream, t, a['name'], a['size'])
        if sha256(tmp) != a['sha256']:
            tmp.unlink()
            failed.append(f'{a["name"]}: sha256 mismatch after joining')
            continue
        os.replace(tmp, t)
        if not keep_parts:
            for f in files:
                f.unlink()
    return failed


class Concat:
    """Read several files as one stream (the archive parts)."""
    def __init__(self, files):
        self.files, self.i, self.f = list(files), 0, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.f:
            self.f.close()

    def read(self, n=-1):
        out = b''
        while self.i < len(self.files) and (n < 0 or len(out) < n):
            if self.f is None:
                self.f = open(self.files[self.i], 'rb')
            block = self.f.read(-1 if n < 0 else n - len(out))
            if not block:
                self.f.close()
                self.f, self.i = None, self.i + 1
                continue
            out += block
        return out


def cmd_download(args):
    m, assets = load_manifest(args)
    whole_ok = {a['name']: present(a) for a in assets if a.get('split')}
    todo = []
    for a in assets:
        if a.get('split') or whole_ok.get(a.get('of')) or present(a):   # split files: joined from parts below
            continue
        todo.append(a)
    print(f'{sum(not a.get("split") for a in assets) - len(todo)} already present, {len(todo)} to fetch '
          f'({sum(a["size"] for a in todo) / 2**30:.2f} GiB)')
    gh, remote = None, {}
    if todo and not args.from_dir:
        gh = GitHub(m['repo'], token(optional=True), args.api, args.uploads)
        try:
            rel = gh.release(m['tag'])
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise SystemExit(f'release {m["tag"]} of {m["repo"]} not reachable (HTTP {e.code}): not uploaded yet, '
                                 'or the repo is private and the token is missing / lacks Contents: read')
            raise
        remote = {x['name']: x for x in gh.assets(rel['id'])}
    bad = []
    for i, a in enumerate(todo, 1):
        t = local_target(a)
        print(f'[{i}/{len(todo)}] {a["name"]} -> {t.relative_to(ROOT)}', flush=True)
        if args.from_dir:
            src = Path(args.from_dir) / a['name']
            with open(src, 'rb') as f:
                tmp = copy_stream(f, t, a['name'], a['size'])
        else:
            if a['name'] not in remote:
                bad.append(f'{a["name"]}: not on the release')
                continue
            tmp = t.with_name(t.name + '.partial')
            with_retries(lambda: gh.download(remote[a['name']], t, a['name'], a['size']), a['name'])
        if sha256(tmp) != a['sha256']:
            tmp.unlink()
            bad.append(f'{a["name"]}: sha256 mismatch')
            continue
        os.replace(tmp, t)
    bad += join_parts(assets, args.keep_parts)
    extract_archives(assets, args.keep_parts)
    if bad:
        raise SystemExit('failed:\n  ' + '\n  '.join(bad))
    print('download complete')


def cmd_verify(args):
    _, assets = load_manifest(args)
    missing = bad = 0
    for a in assets:
        if not a.get('path'):
            continue
        t = local_target(a)
        if not t.is_file():
            missing += 1
            print('missing ', a['path'])
        elif sha256(t) != a['sha256']:
            bad += 1
            print('MISMATCH', a['path'])
    n = sum(bool(a.get('path')) for a in assets)
    print(f'{n - missing - bad}/{n} files OK, {missing} missing, {bad} mismatched')
    return 1 if missing or bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['upload', 'download', 'verify'])
    ap.add_argument('--kind', choices=['checkpoint', 'file', 'archive-part', 'file-part'])
    ap.add_argument('--match', help='only assets whose name matches this glob')
    ap.add_argument('--assets-dir', type=Path, default=ASSETS_DIR, help='upload: staged files (export_repo.py)')
    ap.add_argument('--from-dir', help='download: copy from a local folder of assets instead of GitHub')
    ap.add_argument('--keep-parts', action='store_true', help='download: keep archive parts / file parts after unpacking or joining')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--api', default=API, help=argparse.SUPPRESS)
    ap.add_argument('--uploads', default=UPLOADS, help=argparse.SUPPRESS)
    args = ap.parse_args()
    return {'upload': cmd_upload, 'download': cmd_download, 'verify': cmd_verify}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
