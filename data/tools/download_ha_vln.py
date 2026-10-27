#!/usr/bin/env python3
"""Mirror the complete public HA-VLN 2.0 dataset at pinned source revisions."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request
from datetime import datetime, timezone

DATA = Path(__file__).resolve().parents[1]
ROOT = DATA / 'datasets/HA-VLN'
PLAN = DATA / 'splits/ha_vln_download_plan.json'
STATUS = DATA / 'splits/ha_vln_download_status.json'
HF_REV = '2632eeac7d8a83570bd9ddbde0116fa1be3d76ff'
GH_REV = '4e1d1fe4ceebee7253d3ba53c04468a8bc2c16c8'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def status(phase, **fields):
    write(STATUS, dict(dataset='HA-VLN 2.0', phase=phase, updated_at=now(),
                       downloads_running=phase == 'downloading', pid=os.getpid(), **fields))
    print(now(), phase, flush=True)


def metadata(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'RoleVLN-dataset-mirror/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def make_plan():
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    hf = metadata('https://huggingface.co/api/datasets/fly1113/HA-VLN/revision/' + HF_REV + '?blobs=true')
    gh = metadata('https://api.github.com/repos/UWMILab/HA-VLN/git/trees/' + GH_REV + '?recursive=1')
    assert hf['sha'] == HF_REV and gh['sha'] == GH_REV and not gh.get('truncated')
    assets = []
    for item in hf['siblings']:
        name = item['rfilename']
        assets.append(dict(source='huggingface', source_path=name,
            path=str((ROOT / 'original/huggingface' / name).relative_to(DATA)),
            url='https://huggingface.co/datasets/fly1113/HA-VLN/resolve/' + HF_REV + '/' + name,
            expected_bytes=item['size'], expected_sha256=item.get('lfs', {}).get('sha256'),
            expected_git_blob_sha1=None if item.get('lfs') else item['blobId']))
    for item in gh['tree']:
        name = item['path']
        if item['type'] != 'blob' or not (name.startswith('Data/') or name in
                ('LICENSE', 'README.md', 'scripts/hf_resources.json')):
            continue
        assets.append(dict(source='github', source_path=name,
            path=str((ROOT / 'original/github' / name).relative_to(DATA)),
            url='https://raw.githubusercontent.com/UWMILab/HA-VLN/' + GH_REV + '/' + name,
            expected_bytes=item['size'], expected_sha256=None, expected_git_blob_sha1=item['sha']))
    result = dict(dataset='HA-VLN 2.0', created_at=now(),
        scope='All public HF dataset files, all GitHub Data files, and the released CMA checkpoint; shared MP3D scenes reused.',
        hf_repo='fly1113/HA-VLN', hf_revision=HF_REV,
        github_repo='UWMILab/HA-VLN', github_revision=GH_REV,
        expected_download_bytes=sum(a['expected_bytes'] for a in assets), assets=assets)
    write(PLAN, result)
    return result


def verify(asset):
    path = DATA / asset['path']
    assert path.is_file() and path.stat().st_size == asset['expected_bytes'], 'Missing/wrong size: ' + str(path)
    sha = hashlib.sha256()
    git = hashlib.sha1(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(block)
            git.update(block)
    if asset['expected_sha256']:
        assert sha.hexdigest() == asset['expected_sha256'], 'SHA-256 mismatch: ' + str(path)
    if asset['expected_git_blob_sha1']:
        assert git.hexdigest() == asset['expected_git_blob_sha1'], 'Git hash mismatch: ' + str(path)
    return dict(url=asset['url'], bytes=path.stat().st_size, sha256=sha.hexdigest(),
                git_blob_sha1=git.hexdigest(), upstream_checksum_verified=True, checked_at=now())


def fetch_github(asset):
    path = DATA / asset['path']
    if path.exists():
        verify(asset)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
        '--continue-at', '-', '--retry', '5', '--retry-delay', '3',
        '--connect-timeout', '30', '--output', str(partial), asset['url']], check=True)
    assert partial.stat().st_size == asset['expected_bytes'], 'Download size mismatch: ' + str(path)
    partial.replace(path)
    verify(asset)


def main():
    os.umask(0o007)
    ROOT.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(DATA).free > 30 * 1024**3, 'Need at least 30 GiB free'
    with (DATA / 'splits/ha_vln_download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        release = make_plan()
        try:
            status('downloading', started_at=now(), total_files=len(release['assets']),
                   expected_download_bytes=release['expected_download_bytes'])
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=release['hf_repo'], repo_type='dataset', revision=HF_REV,
                              local_dir=ROOT / 'original/huggingface', token=False, max_workers=8)
            github = [a for a in release['assets'] if a['source'] == 'github']
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(fetch_github, github))
            status('verifying_downloads', expected_download_bytes=release['expected_download_bytes'])
            receipts = {a['path']: verify(a) for a in release['assets']}
            write(ROOT / 'source_manifest.json', dict(
                **{k: v for k, v in release.items() if k != 'assets'}, assets=receipts))
            write(DATA / 'splits/ha_vln_download_receipts.json', receipts)
            status('extracting_and_verifying', downloaded_bytes=release['expected_download_bytes'])
            subprocess.run([sys.executable, str(DATA / 'tools/complete_ha_vln.py')], check=True)
            report = json.loads((DATA / 'splits/ha_vln_completion_report.json').read_text())
            assert report['passed'], 'HA-VLN verification did not pass'
            status('complete', downloads_verified=True, finished_at=now(),
                   downloaded_bytes=release['expected_download_bytes'],
                   report='splits/ha_vln_completion_report.json')
        except BaseException as error:
            status('failed', error=str(error))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
