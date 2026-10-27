#!/usr/bin/env python3
"""Download the public gs://rxr-data release with resumable, MD5-verified files.

Run from any directory. Defaults to the data/ directory containing this script.
The inventory and progress are saved under splits/; RxR-CE shares the resulting
text_features directory via a relative symlink. Requires the requests package.
"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from urllib.parse import quote
import requests

HOST = 'https://storage.googleapis.com'
PREFIXES = [''] + [kind + '/rxr_' + split + '/' for kind, splits in [
    ('text_features', ['test_challenge', 'test_standard', 'val_seen', 'val_unseen', 'train']),
    ('pose_traces', ['val_seen', 'val_unseen', 'train'])] for split in splits]


def write_json(path, obj):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def inventory(prefix):
    params = {'prefix': prefix, 'maxResults': 1000,
              'fields': 'items(name,size,md5Hash,crc32c,generation),nextPageToken'}
    if not prefix:
        params['delimiter'] = '/'
    items = []
    with requests.Session() as session:
        while True:
            for attempt in range(8):
                try:
                    response = session.get(HOST + '/storage/v1/b/rxr-data/o', params=params, timeout=(15, 90))
                    response.raise_for_status()
                    page = response.json()
                    break
                except requests.RequestException:
                    if attempt == 7:
                        raise
                    time.sleep(min(15, 2 ** attempt))
            for item in page.get('items', []):
                if not item['name'].endswith(('/', '_$folder$')):
                    item['size'] = int(item['size'])
                    items.append(item)
            if 'nextPageToken' not in page:
                break
            params['pageToken'] = page['nextPageToken']
    print('Listed %s: %d files' % (prefix or '(root)', len(items)), flush=True)
    return items


def file_digest(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest


def download(session, root, item):
    rel = Path(item['name'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe path: ' + str(rel))
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    size, expected = item['size'], item['md5Hash']
    encode = lambda digest: base64.b64encode(digest.digest()).decode()
    if path.is_file() and path.stat().st_size == size and encode(file_digest(path)) == expected:
        return 'existing'
    partial = path.with_name(path.name + '.part')
    for attempt in range(10):
        try:
            have = partial.stat().st_size if partial.exists() else 0
            if have > size:
                partial.unlink()
                have = 0
            digest = file_digest(partial) if have else hashlib.md5()
            if have == size:
                if encode(digest) == expected:
                    partial.replace(path)
                    return 'downloaded'
                partial.unlink()
                have, digest = 0, hashlib.md5()
            headers = {'Accept-Encoding': 'gzip'}
            if have:
                headers['Range'] = 'bytes=%d-' % have
            url = HOST + '/rxr-data/' + quote(item['name'], safe='/')
            with session.get(url, params={'generation': item['generation']}, headers=headers,
                             stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                if have and response.status_code != 206:
                    have, digest = 0, hashlib.md5()
                if have and not response.headers.get('Content-Range', '').startswith('bytes %d-' % have):
                    raise ValueError('Incorrect range response')
                with partial.open('ab' if have else 'wb') as stream:
                    while True:
                        block = response.raw.read(1024 * 1024, decode_content=False)
                        if not block:
                            break
                        stream.write(block)
                        digest.update(block)
            if partial.stat().st_size != size:
                raise ValueError('Incomplete download')
            if encode(digest) != expected:
                partial.unlink()
                raise ValueError('MD5 mismatch')
            partial.replace(path)
            return 'downloaded'
        except Exception:
            if attempt == 9:
                raise
            time.sleep(min(15, 2 ** attempt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--refresh_manifest', action='store_true')
    args = parser.parse_args()
    os.umask(0o007)
    root, reports = args.data_root / 'datasets/RxR', args.data_root / 'splits'
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    manifest = reports / 'rxr_original_download_manifest.jsonl'
    if manifest.exists() and not args.refresh_manifest:
        with manifest.open() as stream:
            items = [json.loads(line) for line in stream]
    else:
        with cf.ThreadPoolExecutor(max_workers=9) as pool:
            items = [item for group in pool.map(inventory, PREFIXES) for item in group]
        if len({item['name'] for item in items}) != len(items):
            raise ValueError('Duplicate object names')
        temp = manifest.with_suffix('.tmp')
        with temp.open('w') as stream:
            for item in items:
                stream.write(json.dumps(item, separators=(',', ':')) + '\n')
        temp.replace(manifest)
    stats = {'phase': 'downloading', 'expected_files': len(items),
             'expected_bytes': sum(x['size'] for x in items), 'verified_files': 0,
             'verified_bytes': 0, 'downloaded_files': 0, 'existing_files': 0,
             'failed_files': 0, 'source': 'gs://rxr-data', 'destination': str(root)}
    lock, done, work = threading.Lock(), threading.Event(), queue.Queue(args.workers * 4)
    errors = []
    start = time.monotonic()
    status = reports / 'rxr_original_download_status.json'

    def monitor():
        while True:
            with lock:
                snapshot = dict(stats)
            snapshot['elapsed_seconds'] = round(time.monotonic() - start, 1)
            write_json(status, snapshot)
            print('RxR %d/%d files; %.2f/%.2f GiB; %d failures' % (
                snapshot['verified_files'], len(items), snapshot['verified_bytes'] / 2**30,
                stats['expected_bytes'] / 2**30, snapshot['failed_files']), flush=True)
            if done.wait(30):
                return

    def worker():
        with requests.Session() as session:
            while True:
                item = work.get()
                if item is None:
                    work.task_done()
                    return
                try:
                    outcome = download(session, root, item)
                    with lock:
                        stats['verified_files'] += 1
                        stats['verified_bytes'] += item['size']
                        stats[outcome + '_files'] += 1
                except Exception as exc:
                    with lock:
                        stats['failed_files'] += 1
                        errors.append({'name': item['name'], 'error': str(exc)})
                finally:
                    work.task_done()

    reporter = threading.Thread(target=monitor, daemon=True)
    reporter.start()
    workers = [threading.Thread(target=worker) for _ in range(args.workers)]
    for thread in workers:
        thread.start()
    for item in items:
        work.put(item)
    for _ in workers:
        work.put(None)
    work.join()
    for thread in workers:
        thread.join()
    done.set()
    reporter.join()
    stats.update(phase='failed' if errors else 'complete', elapsed_seconds=round(time.monotonic() - start, 1))
    write_json(status, stats)
    write_json(reports / 'rxr_original_download_failures.json', errors)
    if errors:
        raise SystemExit('%d downloads failed; rerun to resume' % len(errors))
    link = args.data_root / 'datasets/RxR_VLNCE_v0/text_features'
    target = root / 'text_features'
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError('Existing text_features path points elsewhere')
    else:
        link.symlink_to('../RxR/text_features', target_is_directory=True)
    print('COMPLETE: published MD5 checksums verified for every RxR object.', flush=True)


if __name__ == '__main__':
    main()
