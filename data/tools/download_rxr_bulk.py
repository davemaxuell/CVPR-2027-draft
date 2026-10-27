#!/usr/bin/env python3
"""Resume the RxR inventory with parallel curl transfers and published MD5 checks.

First run setup_rxr_original.py to create the inventory. This faster transport
uses independent HTTP/1.1 connections and splits large objects into 1 MiB ranges.
Only fully assembled, checksum-verified objects receive their final filenames.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote


def digest(path):
    value = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return base64.b64encode(value.digest()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--connections', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--transfer_timeout', type=int, default=120,
                        help='Maximum seconds per transfer attempt before resuming it')
    parser.add_argument('--stall_timeout', type=int, default=30,
                        help='Retry transfers receiving less than 1 KiB/s for this many seconds')
    parser.add_argument('--retries', type=int, default=2,
                        help='Immediate curl retries before the checksum/resume pass')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=0)
    args = parser.parse_args()
    if min(args.connections, args.batch_size, args.transfer_timeout, args.stall_timeout) < 1 or args.retries < 0:
        parser.error('Connections, batch size, and timeouts must be positive; retries cannot be negative')
    os.umask(0o007)
    data = args.data_root.resolve()
    root, reports, logs = data / 'datasets/RxR', data / 'splits', data / 'logs'
    logs.mkdir(exist_ok=True)
    with (reports / 'rxr_original_download_manifest.jsonl').open() as stream:
        items = [json.loads(line) for line in stream]
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError('Invalid shard selection')
    items = items[args.shard_index::args.shards]
    suffix = '' if args.shards == 1 else '_%02d' % args.shard_index
    status_path = reports / ('rxr_original_download_status%s.json' % suffix)
    status = dict(phase='checking_existing', expected_files=len(items),
                  expected_bytes=sum(x['size'] for x in items), verified_files=0,
                  verified_bytes=0, failed_files=0, source='gs://rxr-data',
                  destination=str(root), connections=args.connections,
                  batch_size=args.batch_size, transfer_timeout_seconds=args.transfer_timeout,
                  stall_timeout_seconds=args.stall_timeout, retries=args.retries)
    start = time.monotonic()
    pending, failures = [], []

    def report(inflight=0):
        status['elapsed_seconds'] = round(time.monotonic() - start, 1)
        status['inflight_bytes'] = inflight
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(status_path)
        print('%s: %d/%d verified; %.2f + %.2f in flight / %.2f GiB; %d failed' % (
            status['phase'], status['verified_files'], len(items), status['verified_bytes']/2**30,
            inflight/2**30, status['expected_bytes']/2**30, status['failed_files']), flush=True)

    report()
    for item in items:
        rel = Path(item['name'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe inventory path')
        path = root / rel
        partial = path.with_name(path.name + '.part')
        probe = path.with_name(path.name + '.probe')
        if probe.exists() and not path.exists() and not partial.exists():
            probe.replace(partial)
        if path.is_file() and path.stat().st_size == item['size'] and digest(path) == item['md5Hash']:
            status['verified_files'] += 1
            status['verified_bytes'] += item['size']
        elif partial.is_file() and partial.stat().st_size == item['size'] and digest(partial) == item['md5Hash']:
            partial.replace(path)
            status['verified_files'] += 1
            status['verified_bytes'] += item['size']
        else:
            pending.append(item)
    status['phase'] = 'downloading'
    report()
    config = logs / ('rxr_active_curl%s.conf' % suffix)
    chunk_root = root / ('.download_chunks' + suffix)
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        for attempt in range(4):
            plans, tasks = [], []
            for item in batch:
                path = root / item['name']
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + '.part')
                have = partial.stat().st_size if partial.exists() else 0
                # A missing partial also has have == 0. Empty upstream objects
                # must still be fetched before checking their published digest.
                if partial.is_file() and have >= item['size']:
                    if have == item['size'] and digest(partial) == item['md5Hash']:
                        partial.replace(path)
                        status['verified_files'] += 1
                        status['verified_bytes'] += item['size']
                        continue
                    partial.unlink()
                    have = 0
                chunks = []
                if item['size'] > 4 * 1024 * 1024:
                    folder = chunk_root / hashlib.sha1(item['name'].encode()).hexdigest()
                    folder.mkdir(parents=True, exist_ok=True)
                    for begin in range(have, item['size'], 1024 * 1024):
                        end = min(begin + 1024 * 1024, item['size']) - 1
                        piece = folder / ('%d-%d.part' % (begin, end))
                        chunks.append((piece, end-begin+1))
                        if not piece.exists() or piece.stat().st_size != end-begin+1:
                            tasks.append((item, piece, '%d-%d' % (begin, end)))
                else:
                    tasks.append((item, partial, None))
                plans.append((item, path, partial, chunks))
            with config.open('w') as stream:
                for number, (item, output, byte_range) in enumerate(tasks):
                    if number:
                        stream.write('next\n')
                    stream.write('http1.1\nfail\nsilent\nshow-error\nretry-all-errors\nconnect-timeout = 30\n')
                    stream.write('retry = %d\nmax-time = %d\nspeed-limit = 1024\nspeed-time = %d\n' %
                                 (args.retries, args.transfer_timeout, args.stall_timeout))
                    stream.write('header = "Accept-Encoding: gzip"\n')
                    url = 'https://storage.googleapis.com/rxr-data/' + quote(item['name'], safe='/')
                    url += '?generation=' + item['generation']
                    stream.write('url = ' + json.dumps(url) + '\n')
                    stream.write('output = ' + json.dumps(str(output)) + '\n')
                    if byte_range:
                        stream.write('range = "' + byte_range + '"\n')
                    else:
                        stream.write('continue-at = "-"\n')
            if tasks:
                print('Batch %d: %d objects, %d transfers, attempt %d' %
                      (offset // args.batch_size + 1, len(plans), len(tasks), attempt + 1), flush=True)
                with (logs / ('rxr_curl_errors%s.log' % suffix)).open('ab') as stderr:
                    proc = subprocess.Popen(['curl', '--parallel', '--parallel-immediate',
                                             '--parallel-max', str(args.connections), '--config', str(config)],
                                            stdout=subprocess.DEVNULL, stderr=stderr)
                    while True:
                        try:
                            result = proc.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            inflight = 0
                            for _, _, partial, chunks in plans:
                                if partial.exists():
                                    inflight += partial.stat().st_size
                                inflight += sum(p.stat().st_size for p, _ in chunks if p.exists())
                            report(inflight)
                if result:
                    print('curl exit %d; checking completed objects before retrying' % result, flush=True)
            retry = []
            for item, path, partial, chunks in plans:
                if chunks:
                    if any(not p.exists() or p.stat().st_size != n for p, n in chunks):
                        retry.append(item)
                        continue
                    with partial.open('ab') as stream:
                        for piece, _ in chunks:
                            with piece.open('rb') as source:
                                shutil.copyfileobj(source, stream, length=1024 * 1024)
                    shutil.rmtree(chunks[0][0].parent)
                if partial.exists() and partial.stat().st_size == item['size'] and digest(partial) == item['md5Hash']:
                    partial.replace(path)
                    status['verified_files'] += 1
                    status['verified_bytes'] += item['size']
                else:
                    if partial.exists() and partial.stat().st_size >= item['size']:
                        partial.unlink()
                    retry.append(item)
            report()
            batch = retry
            if not batch:
                break
        failures.extend(x['name'] for x in batch)
        status['failed_files'] = len(failures)
    status['phase'] = 'failed' if failures else 'complete'
    report()
    (reports / ('rxr_original_download_failures%s.json' % suffix)).write_text(json.dumps(failures, indent=2) + '\n')
    if failures:
        raise SystemExit('Some objects failed; rerun to resume')
    if args.shards == 1:
        link = data / 'datasets/RxR_VLNCE_v0/text_features'
        if link.exists() or link.is_symlink():
            if link.resolve() != (root / 'text_features').resolve():
                raise RuntimeError('Existing feature path conflicts with the RxR mirror')
        else:
            link.symlink_to('../RxR/text_features', target_is_directory=True)
    if chunk_root.exists() and not any(chunk_root.iterdir()):
        chunk_root.rmdir()
    print('COMPLETE: all published RxR file sizes and MD5 checksums verified.', flush=True)


if __name__ == '__main__':
    main()
