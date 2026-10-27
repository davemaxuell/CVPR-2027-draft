#!/usr/bin/env python3
"""Supervise disjoint RxR downloads, then validate the complete dataset layout."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--connections', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Objects per worker batch; larger batches reduce waits for stragglers')
    args = parser.parse_args()
    if min(args.shards, args.connections, args.batch_size) < 1:
        parser.error('Shards, connections, and batch size must be positive')
    os.umask(0o007)
    data = Path(__file__).resolve().parents[1]
    logs, reports = data/'logs', data/'splits'
    logs.mkdir(exist_ok=True)
    lock_handle = (reports/'rxr_download.lock').open('a')
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another RxR supervisor is already running.')
    expected_files = expected_bytes = 0
    with (reports/'rxr_original_download_manifest.jsonl').open() as stream:
        for line in stream:
            item=json.loads(line)
            expected_files += 1
            expected_bytes += item['size']
    status_path = reports/'rxr_original_download_status.json'
    handles, processes = [], []
    start = time.monotonic()
    for index in range(args.shards):
        old = reports/('rxr_original_download_status_%02d.json' % index)
        if old.exists():
            old.unlink()
        handle = (logs/('rxr_download_%02d.log' % index)).open('ab')
        handles.append(handle)
        processes.append(subprocess.Popen([
            sys.executable, '-u', str(data/'tools/download_rxr_bulk.py'),
            '--connections', str(args.connections), '--batch_size', str(args.batch_size),
            '--shards', str(args.shards), '--shard_index', str(index),
        ], stdout=handle, stderr=subprocess.STDOUT))
    initial_bytes = None
    while True:
        rows = []
        for index in range(args.shards):
            path = reports/('rxr_original_download_status_%02d.json' % index)
            if path.exists():
                rows.append(json.loads(path.read_text()))
        status = {'phase': 'downloading', 'source': 'gs://rxr-data',
                  'destination': str(data/'datasets/RxR'), 'supervisor_pid': os.getpid(),
                  'shards': args.shards, 'connections_per_shard': args.connections,
                  'batch_size': args.batch_size, 'elapsed_seconds': round(time.monotonic()-start, 1),
                  'expected_files': expected_files, 'expected_bytes': expected_bytes,
                  'service_unit': 'rolevln-rxr-download.service'}
        for key in ['verified_files','verified_bytes','inflight_bytes','failed_files']:
            status[key] = sum(row.get(key, 0) for row in rows)
        current = status['verified_bytes'] + status['inflight_bytes']
        if initial_bytes is None and len(rows) == args.shards and all(row['phase']!='checking_existing' for row in rows):
            initial_bytes = current
            initial_time = time.monotonic()
        if initial_bytes is not None:
            rate = max(0, current-initial_bytes) / max(1, time.monotonic()-initial_time)
            status['average_mib_per_second'] = round(rate/2**20, 2)
            if rate > 0:
                status['estimated_seconds_remaining'] = round(max(0, expected_bytes-current)/rate)
        return_codes = [proc.poll() for proc in processes]
        status['failed_workers'] = [index for index, code in enumerate(return_codes)
                                    if code is not None and code != 0]
        if status['failed_workers']:
            # File failure counts cannot expose an unexpected worker exit.
            status['phase'] = 'worker_failed'
            status.pop('estimated_seconds_remaining', None)
        elif len(rows) < args.shards or any(row['phase'] == 'checking_existing' for row in rows):
            status['phase'] = 'checking_existing'
        finished = all(code is not None for code in return_codes)
        if finished:
            status['phase'] = 'failed' if any(proc.returncode for proc in processes) else 'verifying'
        save(status_path, status)
        print('%s: %d/%d verified, %.2f + %.2f in flight / %.2f GiB, %.2f MiB/s' % (
            status['phase'], status['verified_files'], status['expected_files'],
            status['verified_bytes']/2**30, status['inflight_bytes']/2**30,
            status['expected_bytes']/2**30, status.get('average_mib_per_second',0)), flush=True)
        if finished:
            break
        time.sleep(30)
    for handle in handles:
        handle.close()
    if status['phase']=='failed':
        raise SystemExit('A download shard failed; see data/logs/rxr_download_*.log and rerun.')
    feature_path = data/'datasets/RxR_VLNCE_v0/text_features'
    if feature_path.exists() or feature_path.is_symlink():
        if feature_path.resolve() != (data/'datasets/RxR/text_features').resolve():
            raise RuntimeError('Existing feature path conflicts with downloaded data')
    else:
        feature_path.symlink_to('../RxR/text_features', target_is_directory=True)
    with (logs/'dataset_completion_verification.log').open('w') as handle:
        check = subprocess.run([sys.executable, str(data/'tools/verify_completed_datasets.py')],
                               stdout=handle, stderr=subprocess.STDOUT)
    status['phase'] = 'complete' if check.returncode == 0 else 'verification_failed'
    status['verification_report'] = str(reports/'dataset_completion_report.json')
    save(status_path, status)
    if check.returncode:
        raise SystemExit('Download completed but dataset verification failed; inspect report.')
    print('COMPLETE: all downloads and dataset checks passed.', flush=True)


if __name__ == '__main__':
    main()
