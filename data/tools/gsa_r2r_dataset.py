#!/usr/bin/env python3
"""GSA-R2R release metadata, resumable ZIP-member transfer, and validation."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zlib


GSA_COMMIT = 'd7310c8e792504fc347cf0747797a18d9c91723c'
SCALE_REVISION = '8ef638db28d49184e23ad3edf8aa5281fd8db7e3'
ROOT = 'datasets/GSA-R2R'
FEATURE_NAME = 'clip_vit-b16_mp3d_hm3d_gibson.hdf5'
FEATURE_MEMBER = dict(
    path=ROOT + '/R2R/features/' + FEATURE_NAME, kind='deflate_member',
    url='https://huggingface.co/datasets/OpenGVLab/ScaleVLN/resolve/' + SCALE_REVISION + '/features.zip',
    expected_bytes=11586333694, compressed_bytes=10874839924,
    data_offset=1075017822, crc32='2654d836', archive_bytes=37479432888,
    archive_sha256='b2c00714a72e2a915fc3afb60a80e9320504a1912a07d60c3c72c18254dfa1a3',
    archive_member='features/' + FEATURE_NAME,
)


def make_job():
    upstream = 'https://raw.githubusercontent.com/honghd16/GSA-VLN/' + GSA_COMMIT + '/'
    return dict(name='GSA-R2R', source='https://github.com/honghd16/GSA-VLN',
                note='Official GSA-R2R instructions, ScaleVLN connectivity and required CLIP ViT-B/16 features. '
                     'Fetch only the required member of the larger feature archive. Model checkpoints and the '
                     'regenerable pano_inputs.h5 training cache are not dataset requirements.',
                assets=[
                    dict(path='datasets/.downloads/GSA-R2R.zip',
                         url='https://drive.usercontent.google.com/download?id=1uBR4F1FD3B6bTdXX17SOFYZRxvQhvYB9&export=download&confirm=t',
                         kind='zip', expected_bytes=116359054,
                         extract_to=ROOT + '/R2R/annotations/ESA_Dataset', strip_prefix='GSA-R2R Dataset/'),
                    dict(path='datasets/.downloads/ScaleVLN-r2r-preprocess.zip',
                         url='https://huggingface.co/datasets/OpenGVLab/ScaleVLN/resolve/' + SCALE_REVISION + '/r2r_preprocess_data.zip',
                         kind='zip', expected_bytes=546588395,
                         expected_sha256='d1820ae38a115e577df3aeb09360d6f2313b82b77a59476dff1386b04ab7345a',
                         extract_to=ROOT + '/R2R', strip_prefix='r2r_preprocess_data/'),
                    dict(path=ROOT + '/README.upstream.md', url=upstream + 'README.md', kind='text', expected_bytes=None),
                    dict(path=ROOT + '/LICENSE', url=upstream + 'LICENSE', kind='text', expected_bytes=None),
                    FEATURE_MEMBER.copy(),
                ])


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_member(data, item, status, chunk_bytes=32 * 1024**2, workers=8):
    """Save byte ranges separately; inflate only after every range is verified.

    Each request must return HTTP 206 and the exact Content-Range. ZIP CRC checks
    the complete inflated member. SHA-256 receipts make local restarts auditable.
    """
    target = data / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    chunks = data / 'datasets/.downloads/GSA-R2R-clip-b16.chunks'
    chunks.mkdir(parents=True, exist_ok=True)
    manifest_path = chunks / 'manifest.json'
    expected = dict(source=item, chunk_bytes=chunk_bytes)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else dict(**expected, chunks={})
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Saved range metadata differs from the pinned source: ' + str(manifest_path))
    atomic_json(manifest_path, manifest)
    count = (item['compressed_bytes'] + chunk_bytes - 1) // chunk_bytes
    completed = {}
    pending = []
    for index in range(count):
        size = min(chunk_bytes, item['compressed_bytes'] - index * chunk_bytes)
        path = chunks / f'{index:05d}.deflate'
        receipt = manifest['chunks'].get(str(index), {})
        if path.is_file() and path.stat().st_size == size and receipt.get('sha256') == file_sha(path):
            completed[index] = size
        else:
            pending.append((index, size, path))
    baseline = sum(completed.values())
    started = time.monotonic()

    def progress():
        done = sum(completed.values())
        speed = (done - baseline) / max(0.001, time.monotonic() - started)
        status(current_file=item['path'], transfer_bytes_done=done,
               transfer_total_bytes=item['compressed_bytes'], transfer_rate_bytes_per_second=round(speed),
               transfer_eta_seconds=round((item['compressed_bytes'] - done) / speed) if speed else None)

    def fetch(part):
        index, size, path = part
        begin = item['data_offset'] + index * chunk_bytes
        end = begin + size - 1
        partial = path.with_name(path.name + '.part')
        header = path.with_name(path.name + '.headers')
        subprocess.run([
            '/usr/bin/curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
            '--connect-timeout', '30', '--max-time', '240', '--speed-time', '30', '--speed-limit', '1024',
            '--retry', '4', '--retry-delay', '5', '--retry-all-errors',
            '--range', f'{begin}-{end}', '--dump-header', str(header), '--output', str(partial), item['url'],
        ], check=True)
        headers = header.read_text(errors='replace')
        statuses = re.findall(r'^HTTP/\S+\s+(\d+)', headers, flags=re.M)
        ranges = re.findall(r'^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$', headers, flags=re.M | re.I)
        if (not statuses or statuses[-1] != '206' or not ranges
                or tuple(map(int, ranges[-1])) != (begin, end, item['archive_bytes'])
                or partial.stat().st_size != size):
            raise ValueError('Incorrect HTTP range response for chunk ' + str(index))
        digest = file_sha(partial)
        partial.replace(path)
        return index, dict(bytes=size, sha256=digest)

    progress()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, part) for part in pending]
        errors = []
        for future in concurrent.futures.as_completed(futures):
            try:
                index, receipt = future.result()
                manifest['chunks'][str(index)] = receipt
                completed[index] = receipt['bytes']
                atomic_json(manifest_path, manifest)
                progress()
                print(f'GSA FEATURES {len(completed)}/{count} chunks', flush=True)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('Feature range download failures: ' + '; '.join(errors[:5]))

    status(current_file='inflating ' + item['path'])
    partial = target.with_name(target.name + '.extracting')
    inflater, crc, size, digest = zlib.decompressobj(-15), 0, 0, hashlib.sha256()
    with partial.open('wb') as output:
        for index in range(count):
            with (chunks / f'{index:05d}.deflate').open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    decoded = inflater.decompress(block)
                    output.write(decoded)
                    crc = zlib.crc32(decoded, crc)
                    size += len(decoded)
                    digest.update(decoded)
        decoded = inflater.flush()
        output.write(decoded)
        crc = zlib.crc32(decoded, crc)
        size += len(decoded)
        digest.update(decoded)
    if (not inflater.eof or inflater.unused_data or size != item['expected_bytes']
            or f'{crc & 0xffffffff:08x}' != item['crc32']):
        raise ValueError('Inflated feature member failed size/CRC/end-of-stream verification')
    with partial.open('rb') as stream:
        if stream.read(8) != b'\x89HDF\r\n\x1a\n':
            raise ValueError('Feature member is not an HDF5 file')
    if target.exists():
        target.rename(target.with_name(target.name + '.previous.' + str(time.time_ns())))
    partial.replace(target)
    return digest.hexdigest()


def annotation_names():
    names = ['Train/R2R_train_enc.json', 'Train/prevalent_aug_train_enc.json']
    for split, prefix in [('Validation', 'val'), ('Test', 'test')]:
        for region, slug in [('Residential', 'residential'), ('Non-residential', 'non_residential')]:
            base = split + '/' + region
            names.append(f'{base}/Basic/{prefix}_{slug}_basic_enc.json')
            if region == 'Non-residential':
                names.append(f'{base}/Scene/{prefix}_{slug}_scene_enc.json')
            names.extend(f'{base}/User/{user}.json' for user in ('child', 'keith', 'moira', 'rachel', 'sheldon'))
    return names


def verify(data):
    import h5py
    import numpy as np
    root = data / ROOT / 'R2R'
    graphs, required_nodes, errors, splits = {}, set(), [], {}
    def graph(scan):
        if scan not in graphs:
            rows = json.loads((root / 'connectivity' / (scan + '_connectivity.json')).read_text())
            included = {row['image_id'] for row in rows if row['included']}
            adjacency = {row['image_id']: {rows[i]['image_id'] for i, yes in enumerate(row['unobstructed'])
                                          if yes and rows[i]['included']}
                         for row in rows if row['included']}
            required_nodes.update(scan + '_' + node for node in included)
            graphs[scan] = adjacency
        return graphs[scan]
    for name in annotation_names():
        rows = json.loads((root / 'annotations/ESA_Dataset' / name).read_text())
        assert isinstance(rows, list) and rows, name
        scans = set()
        instructions = 0
        for row in rows:
            assert all(k in row for k in ('scan', 'path', 'path_id', 'instructions', 'instr_encodings')), name
            scan, path = row['scan'], row['path']
            scans.add(scan)
            adjacency = graph(scan)
            assert path and all(v in adjacency for v in path), (name, row['path_id'], 'missing viewpoint')
            assert all(a == b or b in adjacency[a] for a, b in zip(path, path[1:])), (name, row['path_id'], 'missing edge')
            assert len(row['instructions']) == len(row['instr_encodings']) > 0, (name, row['path_id'])
            instructions += len(row['instructions'])
        splits[name] = dict(paths=len(rows), instructions=instructions, scans=len(scans))
    with h5py.File(root / 'features' / FEATURE_NAME, 'r') as features:
        keys = set(features.keys())
        missing = sorted(required_nodes - keys)
        if missing:
            errors.append(dict(missing_features=missing[:20], total=len(missing)))
        bad_shapes = []
        for key in sorted(required_nodes & keys):
            shape = features[key].shape
            if len(shape) != 2 or shape[0] != 36 or shape[1] < 512:
                bad_shapes.append([key, list(shape)])
        if bad_shapes:
            errors.append(dict(bad_feature_shapes=bad_shapes[:20], total=len(bad_shapes)))
        samples = sorted(required_nodes & keys)
        samples = samples[::max(1, len(samples) // 64)]
        for key in samples:
            if not np.isfinite(features[key][:]).all():
                errors.append(dict(nonfinite_feature=key))
    report = dict(passed=not errors, annotation_files=len(splits), splits=splits, scenes=len(graphs),
                  required_feature_viewpoints=len(required_nodes), available_feature_viewpoints=len(keys),
                  feature_samples_read=len(samples), errors=errors,
                  scope='Official annotations, navigation graphs and precomputed CLIP features; rendering disabled in the upstream baseline.')
    atomic_json(data / 'splits/gsa_r2r_completion_report.json', report)
    return report


def write_readme(data):
    (data / ROOT / 'README.local.md').write_text('''# GSA-R2R local dataset

Official source: https://github.com/honghd16/GSA-VLN

`R2R/annotations/ESA_Dataset` contains all 28 official JSON files, preserving
the upstream Train, Validation and Test directory structure. `R2R/connectivity`
and `R2R/connectivity_mp3d` come from the official ScaleVLN preprocessing archive.
`R2R/features/clip_vit-b16_mp3d_hm3d_gibson.hdf5` is the exact CLIP ViT-B/16
member required by the GSA-R2R baseline. Other ScaleVLN preprocessing files
are retained alongside these annotations.

Set the upstream `--root_dir` to this GSA-R2R directory. The upstream parser
also hardcodes relative feature/connectivity paths: point `img_ft_file` and
`connectivity_dir` to the files above when running from another code checkout.
The baseline uses MatterSim with rendering disabled and these feature/graph
assets. Raw HM3D meshes are not included in this release mirror.

The feature member is downloaded using verified HTTP byte ranges, decompressed
and checked against its upstream ZIP CRC. Its local SHA-256 and range receipts
are recorded under `data/splits` and `data/datasets/.downloads`.
The full 37.5 GB multi-model feature archive is not needed for this baseline.
Model checkpoints are optional; `pano_inputs.h5` is a regenerable pretraining
cache (see the upstream README), rather than an additional annotation release.

Validation results: `data/splits/gsa_r2r_completion_report.json`.
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--verify', action='store_true', required=True)
    args = parser.parse_args()
    try:
        result = verify(args.data_root.resolve())
    except Exception as exc:
        result = dict(passed=False, errors=[str(exc)])
        atomic_json(args.data_root / 'splits/gsa_r2r_completion_report.json', result)
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)
