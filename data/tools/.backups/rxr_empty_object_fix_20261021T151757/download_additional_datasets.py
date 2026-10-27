#!/usr/bin/env python3
"""Download the additional VLN datasets after the existing RxR job verifies successfully.

Only Python's standard library and curl are required. Run --preflight to check the
queue without starting downloads. Status, receipts, and reports live in data/splits.
"""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time
import traceback
import zipfile
import zlib


DATA = Path(__file__).resolve().parents[1]
SERVICE = 'rolevln-additional-datasets.service'
SPLITS = ('train', 'val_seen', 'val_unseen')
R4R_COMMIT = '5b09c22d73a9d35eb6c5d2a99b95677a45053466'
R4R_PYTHON = Path('/home/bufsgpu/yes/envs/sw/bin/python3')
R4R_COUNTS = {'train': (25930, 233613), 'val_seen': (115, 1035), 'val_unseen': (5018, 45162)}
SOON_FILES = (
    ('train', '1tuS3PFHOECwA5U-ofFyv6WZnTXoZjaBr', 18626722),
    ('val_unseen_instrs', '1r6-rkaj02fdVPmFdpTlVrUX9k6WECRNb', 733047),
    ('val_unseen_house', '15Xd9DlQjWVMY-BY-jFInKF2h6syxq79e', 2354256),
    ('test_release', '1zjpcTsNhsEB1frrLvncQxUusfspZqrFA', 1206700),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def crc32(path):
    result = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result = zlib.crc32(block, result)
    return result & 0xffffffff


def drive_url(file_id):
    return 'https://drive.usercontent.google.com/download?id=' + file_id + '&export=download&confirm=t'


def plan():
    jobs = []

    def asset(path, url, kind='json', size=None, **kwargs):
        return dict(path=path, url=url, kind=kind, expected_bytes=size, **kwargs)

    def doc(repo, source, dest):
        return asset(dest, 'https://raw.githubusercontent.com/' + repo + '/master/' + source, 'text')

    soon = [asset('datasets/SOON/' + name + '.json', drive_url(file_id), size=size)
            for name, file_id, size in SOON_FILES]
    soon.append(doc('ZhuFengdaaa/SOON', 'README.md', 'datasets/SOON/README.upstream.md'))
    soon.append(doc('ZhuFengdaaa/SOON', 'LICENSE', 'datasets/SOON/LICENSE'))
    jobs.append(dict(name='SOON', source='https://scenario-oriented-object-navigation.github.io/', assets=soon,
                     note='All four publicly linked original FAO/SOON splits. Bounding boxes are embedded in the JSON. '
                          'The separate val_seen_instrs file referenced by old baseline code is not linked in the public release.'))
    for dataset in ('CVDN', 'NDH'):
        files = [asset(f'datasets/{dataset}/{split}.json', f'https://cvdn.dev/dataset/{dataset}/train_val/{split}.json')
                 for split in SPLITS]
        if dataset == 'NDH':
            files.append(asset('datasets/NDH/test.json', 'https://cvdn.dev/dataset/NDH/test_cleaned/test_cleaned.json', size=2645768))
        files.extend([doc('mmurray/cvdn', f'tasks/{dataset}/data/README.md', f'datasets/{dataset}/README.upstream.md'),
                      doc('mmurray/cvdn', f'tasks/{dataset}/data/download.sh', f'datasets/{dataset}/download.upstream.sh'),
                      doc('mmurray/cvdn', 'LICENSE', f'datasets/{dataset}/LICENSE')])
        jobs.append(dict(name=dataset, source='https://github.com/mmurray/cvdn', assets=files,
                         note='CVDN publishes train/val_seen/val_unseen dialogs. NDH also publishes test_cleaned, saved as test.json per its official script.'))
    hanna = [asset('datasets/.downloads/HANNA_data.zip', 'https://dl.dropboxusercontent.com/s/qfahoe22vh1cdhn/hanna_data.zip',
                   'zip', 51686810, extract_to='datasets/HANNA')]
    for source, dest in [('README.md', 'README.upstream.md'), ('LICENSE', 'LICENSE'), ('data/README.md', 'DATA.upstream.md'),
                         ('data/download.sh', 'download.upstream.sh'), ('data/accepted_objects.txt', 'accepted_objects.txt'),
                         ('data/region_label.txt', 'region_label.txt')]:
        hanna.append(doc('khanhptnk/hanna', source, 'datasets/HANNA/' + dest))
    jobs.append(dict(name='HANNA', source='https://github.com/khanhptnk/hanna/tree/master/data', assets=hanna,
                     note='Five task splits, assistance routes, panoramic action space, vocabulary and scan split.'))
    jobs.append(dict(name='VLN-CE', source='https://github.com/jacobkrantz/VLN-CE', assets=[
        asset('datasets/.downloads/R2R_VLNCE_v1-3.zip', drive_url('1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm'),
              'zip', 2608469, extract_to='datasets/R2R_VLNCE_v1-3', strip_prefix='R2R_VLNCE_v1-3/'),
        doc('jacobkrantz/VLN-CE', 'README.md', 'datasets/VLN-CE/README.upstream.md'),
        doc('jacobkrantz/VLN-CE', 'LICENSE', 'datasets/VLN-CE/LICENSE'),
    ], note='Add the minimal R2R release; reuse existing preprocessed R2R-CE, RxR-CE, text features and all 90 Habitat scenes.'))
    r4r_base = 'https://raw.githubusercontent.com/google-research/google-research/' + R4R_COMMIT + '/'
    jobs.append(dict(name='R4R', source='https://github.com/google-research/google-research/tree/master/r4r', assets=[
        asset('datasets/R4R/upstream/r4r_generate_data.py', r4r_base + 'r4r/r4r_generate_data.py', 'text', 6061,
              expected_sha256='0ee50ec5d91fabb3b5e9775d6893696af8fc84531f050129d43cfa8de69b7c13'),
        asset('datasets/R4R/upstream/graph_utils.py', r4r_base + 'r4r/graph_utils.py', 'text', 5332,
              expected_sha256='1599f7ecca1e4349e7fe68dabeaad58d89737932646454664d42e35a291b806b'),
        asset('datasets/R4R/README.upstream.md', r4r_base + 'r4r/README.md', 'text'),
        asset('datasets/R4R/LICENSE', r4r_base + 'LICENSE', 'text'),
    ], note='Download the pinned official generator and construct train, val_seen and val_unseen from existing R2R '
            'using a 3.0 m distance threshold and no heading threshold. Share existing scenes and image features.',
                     generation=dict(commit=R4R_COMMIT, python=str(R4R_PYTHON), distance_threshold=3.0,
                                     heading_threshold=None, splits=list(SPLITS))))
    jobs.append(dict(name='shared_image_features', source='https://github.com/peteanderson80/Matterport3DSimulator#precomputing-resnet-image-features', assets=[
        asset('datasets/.downloads/ResNet-152-imagenet.zip', 'https://dl.dropboxusercontent.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip',
              'zip', 3064114894, extract_to='img_features')
    ], note='Official current feature link also used by HANNA. The old SOON feature link returns HTTP 403.'))
    jobs.append(dict(name='house_segmentations', source='existing data/.downloads/v1/scans', assets=[],
                     note='Extract all 90 existing house_segmentations archives into the shared MatterSim scene directories.'))
    return dict(created_at=now(), prerequisite='rxr_original_download_status.json: complete AND dataset_completion_report.json: passed',
                service_unit=SERVICE, jobs=jobs, dataset_aliases={'R2R-CE': 'VLN-CE', 'RxR-CE': 'VLN-CE'},
                completion_inventory={'json': 'splits/dataset_inventory.json', 'markdown': 'DATASET_INVENTORY.md'},
                public_release_limits=['SOON val_seen_instrs is not published on the official download page.',
                                       'The official CVDN dialog downloader publishes no test dialog file.',
                                       'The official R4R recipe covers train, val_seen and val_unseen; no separate test release is published there.',
                                       'Private evaluation ground truth and trained model weights are outside this dataset mirror.'])


def rxr_ready(data):
    status = read_json(data / 'splits/rxr_original_download_status.json', {})
    report = read_json(data / 'splits/dataset_completion_report.json', {})
    return (status.get('phase') == 'complete' and report.get('passed') is True
            and 'rxr' in report.get('scope', []) and status.get('verified_files', 0) == status.get('expected_files', -1))


def inspect_file(path, kind):
    if not path.stat().st_size:
        raise ValueError(f'Empty download: {path}')
    if kind == 'json':
        obj = read_json(path)
        if not isinstance(obj, (list, dict)) or not obj:
            raise ValueError(f'Invalid or empty JSON: {path}')
    elif kind == 'zip':
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist() or archive.testzip() is not None:
                raise ValueError(f'Invalid ZIP CRC: {path}')
    elif path.read_bytes()[:256].lstrip().lower().startswith((b'<!doctype html', b'<html')):
        raise ValueError(f'Unexpected HTML: {path}')


class Worker:
    def __init__(self, data, queue):
        self.data, self.queue = data, queue
        self.state_path = data / 'splits/additional_datasets_status.json'
        self.receipts_path = data / 'splits/additional_datasets_download_receipts.json'
        self.receipts = read_json(self.receipts_path, {})
        self.state = dict(phase='waiting_for_rxr', updated_at=now(), service_unit=SERVICE,
                          pid=os.getpid(), jobs={j['name']: {'phase': 'queued'} for j in queue['jobs']})

    def status(self, **kwargs):
        self.state.update(kwargs)
        self.state['updated_at'] = now()
        write_json(self.state_path, self.state)

    def download(self, item):
        path = self.data / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        old = self.receipts.get(item['path'], {})
        if (path.is_file() and old.get('sha256') and path.stat().st_size == old.get('bytes')
                and sha256(path) == old['sha256']
                and (not item.get('expected_sha256') or old['sha256'] == item['expected_sha256'])):
            print('REUSE', item['path'], flush=True)
            return path
        part = path.with_name(path.name + '.part')
        self.status(current_file=item['path'])
        print('DOWNLOAD', item['path'], item['url'], flush=True)
        cmd = ['/usr/bin/curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
               '--connect-timeout', '30', '--speed-time', '180', '--speed-limit', '1024',
               '--retry', '5', '--retry-delay', '10', '--retry-all-errors', '--continue-at', '-',
               '--output', str(part), item['url']]
        result = subprocess.run(cmd)
        if result.returncode:
            # curl 33 means this endpoint cannot resume. Keep the partial copy
            # separately and retry from byte zero, without touching a final file.
            if result.returncode == 33 and part.exists():
                part.rename(part.with_name(part.name + '.unresumable.' + str(time.time_ns())))
                subprocess.run(cmd, check=True)
            else:
                raise RuntimeError(f'curl exited {result.returncode}: {item["path"]}')
        try:
            if item.get('expected_bytes') is not None and part.stat().st_size != item['expected_bytes']:
                raise ValueError(f'Size mismatch: {part.stat().st_size} != {item["expected_bytes"]}')
            inspect_file(part, item['kind'])
            if item.get('expected_sha256') and sha256(part) != item['expected_sha256']:
                raise ValueError('Published source does not match the reviewed checksum: ' + item['path'])
        except Exception:
            if part.exists():
                part.rename(part.with_name(part.name + '.invalid.' + str(time.time_ns())))
            raise
        digest = sha256(part)
        if path.exists():
            # Preserve any unexpected existing local version before replacing it.
            path.rename(path.with_name(path.name + '.previous.' + str(time.time_ns())))
        part.replace(path)
        self.receipts[item['path']] = dict(url=item['url'], bytes=path.stat().st_size,
                                            sha256=digest, checked_at=now(), validation=item['kind'])
        write_json(self.receipts_path, self.receipts)
        return path

    def extract(self, archive_path, destination, prefix=''):
        files = []
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                if not entry.filename.startswith(prefix):
                    raise ValueError(f'Unexpected ZIP layout: {entry.filename}, expected {prefix}')
                name = PurePosixPath(entry.filename[len(prefix):])
                if name.is_absolute() or '..' in name.parts or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f'Unsafe ZIP entry: {entry.filename}')
                if name.parts[0] == '__MACOSX' or name.name == '.DS_Store':
                    continue
                path = destination / name
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f'ZIP destination escapes extraction directory: {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                if not (path.is_file() and path.stat().st_size == entry.file_size and crc32(path) == entry.CRC):
                    temp = path.with_name(path.name + '.extracting')
                    with archive.open(entry) as source, temp.open('wb') as target:
                        shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                    if temp.stat().st_size != entry.file_size or crc32(temp) != entry.CRC:
                        raise ValueError(f'Extracted file failed CRC: {path}')
                    if path.exists():
                        path.rename(path.with_name(path.name + '.previous.' + str(time.time_ns())))
                    temp.replace(path)
                files.append(dict(path=str(path.relative_to(self.data)), bytes=entry.file_size, crc32=f'{entry.CRC:08x}'))
        return files

    def link(self, path, target):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        if path.exists() or path.is_symlink():
            raise ValueError(f'Refusing to replace existing path with a link: {path}')
        path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())

    def generate_r4r(self):
        """Use the reviewed upstream implementation without changing any algorithm."""
        root = self.data / 'datasets/R4R'
        script = root / 'upstream/r4r_generate_data.py'
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.update(PYTHONNOUSERSITE='1', MPLBACKEND='Agg')
        versions = subprocess.run([
            str(R4R_PYTHON), '-c',
            'import json,numpy,networkx,matplotlib; print(json.dumps({x.__name__:x.__version__ for x in (numpy,networkx,matplotlib)}))'
        ], check=True, capture_output=True, text=True, env=env)
        packages = json.loads(versions.stdout)
        manifest_path = root / 'generation_manifest.json'
        previous = read_json(manifest_path, {})
        generated = {}
        code_hashes = {p.name: sha256(p) for p in (script, root / 'upstream/graph_utils.py')}
        for split in SPLITS:
            source = self.data / 'datasets/R2R' / ('R2R_' + split + '.json')
            inputs = json.loads(source.read_text())
            graphs = {scan: sha256(self.data / 'connectivity' / (scan + '_connectivity.json'))
                      for scan in sorted({row['scan'] for row in inputs})}
            signature = dict(input_sha256=sha256(source), connectivity_sha256=graphs,
                             code_sha256=code_hashes, packages=packages, distance_threshold=3.0, heading_threshold=None)
            destination = root / ('R4R_' + split + '.json')
            old = previous.get('splits', {}).get(split, {})
            if (old.get('inputs') == signature and destination.is_file()
                    and old.get('sha256') == sha256(destination)):
                generated[split] = old
                print('REUSE', destination, flush=True)
                continue
            self.status(current_file='generating ' + str(destination.relative_to(self.data)))
            partial = destination.with_name(destination.name + '.part')
            subprocess.run([
                str(R4R_PYTHON), str(script), '--connections_dir', str(self.data / 'connectivity'),
                '--input_file_path', str(source), '--output_file_path', str(partial), '--distance_threshold', '3.0',
            ], check=True, env=env)
            rows = json.loads(partial.read_text())
            counts = (len(rows), sum(len(r['instructions']) for r in rows))
            if counts != R4R_COUNTS[split]:
                raise ValueError(f'R4R {split}: generated {counts}, expected published counts {R4R_COUNTS[split]}')
            if destination.exists():
                destination.rename(destination.with_name(destination.name + '.previous.' + str(time.time_ns())))
            partial.replace(destination)
            generated[split] = dict(inputs=signature, path=str(destination.relative_to(self.data)),
                                    paths=counts[0], instructions=counts[1], bytes=destination.stat().st_size,
                                    sha256=sha256(destination), generated_at=now())
            write_json(manifest_path, dict(upstream_commit=R4R_COMMIT, python=str(R4R_PYTHON), splits=generated))
        write_json(manifest_path, dict(upstream_commit=R4R_COMMIT, python=str(R4R_PYTHON), splits=generated))
        return generated

    def run_job(self, job):
        extracted = []
        for item in job['assets']:
            path = self.download(item)
            if 'extract_to' in item:
                self.status(current_file='extracting ' + item['path'])
                extracted.extend(self.extract(path, self.data / item['extract_to'], item.get('strip_prefix', '')))
        name = job['name']
        generated = self.generate_r4r() if name == 'R4R' else None
        if name == 'house_segmentations':
            for scan in (self.data / 'scans.txt').read_text().split():
                source = self.data / '.downloads/v1/scans' / scan / 'house_segmentations.zip'
                self.status(current_file='extracting ' + str(source.relative_to(self.data)))
                extracted.extend(self.extract(source, self.data / 'mattersim/v1/scans' / scan, scan + '/'))
        if name in ('SOON', 'CVDN', 'NDH', 'HANNA', 'R4R'):
            root = self.data / 'datasets' / name
            self.link(root / 'connectivity', self.data / 'connectivity')
            self.link(root / 'scans', self.data / 'mattersim/v1/scans')
            self.link(root / 'img_features', self.data / 'img_features')
            if name == 'SOON':
                self.link(root / 'test.json', root / 'test_release.json')
            if name == 'NDH':
                self.link(root / 'test_cleaned.json', root / 'test.json')
        if name == 'VLN-CE':
            root = self.data / 'datasets/VLN-CE'
            for folder in ('R2R_VLNCE_v1-3', 'R2R_VLNCE_v1-3_preprocessed', 'RxR_VLNCE_v0'):
                self.link(root / folder, self.data / 'datasets' / folder)
            self.link(root / 'scene_datasets', self.data / 'scene_datasets')
        provenance = dict(source=job['source'], note=job['note'], completed_at=now(),
                          downloads={item['path']: self.receipts[item['path']] for item in job['assets']}, extracted_files=extracted)
        if generated is not None:
            provenance['generated_splits'] = generated
        write_json(self.data / 'splits' / ('additional_' + name.lower().replace('-', '_') + '_manifest.json'), provenance)

    def verify(self):
        report = dict(checked_at=now(), checks=[], datasets={}, public_release_limits=self.queue['public_release_limits'])

        def check(name, passed, detail):
            report['checks'].append(dict(name=name, passed=bool(passed), detail=detail))
            print(('PASS ' if passed else 'FAIL ') + name + ': ' + str(detail), flush=True)

        graphs = {p.name.split('_')[0]: {r['image_id'] for r in json.loads(p.read_text()) if r['included']}
                  for p in (self.data / 'connectivity').glob('*_connectivity.json')}
        required_features = set()

        def scan_id(row):
            boxes = row.get('bboxes') or []
            return row.get('scan') or (boxes[0].get('scan') if boxes else None)

        def annotation(path, dataset):
            rows = json.loads(path.read_text())
            assert isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows), str(path)
            missing = set()
            required = {'SOON': ('instructions',), 'CVDN': ('idx', 'target', 'dialog_history'),
                        'NDH': ('inst_idx', 'target', 'dialog_history'), 'HANNA': ('paths', 'instructions'),
                        'R4R': ('path_id', 'path', 'instructions', 'first_path_id', 'second_path_id')}[dataset]
            for row in rows:
                scan = scan_id(row)
                if scan not in graphs:
                    missing.add('scan:' + str(scan))
                for field in required:
                    if field not in row:
                        missing.add('field:' + field)
                # Check all explicit navigation-node lists, including nested SOON
                # and HANNA paths. Coordinate tuples and headings are not node IDs.
                def nodes(value):
                    if isinstance(value, str) and re.fullmatch('[0-9a-f]{32}', value):
                        yield value
                    elif isinstance(value, (list, tuple)):
                        for sub in value:
                            yield from nodes(sub)
                    elif isinstance(value, dict) and 'pano' in value:
                        yield from nodes(value['pano'])
                for key in ('path', 'paths', 'nav_steps', 'planner_nav_steps', 'nav_history', 'planner_path', 'player_path', 'start_pano', 'end_panos'):
                    for vp in nodes(row.get(key)):
                        required_features.add((scan, vp))
                        if vp not in graphs.get(scan, set()):
                            missing.add(str(scan) + ':' + vp)
                for box in row.get('bboxes', []):
                    vp = box.get('image_id')
                    if vp:
                        required_features.add((box.get('scan', scan), vp))
                        if vp not in graphs.get(box.get('scan', scan), set()):
                            missing.add(str(scan) + ':' + vp)
            detail = dict(records=len(rows), scans=len({scan_id(r) for r in rows}), missing=sorted(missing)[:20])
            if dataset == 'R4R':
                split = path.stem.removeprefix('R4R_')
                detail['instructions'] = sum(len(row['instructions']) for row in rows)
                check('R4R/' + split + ' published counts',
                      (len(rows), detail['instructions']) == R4R_COUNTS[split], detail)
            check(dataset + '/' + path.name, not missing, detail)
            return detail

        for dataset, files in [('SOON', [n + '.json' for n, _, _ in SOON_FILES]),
                               ('CVDN', [s + '.json' for s in SPLITS]),
                               ('NDH', [s + '.json' for s in (*SPLITS, 'test')]),
                               ('R4R', ['R4R_' + s + '.json' for s in SPLITS]),
                               ('HANNA', ['hanna_' + s + '.json' for s in ('train', 'val_seen', 'val_unseen', 'test_seen', 'test_unseen')])]:
            result = {}
            for file in files:
                try:
                    root = self.data / 'datasets' / dataset
                    if dataset == 'HANNA':
                        root = root / 'hanna'
                    result[file] = annotation(root / file, dataset)
                except Exception as exc:
                    check(dataset + '/' + file, False, str(exc))
            report['datasets'][dataset] = result
            for link in ('connectivity', 'scans', 'img_features'):
                check(dataset + ' shared ' + link, (self.data / 'datasets' / dataset / link).is_dir(), link)

        try:
            root = self.data / 'datasets/HANNA/hanna'
            for file in ('routes.json', 'panoramic_action_space.json', 'scan_split.json'):
                obj = json.loads((root / file).read_text())
                check('HANNA/' + file, bool(obj), dict(entries=len(obj)))
            check('HANNA vocabulary', (root / 'vocab.txt').stat().st_size > 0, 'vocab.txt')
            scans = (self.data / 'scans.txt').read_text().split()
            absent = [scan for scan in scans if not (self.data / 'mattersim/v1/scans' / scan / 'house_segmentations' / (scan + '.house')).is_file()]
            check('HANNA house segmentations', len(scans) == 90 and not absent, dict(scans=len(scans), missing=absent))
        except Exception as exc:
            check('HANNA assistance assets', False, str(exc))

        # Validate all feature rows without loading a large TSV into memory.
        try:
            feature_files = sorted((self.data / 'img_features').rglob('*.tsv'))
            count, keys, bad, widths = 0, set(), [], set()
            for path in feature_files:
                with path.open() as stream:
                    for line in stream:
                        row = line.rstrip('\r\n').split('\t')
                        if row[0] == 'scanId':
                            continue
                        if len(row) != 6:
                            bad.append('columns'); continue
                        payload = base64.b64decode(row[-1], validate=True)
                        widths.add(len(payload))
                        key = (row[0], row[1]); count += 1
                        if key in keys or row[1] not in graphs.get(row[0], set()):
                            bad.append(':'.join(key))
                        keys.add(key)
            check('Shared ResNet-152 feature TSV', bool(count) and not bad and widths == {36 * 2048 * 4},
                  dict(files=[str(p.relative_to(self.data)) for p in feature_files], viewpoints=count, payload_sizes=sorted(widths), errors=bad[:20]))
            missing_features = required_features - keys
            check('Image features cover task viewpoints', not missing_features,
                  dict(required=len(required_features), missing=[str(x) for x in list(missing_features)[:20]]))
        except Exception as exc:
            check('Shared ResNet-152 feature TSV', False, str(exc))

        for folder in ('R2R_VLNCE_v1-3', 'R2R_VLNCE_v1-3_preprocessed', 'RxR_VLNCE_v0'):
            try:
                archive_path = self.data / 'datasets/.downloads' / (folder + '.zip')
                checked, bad = 0, []
                with zipfile.ZipFile(archive_path) as archive:
                    for item in archive.infolist():
                        if item.is_dir():
                            continue
                        path = self.data / 'datasets' / item.filename
                        checked += 1
                        if not path.is_file() or path.stat().st_size != item.file_size or crc32(path) != item.CRC:
                            bad.append(item.filename); continue
                        if path.name.endswith('.gz'):
                            with gzip.open(path, 'rt') as stream:
                                obj = json.load(stream)
                            if isinstance(obj, dict) and 'episodes' in obj:
                                for episode in obj['episodes']:
                                    scene = str(episode['scene_id']).replace('data/scene_datasets/', '')
                                    if not (self.data / 'scene_datasets' / scene).is_file():
                                        bad.append(scene)
                check('VLN-CE/' + folder, bool(checked) and not bad, dict(files=checked, errors=bad[:20]))
            except Exception as exc:
                check('VLN-CE/' + folder, False, str(exc))
        feature_link = self.data / 'datasets/RxR_VLNCE_v0/text_features'
        check('VLN-CE RxR text features', feature_link.is_dir() and rxr_ready(self.data), str(feature_link))
        report['passed'] = all(c['passed'] for c in report['checks'])
        write_json(self.data / 'splits/additional_datasets_completion_report.json', report)
        return report['passed']

    def write_inventory(self):
        """Publish the requested dataset list only after both sets of checks pass."""
        reports = ['splits/dataset_completion_report.json', 'splits/additional_datasets_completion_report.json']
        if not all(read_json(self.data / path, {}).get('passed') is True for path in reports):
            raise ValueError('The final inventory requires successful verification of both dataset groups.')
        entries = [
            ('R2R', ['datasets/R2R'], 'Original discrete navigation release'),
            ('RxR', ['datasets/RxR'], 'Original multilingual release, text features and pose traces'),
            ('REVERIE', ['datasets/REVERIE'], 'v1 and v2 annotations with matching bounding boxes'),
            ('SOON', ['datasets/SOON'], 'All four publicly linked original FAO/SOON splits'),
            ('CVDN', ['datasets/CVDN'], 'Public train, val_seen and val_unseen dialog splits'),
            ('NDH', ['datasets/NDH'], 'Navigation instances from dialog history, including public test'),
            ('HANNA', ['datasets/HANNA/hanna'], 'Five task splits and assistance assets'),
            ('R2R-CE', ['datasets/R2R_VLNCE_v1-3', 'datasets/R2R_VLNCE_v1-3_preprocessed'],
             'VLN-CE: minimal and preprocessed releases, including EnvDrop augmentation'),
            ('RxR-CE', ['datasets/RxR_VLNCE_v0'], 'VLN-CE: continuous episodes with shared RxR text features'),
            ('R4R', ['datasets/R4R'], 'Train, val_seen and val_unseen constructed with the official recipe'),
        ]
        for _, paths, _ in entries:
            for path in paths:
                if not (self.data / path).is_dir():
                    raise ValueError('Verified dataset directory is missing: ' + path)
        shared = ['scene_datasets/mp3d', 'mattersim/v1/scans', 'connectivity', 'img_features']
        known = {PurePosixPath(path).parts[1] for _, paths, _ in entries for path in paths} | {'VLN-CE'}
        others = [dict(name=p.name, path=str(p.relative_to(self.data)), status='Not audited by the VLN checks')
                  for p in sorted((self.data / 'datasets').iterdir())
                  if p.is_dir() and not p.name.startswith('.') and p.name not in known]
        inventory = dict(created_at=now(), status='verified', scope='Requested VLN datasets and public releases',
                         datasets=[dict(name=name, paths=paths, notes=notes) for name, paths, notes in entries],
                         shared_assets=shared, verification_reports=reports,
                         release_limits=self.queue['public_release_limits'], other_existing_directories=others)
        write_json(self.data / 'splits/dataset_inventory.json', inventory)
        lines = ['# Completed VLN dataset inventory', '', 'Verified at: ' + inventory['created_at'], '',
                 'All paths below are relative to this data directory. VLN-CE covers R2R-CE and RxR-CE.', '',
                 '| Dataset | Location | Coverage |', '|---|---|---|']
        for name, paths, notes in entries:
            lines.append('| ' + name + ' | ' + ', '.join('`' + p + '`' for p in paths) + ' | ' + notes + ' |')
        lines.extend(['', 'Shared assets: ' + ', '.join('`' + p + '`' for p in shared) + '.', '',
                      'Verification reports (including split counts): ' + ', '.join('`' + p + '`' for p in reports) + '.', '',
                      'Public-release limitations:', ''])
        lines.extend('- ' + note for note in self.queue['public_release_limits'])
        if others:
            lines.extend(['', 'Other existing directories, not audited by the VLN checks:', ''])
            lines.extend('- `' + row['path'] + '`' for row in others)
        temporary = self.data / 'DATASET_INVENTORY.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(self.data / 'DATASET_INVENTORY.md')
        print('Final dataset inventory:', self.data / 'DATASET_INVENTORY.md', flush=True)

    def run(self):
        self.status()
        print('Waiting for the full RxR transfer AND its final verification to pass.', flush=True)
        while not rxr_ready(self.data):
            self.status(prerequisite_phase=read_json(self.data / 'splits/rxr_original_download_status.json', {}).get('phase', 'unknown'))
            time.sleep(30)
        print('RxR verified. Starting the additional dataset queue.', flush=True)
        self.status(phase='running', started_at=now())
        failed = []
        for job in self.queue['jobs']:
            self.state['jobs'][job['name']] = dict(phase='running', started_at=now())
            self.status(current_job=job['name'])
            try:
                self.run_job(job)
                self.state['jobs'][job['name']] = dict(phase='downloaded', finished_at=now())
            except Exception as exc:
                traceback.print_exc()
                failed.append(job['name'])
                self.state['jobs'][job['name']] = dict(phase='failed', error=str(exc), finished_at=now())
            self.status()
        self.status(phase='verifying', current_job=None, current_file=None)
        passed = self.verify()
        if passed and not failed:
            self.write_inventory()
        self.status(phase='complete' if passed and not failed else 'retry_pending',
                    failed_jobs=failed, verification_passed=passed, finished_at=now())
        return 0 if passed and not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--preflight', action='store_true', help='Validate and record the queue without downloading data.')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    os.umask(0o007)
    data = args.data_root.resolve()
    queue = plan()
    for directory in ('splits', 'logs'):
        (data / directory).mkdir(parents=True, exist_ok=True)
    if args.preflight:
        assert shutil.which('curl'), 'curl is required'
        assert shutil.disk_usage(data).free > 45 * 1024**3, 'At least 45 GiB free space is required'
        scans = (data / 'scans.txt').read_text().split()
        assert len(scans) == 90
        assert R4R_PYTHON.is_file(), 'R4R generation interpreter is missing'
        for split in SPLITS:
            assert (data / 'datasets/R2R' / ('R2R_' + split + '.json')).is_file(), 'R4R source split missing: ' + split
        for scan in scans:
            assert (data / '.downloads/v1/scans' / scan / 'house_segmentations.zip').is_file(), scan
            assert os.access(data / 'mattersim/v1/scans' / scan, os.W_OK), 'Cannot extract segmentation files for ' + scan
        write_json(data / 'splits/additional_datasets_plan.json', queue)
        print(json.dumps(dict(jobs=[j['name'] for j in queue['jobs']], rxr_ready=rxr_ready(data),
                              known_download_bytes=sum(a.get('expected_bytes') or 0 for j in queue['jobs'] for a in j['assets']),
                              free_bytes=shutil.disk_usage(data).free), indent=2))
        return 0
    with (data / 'splits/additional_datasets.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = Worker(data, queue)
        if args.verify_only:
            return 0 if worker.verify() else 1
        write_json(data / 'splits/additional_datasets_plan.json', queue)
        return worker.run()


if __name__ == '__main__':
    raise SystemExit(main())
