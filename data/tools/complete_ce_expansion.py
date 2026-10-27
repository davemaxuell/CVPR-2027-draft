#!/usr/bin/env python3
"""Organize and verify the three CE releases listed in ce_expansion_plan.json.

Run after downloading the plan's immutable assets. No network requests, simulator
execution, credentials, or changes to original annotation bytes are involved.
Exit 0 means verification completed; read scene_dependencies_ready for readiness.
"""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import zipfile
import zlib

from download_additional_datasets import DATA, Worker, now, plan, write_json


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, size):
    return (isinstance(value, list) and len(value) == size
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value))


def stream_sha256(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024**2), b''):
        result.update(block)
    return result.digest()


def link(path, target):
    require(target.exists(), 'Missing link target: ' + str(target))
    if path.is_symlink() or path.exists():
        require(path.resolve() == target.resolve(), 'Conflicting existing path: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())


def compressed_copy(source, destination):
    """Preserve the exact published JSON bytes inside a deterministic gzip."""
    if destination.exists():
        with gzip.open(destination, 'rb') as stream, source.open('rb') as original:
            require(stream_sha256(stream) == stream_sha256(original),
                    'Existing compressed copy differs: ' + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with source.open('rb') as inp, temporary.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as out:
            shutil.copyfileobj(inp, out)
    temporary.replace(destination)
    compressed_copy(source, destination)


class Audit:
    def __init__(self, data):
        self.data = data
        self.plan = read(data / 'splits/ce_expansion_plan.json')
        self.receipts = {}
        self.scenes = {}
        self.report = dict(checked_at=now(), requested_datasets=self.plan['requested_datasets'],
                           datasets={}, checks=[], pending_dependencies=[], release_limits=[])

    def check(self, name, **details):
        self.report['checks'].append(dict(name=name, passed=True, **details))
        print('PASS', name, flush=True)

    def hashes(self):
        for name, release in self.plan['datasets'].items():
            for asset in release['assets']:
                path = self.data / asset['path']
                require(path.is_file(), 'Missing release file: ' + str(path))
                size = path.stat().st_size
                require(size == asset['expected_bytes'], 'Size mismatch: ' + str(path))
                sha256 = hashlib.sha256()
                blob = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(4 * 1024**2), b''):
                        sha256.update(block)
                        blob.update(block)
                if asset.get('expected_sha256'):
                    require(sha256.hexdigest() == asset['expected_sha256'], 'LFS hash mismatch: ' + str(path))
                if asset.get('expected_git_blob_sha1'):
                    require(blob.hexdigest() == asset['expected_git_blob_sha1'], 'Git blob mismatch: ' + str(path))
                self.receipts[asset['path']] = dict(
                    url=asset['url'], bytes=size, sha256=sha256.hexdigest(), checked_at=now(),
                    upstream_checksum_verified=bool(asset.get('expected_sha256') or asset.get('expected_git_blob_sha1')),
                    git_blob_sha1=blob.hexdigest())
            self.check(name + ' release sizes and hashes', files=len(release['assets']))
            write_json(self.data / release['directory'] / 'source_manifest.json', dict(
                source=release['source'], github_revisions=self.plan['github_revisions'],
                hf_repo=self.plan['hf_repo'] if name != 'IR2R-CE' else None,
                hf_revision=self.plan['hf_revision'] if name != 'IR2R-CE' else None,
                assets={a['path']: self.receipts[a['path']] for a in release['assets']}))
        self.report['downloaded_files'] = len(self.receipts)
        self.report['downloaded_bytes'] = sum(r['bytes'] for r in self.receipts.values())
        write_json(self.data / 'splits/ce_expansion_download_receipts.json', self.receipts)

    def scene(self, scene):
        scene = scene.removeprefix('data/scene_datasets/')
        if scene in self.scenes:
            return self.scenes[scene]
        relative = PurePosixPath(scene)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe scene path: ' + scene)
        path = self.data / 'scene_datasets' / relative
        navmesh = path.with_suffix('.navmesh')
        missing = [str(p.relative_to(self.data)) for p in (path, navmesh) if not p.is_file()]
        if path.is_file():
            with path.open('rb') as stream:
                header = stream.read(12)
            require(len(header) == 12 and header[:4] == b'glTF', 'Invalid GLB header: ' + scene)
            _, version, size = struct.unpack('<4sII', header)
            require(version == 2 and size == path.stat().st_size, 'Invalid GLB size/version: ' + scene)
        if navmesh.is_file():
            require(navmesh.stat().st_size > 16, 'Empty navmesh: ' + str(navmesh))
        self.scenes[scene] = missing
        return missing

    def scene_summary(self, scenes):
        missing = {s: self.scene(s) for s in sorted(scenes) if self.scene(s)}
        return dict(scene_count=len(scenes), required_scenes=sorted(scenes),
                    scene_dependencies_ready=not missing, missing_scene_assets=missing)

    def extract_maps(self, root):
        with zipfile.ZipFile(root / 'known_maps.zip') as archive:
            require(archive.testzip() is None, 'Known maps ZIP checksum failed')
            files = 0
            for info in archive.infolist():
                rel = PurePosixPath(info.filename)
                require(not rel.is_absolute() and '..' not in rel.parts
                        and rel.parts[0] == 'known_maps'
                        and not stat.S_ISLNK(info.external_attr >> 16), 'Unsafe map archive entry')
                path = root / rel
                if info.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                payload = archive.read(info)
                require((zlib.crc32(payload) & 0xffffffff) == info.CRC, 'Map CRC mismatch')
                if path.exists():
                    require(path.read_bytes() == payload, 'Conflicting map file: ' + str(path))
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(path.suffix + '.tmp')
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                with zipfile.ZipFile(path) as npz:
                    require(npz.testzip() is None and npz.namelist(), 'Corrupt NPZ: ' + str(path))
                    for member in npz.namelist():
                        require(member.endswith('.npy') and npz.read(member)[:6] == b'\x93NUMPY',
                                'Invalid numpy map: ' + str(path))
                files += 1
        self.check('IR2R-CE known map archive, extracted files, and nested NPZ CRCs', files=files)
        return files

    def ir2r(self):
        root = self.data / 'datasets/IR2R-CE'
        tours = read(root / 'tours.json')
        ground_truth = read(root / 'gt_ndtw.json')
        maps = self.extract_maps(root)
        splits = {}
        augmentation_missing = []
        all_scenes = set()
        for split, by_scene in tours.items():
            augmented = split == 'envdrop'
            episodes_dir = 'R2R_VLNCE_v1-3_preprocessed' if augmented else 'R2R_VLNCE_v1-3'
            source = self.data / 'datasets' / episodes_dir / split / (split + '.json.gz')
            eps = read(source)['episodes']
            by_id = {str(e['episode_id']): e for e in eps}
            require(len(by_id) == len(eps), 'Duplicate R2R-CE source IDs')
            flattened = [t for scene_tours in by_scene.values() for t in scene_tours]
            ids = [str(i) for t in flattened for i in t]
            require(len(ids) == len(set(ids)), 'Duplicate IR2R tour episode IDs')
            missing = set(ids) - by_id.keys()
            require(augmented or not missing, 'IR2R benchmark episode missing')
            scenes = set()
            for scan, scene_tours in by_scene.items():
                for tour in scene_tours:
                    require(len(tour) >= 2, 'Unexpected singleton IR2R tour')
                    for eid in tour:
                        episode = by_id.get(str(eid))
                        if episode is None:
                            augmentation_missing.append(dict(episode_id=str(eid), scene=scan))
                            continue
                        require(Path(episode['scene_id']).stem == scan, 'IR2R tour scene mismatch')
                        scenes.add(episode['scene_id'])
            if not augmented:
                gt = ground_truth[split]
                require(set(gt) == {str(i) for i in range(len(flattened))}, 'Missing IR2R target tour')
                for tid, locations in gt.items():
                    allowed = {str(i) for i in flattened[int(tid)]}
                    require(locations and all(vector(x['position'], 3) for x in locations), 'Invalid IR2R targets')
                    require({str(x['episode_id']) for x in locations} == allowed, 'IR2R target episode coverage')
            summary = self.scene_summary(scenes)
            require(summary['scene_dependencies_ready'], 'IR2R Matterport3D scene missing')
            splits[split] = dict(tours=len(flattened), referenced_episodes=len(ids),
                                 available_episodes=len(ids)-len(missing), missing_episode_count=len(missing),
                                 target_tours=len(ground_truth.get(split, {})), **summary)
            all_scenes.update(scenes)
            self.check('IR2R-CE ' + split + ' tours and episode/scene references', **splits[split])
        required_scans = {Path(s).stem for s in all_scenes}
        for semantics in ('gt_semantics', 'predicted_semantics'):
            require(all((root / 'known_maps' / semantics / (s + '.npz')).is_file() for s in required_scans),
                    'Known maps missing a referenced scene')
        write_json(root / 'envdrop_missing_episode_ids.json', augmentation_missing)
        for alias, target in [('episodes', self.data / 'datasets/R2R_VLNCE_v1-3'),
                              ('preprocessed', self.data / 'datasets/R2R_VLNCE_v1-3_preprocessed'),
                              ('scene_datasets', self.data / 'scene_datasets')]:
            link(root / alias, target)
        # Default paths used by the official IVLN-CE configuration.
        for name in ('tours.json', 'gt_ndtw.json', 'known_maps'):
            link(self.data / name, root / name)
        limits = []
        if augmentation_missing:
            limits.append(f'IR2R-CE optional EnvDrop tours reference {len(augmentation_missing)} episode IDs absent '
                          'from the published R2R-CE v1-3 preprocessed EnvDrop file. Core train/validation tours '
                          'are complete. Original files are preserved; missing IDs are listed in '
                          'datasets/IR2R-CE/envdrop_missing_episode_ids.json.')
        self.report['release_limits'].extend(limits)
        self.report['datasets']['IR2R-CE'] = dict(
            paths=['datasets/IR2R-CE'], notes='414 core tours; tour targets and 142 known maps; shared R2R-CE episodes',
            annotations_verified=True, scene_dependencies_ready=True, status='Ready; optional augmentation caveat',
            splits=splits, known_map_files=maps, release_limits=limits)

    def episodes(self, paths, gt_path=None):
        ids, scenes = set(), set()
        one_point, zero_distance, starts_at_goal = 0, 0, 0
        for path in paths:
            content = read(path)
            rows = content if isinstance(content, list) else content['episodes']
            require(isinstance(rows, list), 'Episodes must be a list: ' + str(path))
            for row in rows:
                eid = str(row['episode_id'])
                require(eid not in ids, 'Duplicate episode ID: ' + str(path) + ':' + eid)
                ids.add(eid)
                scenes.add(row['scene_id'])
                require(vector(row['start_position'], 3) and vector(row['start_rotation'], 4), 'Invalid start pose')
                require(abs(sum(x*x for x in row['start_rotation']) - 1.0) < 1e-3, 'Non-unit rotation')
                require(row['goals'] and all(vector(g['position'], 3) for g in row['goals']), 'Invalid goals')
                require(row['reference_path'] and all(vector(p, 3) for p in row['reference_path']), 'Invalid path')
                instruction = row['instruction']
                require(isinstance(instruction['instruction_text'], str) and instruction['instruction_text'].strip(),
                        'Missing instruction')
                require(isinstance(instruction['instruction_tokens'], list), 'Invalid instruction tokens')
                distance = row['info']['geodesic_distance']
                require(isinstance(distance, (int, float)) and math.isfinite(distance) and distance >= 0, 'Invalid distance')
                one_point += len(row['reference_path']) == 1
                zero_distance += distance == 0
                starts_at_goal += row['start_position'] == row['goals'][0]['position']
        require(ids, 'Empty split')
        result = dict(episodes=len(ids), source_files=len(paths), one_point_paths=one_point,
                      zero_distance_episodes=zero_distance, start_equals_goal_episodes=starts_at_goal,
                      **self.scene_summary(scenes))
        if gt_path is not None:
            targets = read(gt_path)
            require(set(targets) == ids, 'Ground truth episode IDs differ: ' + str(gt_path))
            for target in targets.values():
                require(target['locations'] and all(vector(p, 3) for p in target['locations']), 'Invalid target positions')
                require(isinstance(target['actions'], list)
                        and all(isinstance(a, int) for a in target['actions']), 'Invalid oracle actions')
            result['target_episodes'] = len(targets)
        else:
            result['target_episodes'] = None
        return result

    def dynam3d(self, name, prefix):
        root = self.data / 'datasets' / name
        original = root / 'original/data/datasets'
        train_dir = original / (prefix + '_training_data')
        train_gt = original / (prefix + '_train_gt.json')
        splits = {'train': self.episodes(sorted(train_dir.glob('*.json')), train_gt if train_gt.exists() else None)}
        link(root / 'train', train_dir)
        if train_gt.exists():
            link(root / 'train_gt.json', train_gt)
        for split in ('val_unseen', 'test'):
            source = original / (prefix + '_' + split + '_data.json')
            gt = original / (prefix + '_' + split + '_gt.json')
            if not source.exists():
                continue
            splits[split] = self.episodes([source], gt)
            compressed_copy(source, root / split / (split + '.json.gz'))
            compressed_copy(gt, root / split / (split + '_gt.json.gz'))
        link(root / 'scene_datasets', self.data / 'scene_datasets')
        for split, result in splits.items():
            self.check(name + ' ' + split + ' episode schema, IDs and published targets', **result)
        missing = {s: assets for split in splits.values() for s, assets in split['missing_scene_assets'].items()}
        write_json(root / 'required_scenes.json', dict(
            scene_root='../../scene_datasets', missing_scene_count=len(missing), missing_assets=missing,
            required_scenes=sorted({s for split in splits.values() for s in split['required_scenes']})))
        ready = not missing
        limits = []
        if prefix == 'reverie':
            require(not missing, 'REVERIE-CE Matterport3D scene missing')
            limits = [
                'REVERIE-CE is the Dynam3D continuous navigation conversion. No separate object-selection labels '
                'are added by this conversion; original discrete REVERIE annotations remain available separately.',
                'Dynam3D publishes no REVERIE-CE val_seen split or separate dense training oracle file. '
                'Training episodes include reference paths.',
                f"All {splits['test']['episodes']:,} published REVERIE-CE test episodes have one-point paths, "
                'zero distance and start=goal placeholders. These files are preserved for provenance, '
                'not valid full test ground truth; use val_unseen for public evaluation.'
            ]
            require(splits['test']['episodes'] == splits['test']['one_point_paths']
                    == splits['test']['zero_distance_episodes'] == splits['test']['start_equals_goal_episodes'],
                    'Recheck REVERIE test release caveat')
        else:
            limits = ['The selected Dynam3D NavRAG-CE release publishes train and val_unseen; no separate test or val_seen.']
            if missing:
                self.report['pending_dependencies'].append(
                    f'NavRAG-CE training requires {len(missing)} absent HM3D scene meshes/navmeshes. '
                    'An approved HM3D download or existing local scene directory is needed. '
                    'The val_unseen split uses existing Matterport3D scenes and its data dependencies are ready.')
        self.report['release_limits'].extend(limits)
        self.report['datasets'][name] = dict(
            paths=['datasets/' + name], annotations_verified=True, scene_dependencies_ready=ready,
            status='Ready; test placeholders documented' if prefix == 'reverie' else (
                'Ready' if ready else 'Validation ready; training needs HM3D'),
            notes=f"{splits['train']['episodes']:,} train and {splits['val_unseen']['episodes']:,} val_unseen episodes"
                  + ('; public test placeholder files retained' if prefix == 'reverie' else '; training oracle paths included'),
            splits=splits, missing_scene_count=len(missing), release_limits=limits)

    def run(self):
        self.hashes()
        self.ir2r()
        self.dynam3d('REVERIE-CE', 'reverie')
        self.dynam3d('NavRAG-CE', 'navrag')
        self.report.update(downloads_verified=True, annotations_verified=True,
                           finished_at=now(),
                           validation_scope='Release hashes, JSON schema, ID/target coverage, map CRCs, '
                           'and scene mesh/navmesh files. Simulator execution is not part of this data audit.')
        self.publish()

    def publish(self):
        """Reflect explicit user scope choices without claiming absent assets exist."""
        choices = self.plan.get('user_choices', {})
        skipped = []
        if choices.get('skip_navrag_hm3d') and not self.report['datasets']['NavRAG-CE']['scene_dependencies_ready']:
            self.report['pending_dependencies'] = [
                note for note in self.report['pending_dependencies'] if not note.startswith('NavRAG-CE training')]
            count = self.report['datasets']['NavRAG-CE']['missing_scene_count']
            skipped.append(f'NavRAG-CE: download of {count} HM3D training scenes skipped at the user\'s request. '
                           'Annotations and oracle paths are verified; val_unseen uses existing MP3D scenes. '
                           'Training cannot run without the omitted HM3D scene assets.')
            self.report['datasets']['NavRAG-CE']['status'] = 'Validation ready; HM3D training scenes skipped'
        self.report['skipped_dependencies'] = skipped
        self.report['user_choices'] = choices
        self.report['scene_dependencies_ready'] = all(
            item['scene_dependencies_ready'] for item in self.report['datasets'].values())
        self.report['passed'] = bool(self.report['downloads_verified'] and self.report['annotations_verified']
                                     and self.report['scene_dependencies_ready'])
        self.report['requested_scope_complete'] = bool(
            self.report['downloads_verified'] and self.report['annotations_verified']
            and not self.report['pending_dependencies'])
        self.report['published_at'] = now()
        write_json(self.data / 'splits/ce_expansion_completion_report.json', self.report)
        write_json(self.data / 'splits/ce_expansion_status.json', dict(
            phase=('annotations_complete_waiting_for_hm3d' if self.report['pending_dependencies'] else
                   'complete_with_skipped_hm3d' if skipped else 'complete'),
            downloads_running=False, downloads_verified=True, updated_at=now(),
            requested_scope_complete=self.report['requested_scope_complete'],
            requested_datasets=self.plan['requested_datasets'], pending_dependencies=self.report['pending_dependencies'],
            skipped_dependencies=skipped))
        Worker(self.data, plan()).write_inventory()
        print(json.dumps({k: v for k, v in self.report.items() if k not in ('checks', 'datasets')}, indent=2))


if __name__ == '__main__':
    os.umask(0o007)
    Audit(DATA).run()
