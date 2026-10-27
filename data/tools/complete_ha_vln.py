#!/usr/bin/env python3
"""Extract and audit the pinned public HA-VLN 2.0 release without a simulator."""
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import zipfile
import zlib

from download_ha_vln import DATA, ROOT, PLAN, now, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def link(path, target):
    require(target.exists(), 'Missing link target: ' + str(target))
    if path.exists() or path.is_symlink():
        require(path.resolve() == target.resolve(), 'Conflicting path: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent), target_is_directory=target.is_dir())


def crc32(path):
    result = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result = zlib.crc32(block, result)
    return result & 0xffffffff


def glb(path):
    with path.open('rb') as stream:
        header = stream.read(12)
    require(len(header) == 12, 'Truncated GLB: ' + str(path))
    magic, version, size = struct.unpack('<4sII', header)
    require(magic == b'glTF' and version == 2 and size == path.stat().st_size,
            'Invalid GLB header/size: ' + str(path))


def extract():
    archive_path = ROOT / 'original/huggingface/HAPS2_0/HAPS2_0.zip'
    output = ROOT / 'HAPS2_0'
    output.mkdir(exist_ok=True)
    names = set()
    motions = defaultdict(Counter)
    total_bytes = 0
    files = 0
    glbs = 0
    configurations = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            rel = PurePosixPath(member.filename)
            require(rel.parts and not rel.is_absolute() and '..' not in rel.parts
                    and '\\' not in member.filename and not stat.S_ISLNK(member.external_attr >> 16),
                    'Unsafe archive member: ' + member.filename)
            parts = list(rel.parts)
            if parts[0] in ('HAPS2_0', 'human_motion_glbs_v3'):
                parts.pop(0)
            if not parts or member.is_dir():
                continue
            normalized = '/'.join(parts)
            require(normalized not in names, 'Duplicate normalized ZIP path: ' + normalized)
            names.add(normalized)
            target = output.joinpath(*parts)
            require(not target.is_symlink() and not any(p.is_symlink() for p in target.parents),
                    'Symlink extraction path: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                require(target.stat().st_size == member.file_size and crc32(target) == member.CRC,
                        'Existing extracted file differs: ' + str(target))
            else:
                temporary = target.with_name(target.name + '.extract-partial')
                require(not temporary.is_symlink(), 'Unsafe extraction temporary')
                with archive.open(member) as source, temporary.open('wb') as out:
                    shutil.copyfileobj(source, out, 1024**2)
                require(temporary.stat().st_size == member.file_size, 'Extraction size mismatch')
                temporary.replace(target)
            if target.suffix.lower() == '.glb':
                glb(target)
                glbs += 1
            elif target.name.endswith('.object_config.json'):
                config = load(target)
                for field in ('render_asset', 'collision_asset'):
                    if config.get(field):
                        asset = PurePosixPath(config[field])
                        require(not asset.is_absolute() and '..' not in asset.parts
                                and (target.parent / asset).is_file(),
                                'Missing or unsafe motion config asset: ' + str(target))
                configurations += 1
            motions[parts[0]][target.suffix.lower()] += 1
            total_bytes += member.file_size
            files += 1
            if files % 10000 == 0:
                print('Extracted and CRC-verified', files, 'motion files', flush=True)
    require(len(motions) == 486, 'Unexpected number of published HAPS motions')
    require(glbs == 486 * 120, 'HAPS GLB frame coverage differs from the release')
    require(configurations == glbs, 'Motion object configurations do not cover every GLB frame')
    require(all(counts['.glb'] == 120 for counts in motions.values()), 'Incomplete HAPS motion')
    result = dict(files=files, motions=len(motions), glb_frames=glbs, object_configurations=configurations,
                  extracted_bytes=total_bytes, all_zip_crcs_verified=True,
                  file_types=dict(sum(motions.values(), Counter())))
    write(ROOT / 'haps_extraction_report.json', result)
    return result, set(motions)


def vector(value, count):
    return isinstance(value, list) and len(value) == count and all(
        isinstance(v, (float, int)) and math.isfinite(v) for v in value)


def episodes():
    root = ROOT / 'original/huggingface/HA-R2R'
    summaries = {}
    all_scenes = set()
    for split in ('train', 'val_seen', 'val_unseen'):
        variants = []
        for suffix in ('', '_bertidx'):
            payload = load(root / split / (split + suffix + '.json.gz'))
            eps = payload['episodes']
            ids = [str(e['episode_id']) for e in eps]
            require(len(ids) == len(set(ids)), 'Duplicate episode IDs in ' + split + suffix)
            scenes = set()
            for episode in eps:
                require(vector(episode['start_position'], 3) and vector(episode['start_rotation'], 4),
                        'Invalid start pose')
                require(episode['goals'] and all(vector(g['position'], 3) for g in episode['goals']),
                        'Invalid episode goals')
                require(episode['reference_path'] and all(vector(v, 3) for v in episode['reference_path']),
                        'Invalid reference path')
                require(episode['instruction']['instruction_text'].strip(), 'Empty instruction')
                require(episode['instruction']['instruction_tokens'], 'Missing instruction tokens')
                scene = PurePosixPath(episode['scene_id'])
                require(not scene.is_absolute() and '..' not in scene.parts and scene.parts[0] == 'mp3d',
                        'Unexpected episode scene path')
                scenes.add(str(scene))
            variants.append({str(e['episode_id']): e for e in eps})
            summaries[split + suffix] = dict(episodes=len(eps), scenes=len(scenes))
            all_scenes.update(scenes)
        require(variants[0].keys() == variants[1].keys(), 'Token variants have different episode IDs')
        for key, left in variants[0].items():
            right = variants[1][key]
            for field in ('scene_id', 'start_position', 'start_rotation', 'goals', 'reference_path'):
                require(left[field] == right[field], 'Token variants differ in navigation fields')
            require(left['instruction']['instruction_text'] == right['instruction']['instruction_text'],
                    'Token variants have different instruction text')
    for scene in all_scenes:
        path = DATA / 'scene_datasets' / scene
        glb(path)
        require(path.with_suffix('.navmesh').is_file() and path.with_suffix('.navmesh').stat().st_size > 16,
                'Missing navmesh: ' + scene)
    return summaries, all_scenes


def annotations(payload, motion_names):
    require(isinstance(payload, dict) and payload, 'Empty/invalid human motion annotations')
    humans = 0
    motion_refs = set()
    extra_frames = []
    for scan, placements in payload.items():
        require(isinstance(placements, dict), 'Invalid scene placement annotations')
        scene = DATA / 'scene_datasets/mp3d' / scan / (scan + '.glb')
        glb(scene)
        require(scene.with_suffix('.navmesh').is_file(), 'Annotated scene navmesh missing')
        for human_id, placement in placements.items():
            category = str(placement['category']).replace(' ', '_').replace('/', '_')
            motion = category + '_' + str(placement['index'])
            require(motion in motion_names, 'Annotated motion not present in HAPS: ' + motion)
            for field in ('translation', 'rotation'):
                poses = placement[field]
                # The simulator indexes poses with frame_id modulo 120. Retain
                # any extra source entries and report them without truncating.
                require(len(poses) >= 120 and all(vector(v, 3) for v in poses),
                        'Invalid human motion poses for ' + motion)
            if len(placement['translation']) != 120 or len(placement['rotation']) != 120:
                extra_frames.append(dict(scan=scan, human_id=human_id, motion=motion,
                    translation_frames=len(placement['translation']),
                    rotation_frames=len(placement['rotation']), simulator_frames=120))
            humans += 1
            motion_refs.add(motion)
    return dict(scenes=len(payload), placed_humans=humans, distinct_motion_references=len(motion_refs),
                all_motion_and_scene_references_verified=True, extra_pose_frames=extra_frames,
                simulator_frame_index_source=(
                    'https://github.com/UWMILab/HA-VLN/blob/4e1d1fe4ceebee7253d3ba53c04468a8bc2c16c8/HASimulator/environments.py#L122'))


def organize():
    hf = ROOT / 'original/huggingface'
    github = ROOT / 'original/github'
    for name in ('HA-R2R', 'checkpoints'):
        link(ROOT / name, hf / name)
    for name in ('Multi-Human-Annotations', 'HA-VLN-DE', 'ddppo-models'):
        link(ROOT / name, github / 'Data' / name)
    tools = ROOT / 'HA-R2R-tools'
    tools.mkdir(exist_ok=True)
    for source in sorted((github / 'Data/HA-R2R-tools').iterdir()):
        link(tools / source.name, source)
    for source in sorted((hf / 'HA-R2R-tools').iterdir()):
        if not (tools / source.name).exists():
            link(tools / source.name, source)
    link(ROOT / 'HAPS2_0/README.md', hf / 'HAPS2_0/README.md')
    link(ROOT / 'scene_datasets', DATA / 'scene_datasets')
    link(ROOT / 'LICENSE', github / 'LICENSE')
    link(ROOT / 'README.upstream.md', github / 'README.md')


def main():
    release = load(PLAN)
    receipts = load(DATA / 'splits/ha_vln_download_receipts.json')
    require(len(receipts) == len(release['assets']) and all(x['upstream_checksum_verified'] for x in receipts.values()),
            'Release files must pass size and upstream checksum verification first')
    haps, motion_names = extract()
    organize()
    splits, scenes = episodes()
    motion_path = ROOT / 'Multi-Human-Annotations/human_motion.json'
    human_annotations = load(motion_path)
    human_summary = annotations(human_annotations, motion_names)
    other_json = []
    for path in (ROOT / 'HA-R2R-tools').iterdir():
        if path.name.endswith(('.json', '.json.gz')):
            load(path)
            other_json.append(path.name)
    checkpoint = ROOT / 'checkpoints/HA-VLN-CMA/ckpt.39.pth'
    require(checkpoint.is_file(), 'CMA checkpoint missing')
    # Integrity checks only; never unpickle downloaded model weights to audit them.
    if zipfile.is_zipfile(checkpoint):
        with zipfile.ZipFile(checkpoint) as archive:
            require(archive.testzip() is None, 'Checkpoint archive CRC mismatch')
    limits = [
        'HA-VLN 2.0 mirrors all public train, val_seen and val_unseen episode files in both published tokenizations. No public test episodes are included in this release.',
        'HA-VLN 2.0 includes its published CMA checkpoint. Simulator/Docker installation and training/evaluation execution are outside this dataset download.',
        'HA-VLN-DE currently has only a README placeholder in this source repository; no separate discrete episode release is fabricated.',
    ]
    if human_summary['extra_pose_frames']:
        limits.append(
            'HA-VLN has ' + str(len(human_summary['extra_pose_frames'])) +
            ' published human placements with extra pose entries beyond the simulator\'s 120-frame cycle. '
            'All required frames and referenced assets are valid; the extra entries are preserved unchanged '
            'and listed in the completion report.')
    report = dict(checked_at=now(), dataset='HA-VLN 2.0', passed=True,
        downloads_verified=True, annotations_verified=True, scene_dependencies_ready=True,
        paths=['datasets/HA-VLN'], status='Verified; released CMA checkpoint included',
        notes='HA-R2R train/validation episodes, HAPS 2.0 motions, human annotations, and CMA checkpoint',
        downloaded_files=len(receipts), downloaded_bytes=sum(v['bytes'] for v in receipts.values()),
        haps=haps, splits=splits, required_scenes=sorted(scenes),
        human_placements=human_summary,
        annotation_root_type=type(human_annotations).__name__, annotation_top_level_entries=len(human_annotations),
        auxiliary_json_files=other_json, release_limits=limits,
        checks=[dict(name=name, passed=True) for name in (
            'All published file sizes and upstream hashes', 'HAPS archive CRCs and 486 complete 120-frame GLB motions',
            'All extracted GLB headers and file sizes', 'All episode JSONs, poses, paths, and token variant consistency',
            'Every referenced Matterport3D mesh and navmesh', 'Human pose frame coverage, motion/scene references, and auxiliary JSONs',
            'Published CMA checkpoint integrity without unpickling')])
    write(DATA / 'splits/ha_vln_completion_report.json', report)
    text = '\n'.join([
        '# HA-VLN 2.0 local dataset', '',
        'All publicly released dataset files and the CMA checkpoint are downloaded at pinned revisions.', '',
        '- `HA-R2R/`: train, val_seen and val_unseen, with original and BERT-indexed tokens.',
        '- `HAPS2_0/`: 486 motions, 120 GLB frames per motion; all archive files retained.',
        '- `Multi-Human-Annotations/human_motion.json`: official human placement and motion annotations.',
        '- `HA-R2R-tools/`: vocabulary, embeddings and published collision statistics.',
        '- `checkpoints/HA-VLN-CMA/ckpt.39.pth`: published baseline checkpoint.',
        '- `scene_datasets/`: relative link to the existing shared Matterport3D scenes.',
        '- `original/huggingface/` and `original/github/`: unmodified source files.', '',
        'Sources: https://github.com/UWMILab/HA-VLN and https://huggingface.co/datasets/fly1113/HA-VLN.',
        'Hugging Face revision: ' + release['hf_revision'],
        'GitHub revision: ' + release['github_revision'], '',
        'Verification: `source_manifest.json`, `haps_extraction_report.json`, and `data/splits/ha_vln_completion_report.json`.', '',
        *limits, '',
    ])
    (ROOT / 'README.local.md').write_text(text)
    from download_additional_datasets import Worker, plan
    Worker(DATA, plan()).write_inventory()
    from download_ha_vln import status
    status('complete', downloads_verified=True, finished_at=now(),
           downloaded_bytes=report['downloaded_bytes'], report='splits/ha_vln_completion_report.json')
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'required_scenes')}, indent=2), flush=True)


if __name__ == '__main__':
    os.umask(0o007)
    main()
