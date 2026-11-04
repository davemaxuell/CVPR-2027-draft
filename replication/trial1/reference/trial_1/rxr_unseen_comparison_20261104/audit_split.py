"""CPU-only audit of the complete held-out English-guide RxR-CE split."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
sys.path.insert(0, str(WORKSPACE / 'queue_after_stageA'))
from common import atomic_json, sha256, utc


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main():
    data = WORKSPACE / 'data'
    source = data / 'datasets/RxR_VLNCE_v0'
    annotations = source / 'val_unseen/val_unseen_guide.json.gz'
    gt_path = source / 'val_unseen/val_unseen_guide_gt.json.gz'
    follower_gt_path = source / 'val_unseen/val_unseen_follower_gt.json.gz'
    train_path = source / 'train/train_guide.json.gz'
    raw = read_gzip(annotations)['episodes']
    selected = [row for row in raw if row['instruction']['language'] in {'en-US', 'en-IN'}]
    selected.sort(key=lambda row: (row['scene_id'], str(row['episode_id'])))
    assert len(selected) == 3669 and len(raw) == 11006
    ids = [str(row['episode_id']) for row in selected]
    assert len(ids) == len(set(ids))
    assert all(row['info']['role'] == 'guide' for row in selected)
    scenes = sorted({row['scene_id'] for row in selected})
    assert len(scenes) == 11
    train = read_gzip(train_path)['episodes']
    train_scenes = {row['scene_id'] for row in train}
    r2r_train = data / 'datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz'
    train_scenes.update(row['scene_id'] for row in read_gzip(r2r_train)['episodes'])
    assert not set(scenes) & train_scenes
    guide_gt = read_gzip(gt_path)
    follower_gt = read_gzip(follower_gt_path)
    overlap = set(ids) & set(follower_gt)
    # Upstream NDTW merges both role maps. A colliding key must never replace
    # a guide reference with a different follower trajectory.
    assert all(guide_gt[key] == follower_gt[key] for key in overlap)
    episodes = {}
    for row in selected:
        key = str(row['episode_id'])
        reference = guide_gt[key]
        assert reference['locations'] and reference['actions']
        assert all(math.isfinite(v) for point in reference['locations'] for v in point)
        assert set(reference['actions']) <= {0, 1, 2, 3}
        assert reference['actions'][-1] == 0
        text = row['instruction']['instruction_text']
        assert text.strip()
        episodes[key] = dict(scene_id=row['scene_id'], language=row['instruction']['language'],
            instruction_id=str(row['instruction']['instruction_id']), role='guide',
            instruction_sha256=hashlib.sha256(text.encode()).hexdigest(),
            annotation_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
    assets = {}
    for name in scenes:
        glb = data / 'scene_datasets' / name
        for path in [glb, glb.with_suffix('.navmesh')]:
            assert path.is_file(), str(path)
            assets[str(path)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    for path in [annotations, gt_path, follower_gt_path, train_path, r2r_train]:
        assets[str(path)] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    audit = dict(status='complete', passed=True, time=utc(), dataset='rxr', split='val_unseen',
        release='RxR_VLNCE_v0', role='guide', languages=['en-US', 'en-IN'], expected=3669,
        selected_episode_ids=ids, episodes=episodes, scene_ids=scenes,
        language_counts=dict(collections.Counter(row['instruction']['language'] for row in selected)),
        all_guide_episodes=len(raw), excluded_language_counts=dict(collections.Counter(
            row['instruction']['language'] for row in raw if row['instruction']['language'] not in {'en-US', 'en-IN'})),
        evaluation_exclusions=[], train_scene_overlap=[],
        split_local_numeric_id_overlap_with_train=len(set(ids) & {str(row['episode_id']) for row in train}),
        id_namespace='dataset/split/episode_id; numeric IDs are local to each split, not global.',
        selected_guide_follower_gt_overlap=sorted(overlap), assets=assets,
        episode_identity_sha256=hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest())
    atomic_json(HERE / 'split_audit.json', audit)
    print(json.dumps({key:audit[key] for key in ['status','expected','language_counts','train_scene_overlap','episode_identity_sha256']}))


if __name__ == '__main__':
    main()
