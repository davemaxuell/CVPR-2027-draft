"""Bind every matched Step 4 sample and fixed held-out C validation sample to RGB hashes."""
import hashlib
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.common import CORPUS, DATA, REFERENCE, ROOT, RUN, now, read, sha, write


def main():
    out = RUN / 'data_index'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        print((out / 'manifest.json').read_text()); return
    corpus = read(CORPUS / 'manifest.json')
    reference = read(REFERENCE / 'reference.json')
    assert corpus['status'] == 'complete'
    assert sha(CORPUS / 'catalog.sqlite') == corpus['catalog_sha256']
    identity = reference['policy_identity_sha256']
    assert identity == corpus['selected_policy_identity_sha256']
    partition = read(ROOT / 'manifest/scene_partition.json')['groups']
    scenes_a, scenes_c = (set(partition[x]['scenes']) for x in ['A', 'C'])
    assert not (scenes_a & scenes_c)
    db = sqlite3.connect(f"file:{CORPUS / 'catalog.sqlite'}?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    # A fresh build can replace only this run's incomplete index.
    partial = out / 'images.sqlite.partial'
    partial.unlink(missing_ok=True)
    images = sqlite3.connect(partial)
    images.execute('CREATE TABLE images(id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE, path TEXT, group_name TEXT)')
    images.execute('CREATE TABLE episodes(source TEXT, dataset TEXT, episode_id TEXT, metadata_path TEXT, sha256 TEXT, group_name TEXT)')
    hashes = {}
    pending_images = []

    def image_id(path, expected, group):
        if expected not in hashes:
            number = len(hashes)
            hashes[expected] = number
            pending_images.append((number, expected, str(path), group))
        return hashes[expected]

    train = np.lib.format.open_memmap(out / 'train_samples.npy', mode='w+', dtype=np.int32,
                                     shape=(corpus['counts']['matched'], 13))
    transition_ids = np.lib.format.open_memmap(out / 'train_transition_ids.npy', mode='w+', dtype=np.int64,
                                              shape=(len(train),))
    train[:] = -1
    sources = ['expert', 'dagger_round1', 'dagger_round2', 'selected_policy']
    count = 0
    episodes = list(db.execute('SELECT * FROM episodes ORDER BY id'))
    for index, episode in enumerate(episodes):
        assert episode['scene'] in scenes_a
        directory = Path(episode['directory'])
        path = directory / 'meta.json'
        assert sha(path) == episode['meta_sha256']
        meta = read(path)
        images.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?)',
                       (episode['source'], episode['dataset'], episode['episode_id'], str(path), episode['meta_sha256'], 'A'))
        frames = dict(db.execute('SELECT relative_path,sha256 FROM frames WHERE episode_pk=?', (episode['id'],)))
        extension = 'png' if episode['encoding'] == 'PNG' else 'jpg'
        rows = db.execute('SELECT s.slot,t.* FROM transitions t JOIN matched_samples s ON s.transition_id=t.id WHERE t.episode_pk=? ORDER BY t.id', (episode['id'],))
        for row in rows:
            t = row['t']
            names = [f'rgb/{i:04d}.{extension}' for i in range(t - 3, t + 1)] + [row['target_frame']]
            frame_ids = [image_id(directory / name, frames[name], 'A') for name in names]
            commands = meta['executed_action'][t - 3:t] + [row['action']]
            assert len(commands) == 4 and all(a in (1, 2, 3) for a in commands)
            train[row['slot']] = frame_ids + [a - 1 for a in commands] + [int(episode['dataset'] == 'rxr'), sources.index(episode['source']), row['collided'], row['blocked']]
            transition_ids[row['slot']] = row['id']
            count += 1
        if pending_images:
            images.executemany('INSERT INTO images VALUES (?,?,?,?)', pending_images)
            pending_images.clear()
        if (index + 1) % 500 == 0:
            images.commit()
            write(out / 'progress.json', {'status': 'indexing_train', 'episodes': index + 1, 'total_episodes': len(episodes), 'samples': count, 'unique_images': len(hashes)})
            print(json.dumps({'episodes': index + 1, 'samples': count, 'unique_images': len(hashes)}), flush=True)
    assert count == len(train) and np.all(train >= 0)
    train.flush(); transition_ids.flush()
    # Fixed validation selection: 512 per dataset/action, cycled across C scenes.
    # The complete expert-quality filter is the same one used in policy validation.
    validation_root = DATA / 'expert_groupC_600'
    pools = defaultdict(lambda: defaultdict(list))
    metadata = {}
    for dataset in ['r2r', 'rxr']:
        for path in sorted((validation_root / dataset).glob('*/meta.json')):
            m = read(path)
            assert m['group'] == 'C' and m['scene'] in scenes_c
            if m['metrics']['success'] != 1 or m['truncated'] or m['max_location_deviation_m'] >= 0.5:
                continue
            metadata[str(path)] = m
            actions = m['executed_action']
            assert len(actions) == m['num_frames'] and actions[-1] == 0
            for t, action in enumerate(actions):
                if t >= 3 and action != 0:
                    pools[(dataset, action)][m['scene']].append((str(path), t))
    selected = []
    rng = random.Random(5)
    for dataset in ['r2r', 'rxr']:
        for action in (1, 2, 3):
            groups = pools[(dataset, action)]
            for values in groups.values(): rng.shuffle(values)
            chosen = []
            while len(chosen) < 512:
                before = len(chosen)
                for scene in sorted(groups):
                    if groups[scene] and len(chosen) < 512:
                        chosen.append(groups[scene].pop())
                assert len(chosen) > before, 'Insufficient group-C validation transitions'
            selected.extend(chosen)
    val = np.empty((len(selected), 13), dtype=np.int32)
    val_ids, val_image_hashes = [], {}
    for slot, (path, t) in enumerate(selected):
        path = Path(path)
        m = metadata[str(path)]
        frame_paths = [path.parent / f'rgb/{i:04d}.jpg' for i in range(t - 3, t + 2)]
        ids = []
        for frame in frame_paths:
            if str(frame) not in val_image_hashes:
                val_image_hashes[str(frame)] = sha(frame)
            digest = val_image_hashes[str(frame)]
            assert digest not in hashes or any(r[0] == 'C' for r in images.execute('SELECT group_name FROM images WHERE sha256=?', (digest,))), 'Cross-group byte-identical image needs review'
            ids.append(image_id(frame, digest, 'C'))
        actions = m['executed_action'][t - 3:t + 1]
        assert all(a in (1, 2, 3) for a in actions)
        collided = int(m['collided_after_action'][t])
        blocked = int(actions[-1] == 1 and np.linalg.norm(np.asarray(m['position'][t + 1]) - m['position'][t]) < 1e-5)
        val[slot] = ids + [a - 1 for a in actions] + [int(m['dataset'] == 'rxr'), 0, collided, blocked]
        val_ids.append({'dataset': m['dataset'], 'episode_id': m['episode_id'], 'scene': m['scene'], 't': t, 'action': actions[-1], 'meta_path': str(path), 'meta_sha256': sha(path)})
        if pending_images:
            images.executemany('INSERT INTO images VALUES (?,?,?,?)', pending_images)
            pending_images.clear()
    for path in sorted({p for p, _ in selected}):
        m = metadata[path]
        images.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?)', ('expert_validation', m['dataset'], m['episode_id'], path, sha(path), 'C'))
    np.save(out / 'validation_samples.npy', val)
    write(out / 'validation_identities.json', val_ids)
    images.commit(); images.close(); db.close()
    partial.rename(out / 'images.sqlite')
    manifest = {'status': 'complete', 'format': 'wm_training_samples_v1', 'created_utc': now(),
                'selected_policy_identity_sha256': identity, 'corpus': str(CORPUS), 'corpus_catalog_sha256': corpus['catalog_sha256'],
                'training_group': 'A', 'validation_group': 'C', 'train_samples': len(train), 'validation_samples': len(val),
                'unique_images': len(hashes), 'matched_roots_only': True, 'early_dino_extra_used': False,
                'seed': 5, 'validation_selection': '512 per dataset/action; scene-cycled seeded sampling from validated group-C experts',
                'validation_limit': 'Expert-path validation; new policy failures are represented in group-A training but not this held-out set',
                'validation_diagnostics': {'collisions': int(val[:, 11].sum()), 'blocked_forward': int(val[:, 12].sum())},
                'sample_columns': ['frame_t-3', 'frame_t-2', 'frame_t-1', 'frame_t', 'target_frame', 'action_t-3', 'action_t-2', 'action_t-1', 'candidate_action', 'dataset_0r2r_1rxr', 'source_index', 'collided', 'blocked'],
                'action_indices': {'0': 'FORWARD', '1': 'LEFT', '2': 'RIGHT'}, 'source_order': sources,
                'files': {name: sha(out / name) for name in ['images.sqlite', 'train_samples.npy', 'train_transition_ids.npy', 'validation_samples.npy', 'validation_identities.json']}}
    write(out / 'manifest.json', manifest)
    write(out / 'progress.json', {'status': 'complete', 'train_samples': len(train), 'validation_samples': len(val), 'unique_images': len(hashes)})
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
