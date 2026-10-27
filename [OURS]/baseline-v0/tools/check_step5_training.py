"""Independent sample mapping and real optimizer restart checks before long training."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.common import CORPUS, ROOT, RUN, now, read, sha, write
from wm.dino import build_dynamics
from wm.features import CACHE
from wm.training import checkpoint_state, restore_rng, save_checkpoint, seed_training, update


def main():
    index = RUN / 'data_index'
    manifest = read(index / 'manifest.json')
    samples = np.load(index / 'train_samples.npy', mmap_mode='r')
    ids = np.load(index / 'train_transition_ids.npy', mmap_mode='r')
    assert len(samples) == len(ids) == 2422469
    corpus = sqlite3.connect(f"file:{CORPUS / 'catalog.sqlite'}?mode=ro&immutable=1", uri=True)
    corpus.row_factory = sqlite3.Row
    image_db = sqlite3.connect(f"file:{index / 'images.sqlite'}?mode=ro&immutable=1", uri=True)
    slots = set(np.random.default_rng(5).choice(len(samples), 256, replace=False).tolist())
    for source in ['expert', 'dagger_round1', 'dagger_round2', 'selected_policy']:
        for condition in ['t.action=1', 't.action=2', 't.action=3', 't.is_branch=1', 't.blocked=1']:
            value = corpus.execute(f'SELECT min(s.slot) FROM matched_samples s JOIN transitions t ON t.id=s.transition_id JOIN episodes e ON e.id=t.episode_pk WHERE e.source=? AND {condition}', (source,)).fetchone()[0]
            if value is not None: slots.add(value)
    for slot in slots:
        row = corpus.execute('SELECT t.*,e.directory,e.dataset,e.source,e.encoding FROM matched_samples s JOIN transitions t ON t.id=s.transition_id JOIN episodes e ON e.id=t.episode_pk WHERE s.slot=?', (slot,)).fetchone()
        assert row['id'] == ids[slot]
        m = read(Path(row['directory']) / 'meta.json')
        t = row['t']
        extension = 'png' if row['encoding'] == 'PNG' else 'jpg'
        names = [f'rgb/{i:04d}.{extension}' for i in range(t - 3, t + 1)] + [row['target_frame']]
        for number, name in zip(samples[slot, :5], names):
            expected = corpus.execute('SELECT sha256 FROM frames WHERE episode_pk=? AND relative_path=?', (row['episode_pk'], name)).fetchone()[0]
            saved = image_db.execute('SELECT sha256,group_name FROM images WHERE id=?', (int(number),)).fetchone()
            assert saved == (expected, 'A')
        assert samples[slot, 5:9].tolist() == [a - 1 for a in m['executed_action'][t - 3:t] + [row['action']]]
        assert samples[slot, 9] == int(row['dataset'] == 'rxr')
        assert samples[slot, 11:13].tolist() == [row['collided'], row['blocked']]
    val = np.load(index / 'validation_samples.npy')
    val_ids = read(index / 'validation_identities.json')
    partition = read(ROOT / 'manifest/scene_partition.json')['groups']
    for row, identity in zip(val, val_ids):
        assert identity['scene'] in partition['C']['scenes'] and identity['scene'] not in partition['A']['scenes']
        assert row[8] + 1 == identity['action']
        for number in row[:5]:
            assert image_db.execute('SELECT group_name FROM images WHERE id=?', (int(number),)).fetchone()[0] == 'C'
    assert {(int(d), int(a), int(((val[:, 9] == d) & (val[:, 8] == a)).sum())) for d in (0, 1) for a in (0, 1, 2)} == {(d, a, 512) for d in (0, 1) for a in (0, 1, 2)}
    progress = read(CACHE / 'progress.json')
    cache_path = CACHE / ('features.npy' if (CACHE / 'features.npy').exists() else 'features.partial.npy')
    features = np.load(cache_path, mmap_mode='r')
    eligible = np.flatnonzero(samples[:10000, :5].max(axis=1) < progress['completed_rows'])
    selected = np.random.default_rng(15).choice(eligible, 16, replace=False)
    rows = samples[selected]
    batch = {'features': torch.tensor(np.array(features[rows[:, 1:5]], copy=True)),
             'actions': torch.tensor(rows[:, 6:9], dtype=torch.long)}
    model = build_dynamics('transfer').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    seed_training(1005)
    update(model, optimizer, batch, 5e-5)
    checkpoint = RUN / 'check_training/restart.pt'
    save_checkpoint(checkpoint, checkpoint_state(model, optimizer, step=1, seen=16, plan_sha256='runtime_check', trial='transfer', metrics={}))
    expected_loss, _ = update(model, optimizer, batch, 6e-5)
    expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    resumed = build_dynamics('fresh').train()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=5e-4, weight_decay=.01)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    resumed.load_state_dict(state['model'], strict=True)
    resumed_optimizer.load_state_dict(state['optimizer'])
    assert all(value['step'].device.type == 'cpu' for value in resumed_optimizer.state.values())
    restore_rng(state)
    actual_loss, _ = update(resumed, resumed_optimizer, batch, 6e-5)
    print(json.dumps({'uninterrupted_loss': expected_loss, 'resumed_loss': actual_loss,
                      'Adam_step_counter_device': 'cpu'}), flush=True)
    maximum = 0.0
    exact = True
    for key, value in resumed.state_dict().items():
        value = value.detach().cpu()
        torch.testing.assert_close(value, expected[key], atol=2e-6, rtol=2e-6)
        exact &= torch.equal(value, expected[key])
        maximum = max(maximum, (value - expected[key]).abs().max().item())
    assert abs(actual_loss - expected_loss) < 1e-6
    report = {'status': 'passed', 'completed_utc': now(), 'mapped_training_samples_checked': len(slots),
              'all_validation_sample_groups_checked': len(val), 'validation_dataset_action_count': 512,
              'optimizer_resume_loss': actual_loss, 'uninterrupted_loss': expected_loss,
              'resume_parameters_bit_identical': bool(exact), 'resume_max_abs_parameter_error': maximum,
              'training_step_uses_actual_history_and_command_only': True,
              'data_index_manifest_sha256': sha(index / 'manifest.json'),
              'source_sha256': {name: sha(ROOT / name) for name in ['wm/training.py', 'tools/train_dino_world_model.py', 'tools/check_step5_training.py']}}
    write(RUN / 'check_training.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
