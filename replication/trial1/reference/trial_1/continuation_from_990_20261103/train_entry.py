"""Guarded additional full-data SFT from the preserved stage-two checkpoint 990."""
from datetime import datetime, timezone
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
sys.path.insert(0, str(WORKSPACE / 'queue_after_stageA'))
from common import atomic_json, read_json, sha256


def verify_source():
    snapshot = read_json(HERE / 'SOURCE_SNAPSHOT.json')
    for relative, digest in snapshot['files'].items():
        if sha256(HERE / relative) != digest:
            raise RuntimeError('Frozen continuation source identity changed: ' + relative)
    return sha256(HERE / 'SOURCE_SNAPSHOT.json')


def atomic_checkpoint_writer(original_save, identity_sha):
    def save(path, policy, optimizer, scheduler, step, samples_seen, keep_states, run_dir):
        if path.exists():
            raise RuntimeError('Refusing to overwrite an existing checkpoint: ' + str(path))
        temporary = path.with_name('.' + path.name + '.partial_' + str(time.time_ns()))
        original_save(temporary, policy, optimizer, scheduler, step, samples_seen, keep_states, run_dir)
        files = {p.name: sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        assert {'model.safetensors', 'master_weights.safetensors', 'trainer_state.pt'} <= set(files)
        atomic_json(temporary / 'complete.json', dict(status='complete', step=step, samples_seen=samples_seen,
            training_identity_sha256=identity_sha, files=files, time=datetime.now(timezone.utc).isoformat()))
        os.replace(temporary, path)
    return save


def validate_args(args, auth):
    if Path(args.run_dir).resolve() != Path(auth['run_dir']):
        raise RuntimeError('Refusing to modify another run.')
    if args.recipe != 'streamvln_stage2' or args.train_root != [auth['expert_root']]:
        raise RuntimeError('Continuation must use the approved expert-only SFT recipe.')
    assert args.val_root == [auth['validation_root']]
    assert args.mode == 'full' and args.epochs == 1 and args.max_steps == 0
    assert args.batch_size * args.grad_accum == 128 and args.source_weights is None
    assert args.lr == 2e-5 and args.vision_lr == 5e-6 and args.weight_decay == 0
    assert args.warmup_frac == 0.03 and args.schedule == 'cosine'
    assert args.augment and not args.no_grad_ckpt
    assert args.keep_final_state and args.keep_trainer_states >= 10000
    assert args.datasets == ['r2r', 'rxr'] and args.label_source == 'executed'
    if not args.resume and args.init != auth['initial_checkpoint']:
        raise RuntimeError('This continuation must initialize from checkpoint 990.')
    if args.resume and args.init:
        raise RuntimeError('Resume must restore the new run, not initialize it again.')


def main():
    auth = read_json(HERE / 'authorization.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or auth['gpu'] != 0 or auth['status'] != 'authorized':
        raise RuntimeError('Scoped GPU 0 authorization required.')
    holds = read_json(WORKSPACE / 'queue_after_stageA/expanded_stage2/USER_JOB_HOLDS.json', {}).get('jobs', {})
    if holds.get('trial1_continue_990_pipeline', {}).get('status') == 'held':
        raise RuntimeError('User hold prevents this continuation.')
    source_sha = verify_source()
    audit = read_json(HERE / 'evidence/expert_data_audit.json')
    check = read_json(HERE / 'evidence/implementation_checks.json')
    if not audit.get('passed') or check.get('status') != 'complete' or check['source_snapshot_sha256'] != source_sha:
        raise RuntimeError('Data and implementation checks must pass before training.')
    spec = importlib.util.spec_from_file_location('continuation_navigation_trainer', Path(auth['source_root']) / 'tools/train_stream_policy.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    args = trainer.parser('streamvln_stage2').parse_args()
    validate_args(args, auth)
    run = Path(auth['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    run_lock = (run / 'training.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(trial_id='trial_1', experiment_id=auth['experiment_id'], training_stage='additional_full_split_expert_SFT',
        source_snapshot_sha256=source_sha, data_identity_sha256=sha256(HERE / 'evidence/data_identity.json'),
        data_audit_sha256=sha256(HERE / 'evidence/expert_data_audit.json'),
        source_inventory_sha256=sha256(HERE / 'preparation/inventory.json'),
        train_roots=[auth['expert_root']], validation_root=auth['validation_root'],
        validation_split='official val_seen; no training examples used as held-out validation',
        initialization=auth['initial_checkpoint'], initialization_model_sha256=auth['initial_model_sha256'],
        mode='full', epochs=1, effective_batch=128, fresh_optimizer=True,
        new_dagger_collection=False, dagger_data_replayed=False, custom_saved_places=False)
    run.mkdir(parents=True, exist_ok=True)
    old = read_json(run / 'continuation_identity.json')
    if old and old != identity:
        raise RuntimeError('Continuation identity changed; resume is refused.')
    atomic_json(run / 'continuation_identity.json', identity)
    identity_sha = sha256(run / 'continuation_identity.json')
    if args.resume:
        ck = Path(args.resume).resolve()
        proof = read_json(ck / 'complete.json', {})
        assert ck.parent == run / 'checkpoints'
        assert proof.get('status') == 'complete' and proof['training_identity_sha256'] == identity_sha
        for name, digest in proof['files'].items():
            assert sha256(ck / name) == digest, 'Recovery checksum changed: ' + name
    else:
        assert sha256(Path(auth['initial_checkpoint']) / 'model.safetensors') == auth['initial_model_sha256']
    trainer.save_checkpoint = atomic_checkpoint_writer(trainer.save_checkpoint, identity_sha)
    trainer.main()
    status = read_json(run / 'status.json')
    assert status['status'] == 'complete'
    final = run / 'checkpoints' / ('step_%06d' % status['steps'])
    assert read_json(final / 'complete.json')['training_identity_sha256'] == identity_sha
    status.update(trial_id='trial_1', experiment_id=auth['experiment_id'],
        continuation_identity_sha256=identity_sha, initialized_from=auth['initial_checkpoint'], recovery_states_retained=True)
    atomic_json(run / 'status.json', status)


if __name__ == '__main__':
    main()
