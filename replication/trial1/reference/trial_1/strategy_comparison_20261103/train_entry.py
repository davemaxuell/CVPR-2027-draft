"""Guard the unchanged native trainer with branch identity and atomic resume state."""
import fcntl
import importlib.util
import os
from pathlib import Path
import sys
import time
from context import *


def atomic_checkpoint_writer(original, identity_sha):
    def save(path, policy, optimizer, scheduler, step, samples_seen, keep_states, run_dir):
        if path.exists():
            raise RuntimeError('Refusing to overwrite an existing checkpoint: ' + str(path))
        temporary = path.with_name('.' + path.name + '.partial_' + str(time.time_ns()))
        original(temporary, policy, optimizer, scheduler, step, samples_seen, keep_states, run_dir)
        files = {p.name: sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        assert {'model.safetensors','master_weights.safetensors','trainer_state.pt'} <= set(files)
        atomic_json(temporary / 'complete.json', dict(status='complete', step=step, samples_seen=samples_seen,
            training_identity_sha256=identity_sha, files=files, time=utc()))
        os.replace(temporary, path)
    return save


def main():
    check_authorization()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Only GPU 0 is authorized.')
    branch = os.environ['ROLEVLN_STRATEGY_BRANCH']
    identity = binding(branch)
    check = read_json(HERE / 'evidence/implementation_checks.json', {})
    assert check.get('status') == 'complete' and check['source_snapshot_sha256'] == verify_source()
    f_checkpoint()
    module_spec = importlib.util.spec_from_file_location('strategy_native_trainer', SOURCE / 'tools/train_stream_policy.py')
    trainer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(trainer)
    args = trainer.parser('streamvln_stage2').parse_args()
    training_args_check(args, branch, identity)
    run = Path(spec(branch)['run_dir']); run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = read_json(run / 'training_identity.json')
    if prior and prior != identity:
        raise RuntimeError('Training identity changed; resume is refused.')
    atomic_json(run / 'training_identity.json', identity)
    identity_sha = sha256(run / 'training_identity.json')
    if args.resume:
        ck = Path(args.resume); proof = read_json(ck / 'complete.json', {})
        assert proof.get('status') == 'complete' and proof['training_identity_sha256'] == identity_sha
        for name, digest in proof['files'].items():
            if sha256(ck / name) != digest:
                raise RuntimeError('Recovery checksum changed: ' + name)
    else:
        assert sha256(Path(identity['initial_checkpoint']) / 'model.safetensors') == identity['initial_model_sha256']
    trainer.save_checkpoint = atomic_checkpoint_writer(trainer.save_checkpoint, identity_sha)
    trainer.main()
    status = read_json(run / 'status.json')
    assert status['status'] == 'complete' and status['steps'] == identity['updates']
    assert status['samples_seen'] == identity['samples']
    status.update(trial_id='trial_1', strategy=branch, initialized_from=identity['initial_checkpoint'],
                  training_identity_sha256=identity_sha, recovery_states_retained=True)
    atomic_json(run / 'status.json', status)


if __name__ == '__main__':
    main()
