"""One additional SFT pass from 990, verification, fresh full R2R evaluation."""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
BASE = HERE.parents[2]
QUEUE = WORKSPACE / 'queue_after_stageA'
sys.path.insert(0, str(QUEUE))
from common import atomic_json, atomic_text, read_json, sha256, utc, process_identity, capture_process
from train_entry import verify_source
from evaluation_checks import verify_split, SPLITS
import worker_stage

POL = '/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python'
SIM = '/home/bufsgpu/yes/envs/rolevln/bin/python'
AUTH = read_json(HERE / 'authorization.json')
SOURCE = Path(AUTH['source_root'])
EVIDENCE = HERE / 'evidence'
STAGES = ['continuation_train', 'checkpoint_verify', 'val_seen', 'val_unseen', 'report']


def progress(stage, child=None, log=None, progress_path=None):
    done = read_json(HERE / 'completion.json', {}).get('status') == 'complete'
    detail = read_json(progress_path, {}) if progress_path else {}
    counts = {k: detail[k] for k in ('step', 'max_steps', 'samples_seen', 'completed', 'expected') if k in detail}
    size = Path(log).stat().st_size if log and Path(log).exists() else 0
    atomic_json(HERE / 'progress.json', dict(status='complete' if done else 'running', stage=stage,
        time=utc(), process=child, counts=counts, activity=[stage, counts, size],
        next_stage=None if stage == 'report' else STAGES[STAGES.index(stage) + 1]))
    lines = ['# Trial 1: full-data continuation from checkpoint 990', '', 'Updated: ' + utc(), '',
        'Status: ' + ('complete' if done else stage) + '; progress: ' + json.dumps(counts) + '.',
        'Initialization: preserved step_000990. This is an additional SFT stage, with a fresh optimizer.',
        'Data: 10,819 R2R + 19,402 usable RxR English-guide expert trajectories; 109,784 eligible windows.',
        'Budget: one epoch, 857 updates at effective batch 128. No additional DAgger collection.',
        'The original checkpoints 623/990 remain immutable. The stopped from-pretrained run and Trial 2 training remain held.', '',
        '| Stage | State | Evidence/log |', '|---|---|---|']
    for name in STAGES:
        row = read_json(HERE / 'stages' / name / 'state.json', {})
        lines.append('| ' + name + ' | ' + row.get('status', 'pending') + ' | ' + row.get('log', '') + ' |')
    atomic_text(HERE / 'STATUS.md', '\n'.join(lines) + '\n')
    path = HERE.parent / 'trial.json'
    trial = read_json(path)
    if trial['continuation_experiment'].get('current_stage') != stage:
        trial.update(status='complete_with_continuation' if done else 'continuation_running', updated_utc=utc())
        trial['continuation_experiment'].update(current_stage=stage, status='complete' if done else 'running')
        atomic_json(path, trial)


for name in ('HERE', 'SOURCE', 'read_json', 'atomic_json', 'utc', 'process_identity', 'capture_process', 'progress'):
    setattr(worker_stage, name, globals()[name])
stage = worker_stage.stage


def training_command():
    run = Path(AUTH['run_dir'])
    windows = read_json(EVIDENCE / 'expert_data_audit.json')['train']['windows']['windows']
    required = math.ceil((windows // 128) / 156) * 64 * 2**30 + 256 * 2**30
    if shutil.disk_usage(WORKSPACE / 'data').free < required:
        raise RuntimeError('Insufficient storage for retained recovery checkpoints; no files deleted.')
    command = [POL, '-B', '-u', str(HERE / 'train_entry.py'), '--recipe', 'streamvln_stage2',
        '--run-dir', str(run), '--train-root', AUTH['expert_root'], '--val-root', AUTH['validation_root'],
        '--keep-trainer-states', '10000', '--keep-final-state']
    checkpoints = sorted((run / 'checkpoints').glob('step_*/complete.json'))
    if checkpoints:
        command += ['--resume', str(checkpoints[-1].parent)]
    else:
        previous = read_json(HERE / 'stages/continuation_train/state.json', {})
        if (run / 'progress.json').exists() and previous.get('attempts', 0) and not worker_stage.live(previous.get('process')):
            preserved = run.with_name(run.name + '.interrupted_' + utc().replace(':', '').replace('+', '_'))
            run.rename(preserved)
            history = read_json(EVIDENCE / 'restarts.json', [])
            history.append(dict(time=utc(), preserved_run=str(preserved), reason='No complete recovery checkpoint; bounded restart from the authorized 990 initializer.'))
            atomic_json(EVIDENCE / 'restarts.json', history)
        command += ['--init', AUTH['initial_checkpoint']]
    return command


def verify_checkpoint():
    from safetensors import safe_open
    run = Path(AUTH['run_dir'])
    status = read_json(run / 'status.json')
    args = read_json(run / 'train_args.json')
    data = read_json(run / 'data_summary.json')
    assert status['status'] == 'complete' and status['steps'] == data['train_windows'] // 128 == 857
    assert status['samples_seen'] == status['steps'] * 128
    assert args['mode'] == 'full' and args['resolved_max_steps'] == status['steps']
    assert args['init'] == AUTH['initial_checkpoint'] and args['train_root'] == [AUTH['expert_root']]
    checkpoint = run / 'checkpoints' / ('step_%06d' % status['steps'])
    completion = read_json(checkpoint / 'complete.json')
    assert completion['status'] == 'complete'
    for name, digest in completion['files'].items():
        assert sha256(checkpoint / name) == digest, 'Recovery checksum changed: ' + name
    assert (checkpoint / 'trainer_state.pt').is_file() and (checkpoint / 'master_weights.safetensors').is_file()
    with safe_open(str(Path(AUTH['initial_checkpoint']) / 'model.safetensors'), framework='pt', device='cpu') as ref, safe_open(str(checkpoint / 'model.safetensors'), framework='pt', device='cpu') as actual:
        assert set(ref.keys()) == set(actual.keys())
        for key in ref.keys():
            assert ref.get_slice(key).get_shape() == actual.get_slice(key).get_shape()
    proof = dict(status='complete', time=utc(), trial_id='trial_1', experiment_id=AUTH['experiment_id'],
        run_dir=str(run), checkpoint=str(checkpoint), checkpoint_sha256={'model.safetensors':completion['files']['model.safetensors']},
        steps=status['steps'], samples_seen=status['samples_seen'], initialized_from=AUTH['initial_checkpoint'],
        continuation_identity_sha256=sha256(run / 'continuation_identity.json'))
    atomic_json(EVIDENCE / 'checkpoint_verified.json', proof)
    atomic_json(HERE.parent / 'checkpoints/continuation_from_990_final.json', proof)
    link = HERE.parent / 'checkpoints/continuation_from_990_final'
    if not link.exists():
        link.symlink_to(checkpoint, target_is_directory=True)
    assert link.resolve() == checkpoint


def evaluate(split):
    proof = read_json(EVIDENCE / 'checkpoint_verified.json')
    root = Path(AUTH['evaluation_root'])
    command = [SIM, '-B', '-u', str(root / 'source_v1/evaluate.py'), '--dataset', 'r2r', '--split', split,
        '--name', 'phase1/' + split, '--run-dir', proof['run_dir'], '--weights', proof['checkpoint'], '--workers', '6']
    env = dict(ROLEVLN_QUEUE_OUTPUT=str(root), ROLEVLN_QUEUE_CODE_ROOT=str(root / 'source_v1'),
        ROLEVLN_NATIVE_CODE_ROOT=str(SOURCE), ROLEVLN_QWEN_PYTHON=POL)
    stage(split, command, lambda: read_json(root / 'phase1' / split / 'complete.json', {}).get('status') == 'complete',
        root / 'phase1' / split / 'progress.json', gpu=True, extra_env=env)
    atomic_json(EVIDENCE / (split + '_verified.json'), dict(status='complete', time=utc(), checkpoint=proof, **verify_split(root, split, proof)))


def report():
    checkpoint = read_json(EVIDENCE / 'checkpoint_verified.json')
    root = Path(AUTH['evaluation_root'])
    baseline = WORKSPACE / 'dagger_r1_full_eval_20261031'
    old = read_json(baseline / 'completion.json')
    results = {split:dict(continuation=verify_split(root, split, checkpoint)['report'],
        original_990=verify_split(baseline, split, old)['report']) for split in SPLITS}
    run = Path(AUTH['run_dir'])
    rows = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()]
    training = dict(status=read_json(run / 'status.json'), data=read_json(run / 'data_summary.json'),
        peak_memory_gib=max(row.get('max_mem_gib', 0) for row in rows),
        attempts=read_json(HERE / 'stages/continuation_train/state.json'), progress=read_json(run / 'progress.json'))
    result = dict(status='complete', time=utc(), trial_id='trial_1', experiment_id=AUTH['experiment_id'],
        description='Additional full-split expert SFT from the completed stage-two checkpoint 990.',
        training=training, evaluations=results, checkpoint=checkpoint, new_dagger_collection=False,
        preserved_initial_checkpoint=AUTH['initial_checkpoint'], data_identity=read_json(EVIDENCE / 'data_identity.json'),
        val_unseen_used_for_selection=False, trial_2_training_remains_held=True)
    atomic_json(EVIDENCE / 'comparison.json', result)
    lines = ['# Trial 1: continuation from 990 results', '', result['description'], '',
        '| Split | Episodes | Original 990 SR | Continued SR | Original 990 SPL | Continued SPL |',
        '|---|---:|---:|---:|---:|---:|']
    for split, value in results.items():
        before, after = value['original_990'], value['continuation']
        lines.append(f"| {split} | {after['episodes']} | {before['metrics']['success']:.6f} | {after['metrics']['success']:.6f} | {before['metrics']['spl']:.6f} | {after['metrics']['spl']:.6f} |")
    lines += ['', 'Final checkpoint: ' + checkpoint['checkpoint'], 'SHA256: ' + checkpoint['checkpoint_sha256']['model.safetensors'],
        '', 'Original checkpoints 623/990 and their results remain unchanged. Details: evidence/comparison.json.']
    atomic_text(HERE / 'RESULTS.md', '\n'.join(lines) + '\n')
    trial = read_json(HERE.parent / 'trial.json')
    trial.update(status='complete_with_continuation', updated_utc=utc(), next_scheduled_work='none within this completed continuation')
    trial['continuation_experiment'].update(status='complete', checkpoint=checkpoint, results=str(HERE / 'RESULTS.md'))
    atomic_json(HERE.parent / 'trial.json', trial)
    atomic_json(HERE / 'completion.json', dict(status='complete', time=utc(), checkpoint=checkpoint, result=str(HERE / 'RESULTS.md')))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or AUTH['status'] != 'authorized':
        raise RuntimeError('Explicit continuation authorization on GPU 0 required.')
    lock = (HERE / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    holds = read_json(QUEUE / 'expanded_stage2/USER_JOB_HOLDS.json', {}).get('jobs', {})
    if holds.get('trial1_continue_990_pipeline', {}).get('status') == 'held':
        raise RuntimeError('User hold prevents continuation execution.')
    source_sha = verify_source()
    checks = read_json(EVIDENCE / 'implementation_checks.json', {})
    if checks.get('status') != 'complete' or checks['source_snapshot_sha256'] != source_sha:
        raise RuntimeError('Missing or stale continuation implementation checks.')
    run = Path(AUTH['run_dir'])
    stage('continuation_train', training_command(), lambda: read_json(run / 'status.json', {}).get('status') == 'complete', run / 'progress.json', gpu=True)
    stage('checkpoint_verify', [POL, '-B', '-u', str(HERE / 'pipeline.py'), 'verify'],
        lambda: read_json(EVIDENCE / 'checkpoint_verified.json', {}).get('status') == 'complete')
    for split in SPLITS:
        evaluate(split)
    stage('report', [POL, '-B', '-u', str(HERE / 'pipeline.py'), 'report'],
        lambda: read_json(HERE / 'completion.json', {}).get('status') == 'complete')


if __name__ == '__main__':
    if sys.argv[1:] == ['verify']:
        verify_checkpoint()
    elif sys.argv[1:] == ['report']:
        report()
    else:
        main()
