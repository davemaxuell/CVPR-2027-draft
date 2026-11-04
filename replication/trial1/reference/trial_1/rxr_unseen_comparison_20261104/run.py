"""Existing F/E/G/J suite extended with matched RxR evaluations for six policies."""
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

from rxr_context import *

sys.path.insert(0, str(SUITE))
import pipeline as original

repair_path = TRIAL / 'orchestration_recovery_20261104/launch.py'
repair_spec = importlib.util.spec_from_file_location('verified_child_capture_repair', repair_path)
repair = importlib.util.module_from_spec(repair_spec)
repair_spec.loader.exec_module(repair)
repair.verify_recovery()
repair.install(original)

STAGES = [model + '_rxr_val_unseen' for model in ('F', '990', '623')]
for branch in original.BRANCHES:
    if branch == 'J':
        STAGES += ['J_collect', 'J_audit']
    STAGES += [branch + '_' + suffix for suffix in ('train','verify','val_seen','val_unseen','rxr_val_unseen')]
STAGES += ['comparison']
original.STAGES = STAGES


def progress(stage, child=None, log=None, progress_path=None):
    done = read_json(HERE / 'completion.json', {}).get('status') == 'complete'
    detail = read_json(progress_path, {}) if progress_path else {}
    counts = {key:detail[key] for key in ('step','max_steps','samples_seen','completed','completed_episodes','expected') if key in detail}
    size = Path(log).stat().st_size if log and Path(log).is_file() else 0
    next_stage = None if stage == STAGES[-1] else STAGES[STAGES.index(stage) + 1]
    record = dict(status='complete' if done else 'running', stage=stage, time=utc(), process=child,
                  counts=counts, activity=[stage, counts, size], next_stage=next_stage)
    for directory in (HERE, SUITE):
        atomic_json(directory / 'progress.json', record)
    lines = ['# Trial 1: F / E / G / J and six-model RxR comparison', '', 'Updated: ' + utc(), '',
        'Current stage: ' + stage + '; progress: ' + json.dumps(counts) + '.',
        'Next automatically: ' + str(next_stage) + '.', '',
        'Each policy uses all 3,669 English-guide RxR val-unseen episodes across 11 held-out scenes.',
        'Matched internal protocol: 15-degree turns, 700 steps, STOP bias 0, no Saved places slot. This differs from the official 30-degree RxR protocol.', '',
        '| Policy | RxR unseen stage | Completed episodes |', '|---|---|---:|']
    for model in MODELS:
        root = Path(model_spec(model)['output_root']) / 'phase1/val_unseen'
        state = read_json(SUITE / 'stages' / (model + '_rxr_val_unseen') / 'state.json', {})
        number = read_json(root / 'progress.json', {}).get('completed', 0)
        lines.append(f"| {model} | {state.get('status', 'pending')} | {number}/3669 |")
    lines += ['', 'F training/R2R work is reused. Training order remains E → G → J after F and the RxR anchor evaluations.',
        'GPU 0 only. Original 623/990 and all frozen sources are preserved; Trial 2 training remains held.',
        '', '| Stage | State |', '|---|---|']
    for name in STAGES:
        state = read_json(SUITE / 'stages' / name / 'state.json', {})
        lines.append('| ' + name + ' | ' + state.get('status', 'pending') + ' |')
    for directory in (HERE, SUITE):
        atomic_text(directory / 'STATUS.md', '\n'.join(lines) + '\n')
    trial = read_json(TRIAL / 'trial.json')
    if trial['strategy_comparison'].get('current_stage') != stage or done:
        trial['strategy_comparison'].update(current_stage=stage, status='complete' if done else 'running',
            live_status=str(HERE / 'STATUS.md'), full_rxr_unseen_in_scope=True)
        trial['rxr_unseen_comparison'].update(status='complete' if done else 'running', current_stage=stage)
        trial.update(status='complete_with_strategy_comparison' if done else 'strategy_comparison_running',
                     updated_utc=utc(), next_scheduled_work=next_stage)
        atomic_json(TRIAL / 'trial.json', trial)


original.worker_stage.progress = progress
original.progress = progress


def stage(*args, **kwargs):
    authorize()
    return original.stage(*args, **kwargs)


def evaluate_rxr(model):
    binding = bind_model(model)
    root = Path(model_spec(model)['output_root'])
    command = [SIM, '-B', '-u', str(EVAL_CODE / 'evaluate.py'), '--dataset', 'rxr', '--split', 'val_unseen',
        '--name', 'phase1/val_unseen', '--run-dir', binding['run_dir'], '--weights', binding['checkpoint'],
        '--workers', str(read_json(HERE / 'protocol.json')['workers'])]
    environment = dict(ROLEVLN_QUEUE_OUTPUT=str(root), ROLEVLN_QUEUE_CODE_ROOT=str(EVAL_CODE),
        ROLEVLN_NATIVE_CODE_ROOT=str(SOURCE), ROLEVLN_QWEN_PYTHON=POL,
        ROLEVLN_RXR_AUDIT=str(HERE / 'split_audit.json'), ROLEVLN_RXR_PROTOCOL=str(HERE / 'protocol.json'))
    stage(model + '_rxr_val_unseen', command,
        lambda: read_json(root / 'phase1/val_unseen/complete.json', {}).get('status') == 'complete',
        root / 'phase1/val_unseen/progress.json', gpu=True, extra_env=environment)
    atomic_json(HERE / 'evidence' / (model + '_rxr_verified.json'), verify_result(model))


def comparison():
    authorize()
    rxr = {model:verify_result(model) for model in MODELS}
    configs = [read_json(Path(model_spec(model)['output_root']) / 'phase1/val_unseen/config.json') for model in MODELS]
    for key in ('simulator','selected_episodes','seed','protocol_sha256','split_audit_sha256',
                'evaluation_sha256','instrumentation_sha256','metric_loader_sha256','memory_mode','force_slot'):
        if any(cfg[key] != configs[0][key] for cfg in configs):
            raise RuntimeError('Comparison protocols differ: ' + key)
    results = {}
    for branch in ('E','F','G','J'):
        checkpoint = resolve_model(branch); run = Path(checkpoint['run_dir'])
        root = Path(original.AUTH['existing_F_evaluation']) if branch == 'F' else Path(original.spec(branch)['evaluation_root'])
        rows = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()]
        attempt = F_PREP / 'stages/continuation_train/state.json' if branch == 'F' else SUITE / 'stages' / (branch + '_train') / 'state.json'
        results[branch] = dict(checkpoint=checkpoint, training_status=read_json(run / 'status.json'),
            data=read_json(run / 'data_summary.json'), args=read_json(run / 'train_args.json'),
            progress=read_json(run / 'progress.json'), attempt_history=read_json(attempt),
            peak_memory_gib=max(row.get('max_mem_gib',0) for row in rows),
            evaluation={split:original.verify_split(root,split,checkpoint)['report'] for split in original.SPLITS},
            rxr_unseen=rxr[branch]['report'])
    baseline_root = WORKSPACE / 'dagger_r1_full_eval_20261031'
    baseline = read_json(baseline_root / 'completion.json')
    old = {split:original.verify_split(baseline_root,split,baseline)['report'] for split in original.SPLITS}
    metric_keys = ('success','spl','ndtw','sdtw')
    differences = {model:{key:100*(rxr[model]['report']['metrics'][key]-rxr['990']['report']['metrics'][key])
                          for key in metric_keys} for model in MODELS}
    result = dict(status='complete', time=utc(), trial_id='trial_1', strategies=results,
        original_990=old, rxr_unseen=rxr, rxr_delta_vs_990_percentage_points=differences,
        protocol=read_json(HERE / 'protocol.json'), split_audit=str(HERE / 'split_audit.json'),
        source_snapshot_sha256=verify_source(), fairness=original.AUTH['fairness'],
        fresh_collection=read_json(Path(original.AUTH['fresh_dagger_root']) / 'summary.json'),
        unseen_used_for_selection=False, full_RxR_evaluation_in_scope=True,
        rxr_scope='Complete English-guide val-unseen, matched 15-degree / 700-step internal protocol',
        old_checkpoints_623_990_preserved=True, trial_2_training_remains_held=True)
    atomic_json(HERE / 'evidence/comparison.json', result)
    atomic_json(SUITE / 'evidence/comparison.json', result)
    lines = ['# Trial 1: R2R and RxR comparison', '', original.AUTH['fairness'], '',
        'All policies use the same 3,669 English-guide RxR val-unseen episodes in 11 held-out scenes.',
        'RxR uses the established 15-degree turns and 700-step cap. These are matched internal results, not official 30-degree protocol scores.', '',
        '| Model | RxR unseen SR (%) | SPL (%) | nDTW (%) | sDTW (%) |', '|---|---:|---:|---:|---:|']
    for model in ('623','990','F','E','G','J'):
        metrics = rxr[model]['report']['metrics']
        lines.append('| ' + model + ' | ' + ' | '.join(f'{100*metrics[key]:.2f}' for key in metric_keys) + ' |')
    lines += ['', '| Model | R2R seen SR (%) | Seen SPL (%) | Unseen SR (%) | Unseen SPL (%) |', '|---|---:|---:|---:|---:|']
    for model in ('990','F','E','G','J'):
        measured = old if model == '990' else results[model]['evaluation']
        values = [measured[split]['metrics'][key] for split in ('val_seen','val_unseen') for key in ('success','spl')]
        lines.append('| ' + model + ' | ' + ' | '.join(f'{100*value:.2f}' for value in values) + ' |')
    lines += ['', 'Full R2R scores for 623 were not part of this extension and are not inferred from diagnostics.',
        'Final checkpoints are fixed by training completion; no unseen-based selection, tuning, early stopping or extra rounds.',
        'One seed per strategy; no claim of between-seed significance. Counts, optimizer updates, duration, restarts, memory, hashes, episode audits and protocol details are in evidence/comparison.json.']
    for directory in (HERE, SUITE):
        atomic_text(directory / 'RESULTS.md', '\n'.join(lines) + '\n')
    completion = dict(status='complete', time=utc(), report=str(HERE / 'RESULTS.md'),
        comparison=str(HERE / 'evidence/comparison.json'), checkpoint_receipts={model:resolve_model(model) for model in MODELS},
        full_rxr_unseen_verified=True, rxr_episodes_per_model=3669)
    atomic_json(SUITE / 'completion.json', completion)
    atomic_json(HERE / 'completion.json', completion)
    trial = read_json(TRIAL / 'trial.json')
    trial['strategy_comparison'].update(status='complete', results=str(HERE / 'RESULTS.md'), full_rxr_unseen_in_scope=True)
    trial['rxr_unseen_comparison'].update(status='complete', results=str(HERE / 'RESULTS.md'))
    trial.update(status='complete_with_strategy_comparison', updated_utc=utc(), next_scheduled_work=None)
    atomic_json(TRIAL / 'trial.json', trial)


def main():
    authorize(); original.check_authorization()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('GPU 0 is required.')
    lock = (SUITE / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own_lock = (HERE / 'pipeline.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checks = read_json(HERE / 'checks.json', {})
    if checks.get('status') != 'complete' or checks['source_snapshot_sha256'] != verify_source():
        raise RuntimeError('RxR extension checks are missing or stale.')
    old_checks = read_json(SUITE / 'evidence/implementation_checks.json', {})
    if old_checks.get('status') != 'complete' or old_checks['source_snapshot_sha256'] != original.verify_source():
        raise RuntimeError('Original strategy checks are missing or stale.')
    original.f_checkpoint()
    assert read_json(SUITE / 'audits/existing_mixture.json')['passed']
    for path, asset in read_json(HERE / 'split_audit.json')['assets'].items():
        if sha256(path) != asset['sha256']:
            raise RuntimeError('Audited RxR asset changed: ' + path)
    for model in ('F', '990', '623'):
        evaluate_rxr(model)
    for branch in original.BRANCHES:
        authorize()
        if branch == 'J':
            original.collect_j()
        run = Path(original.spec(branch)['run_dir'])
        ready = lambda: read_json(run / 'status.json', {}).get('status') == 'complete'
        # A valid script placeholder lets the unchanged helper skip a completed
        # training run on resume without issuing an invalid '-c' launch command.
        command = original.training_command(branch) if not ready() else [POL, '-B', '-u', str(SUITE / 'train_entry.py')]
        stage(branch + '_train', command, ready, run / 'progress.json', gpu=True,
              extra_env={'ROLEVLN_STRATEGY_BRANCH':branch})
        stage(branch + '_verify', [POL, '-B', '-u', str(SUITE / 'pipeline.py'), 'verify', branch],
              lambda:read_json(SUITE / 'checkpoints' / (branch + '.json'), {}).get('status') == 'complete')
        for split in original.SPLITS:
            original.evaluate(branch, split)
        evaluate_rxr(branch)
    stage('comparison', [POL, '-B', '-u', str(HERE / 'run.py'), 'report'],
          lambda:read_json(HERE / 'completion.json', {}).get('status') == 'complete')


if __name__ == '__main__':
    if sys.argv[1:] == ['report']:
        verify_source(); comparison()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: run.py [report]')
