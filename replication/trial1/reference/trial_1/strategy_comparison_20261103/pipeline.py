"""Sequential E/G/J experiments after the preserved, already-running F pipeline."""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
from context import *
from common import process_identity, capture_process
from evaluation_checks import verify_split, SPLITS
import worker_stage

STAGES = [branch + '_' + name for branch in ('E','G') for name in ('train','verify','val_seen','val_unseen')]
STAGES += ['J_collect','J_audit','J_train','J_verify','J_val_seen','J_val_unseen','comparison']


def progress(stage, child=None, log=None, progress_path=None):
    done = read_json(HERE/'completion.json', {}).get('status') == 'complete'
    detail = read_json(progress_path, {}) if progress_path else {}
    counts = {k:detail[k] for k in ('step','max_steps','samples_seen','completed','completed_episodes','expected') if k in detail}
    size = Path(log).stat().st_size if log and Path(log).exists() else 0
    atomic_json(HERE/'progress.json', dict(status='complete' if done else 'running', stage=stage,
        time=utc(), process=child, counts=counts, activity=[stage,counts,size],
        next_stage=None if stage == 'comparison' else STAGES[STAGES.index(stage)+1]))
    lines = ['# Trial 1: four strategies E / F / G / J', '', 'Updated: ' + utc(), '',
        'Current stage: ' + ('complete' if done else stage) + '; progress: ' + json.dumps(counts),
        'F is reused once. E and G replay original frozen-623 corrections; J collects once with frozen F.',
        'Every added training pass uses full fine-tuning for one epoch. Budgets differ with mixture size.',
        'GPU 0 only; original checkpoints/results, the stopped fresh-start hold, Trial 2 holds and MMC4 workers are preserved.', '',
        '| Stage | State | Evidence/log |', '|---|---|---|']
    for name in STAGES:
        row = read_json(HERE/'stages'/name/'state.json', {})
        lines.append('| '+name+' | '+row.get('status','pending')+' | '+row.get('log','')+' |')
    atomic_text(HERE/'STATUS.md', '\n'.join(lines)+'\n')
    trial = read_json(HERE.parent/'trial.json')
    suite = trial['strategy_comparison']
    if suite.get('current_stage') != stage or done:
        suite.update(current_stage=stage,status='complete' if done else 'running',live_status=str(HERE/'STATUS.md'))
        trial.update(status='complete_with_strategy_comparison' if done else 'strategy_comparison_running',updated_utc=utc(),
            next_scheduled_work='No further suite work.' if done else ('After '+stage+': '+str(None if stage=='comparison' else STAGES[STAGES.index(stage)+1])))
        atomic_json(HERE.parent/'trial.json',trial)


for key in ('HERE','SOURCE','read_json','atomic_json','utc','process_identity','capture_process','progress'):
    setattr(worker_stage,key,globals()[key])


def stage(*args, **kwargs):
    check_authorization()
    return worker_stage.stage(*args, **kwargs)


def training_command(branch):
    item = spec(branch); identity = binding(branch, create=True); run = Path(item['run_dir'])
    required = math.ceil(identity['updates']/156)*64*2**30 + 256*2**30
    if shutil.disk_usage(WORKSPACE/'data').free < required:
        raise RuntimeError('Insufficient storage for retained recovery checkpoints; no data or checkpoints deleted.')
    command = [POL,'-B','-u',str(HERE/'train_entry.py'),'--recipe','streamvln_stage2','--run-dir',str(run),
        '--train-root',*identity['train_roots'],'--val-root',AUTH['validation_root'],
        '--keep-trainer-states','10000','--keep-final-state']
    checkpoints = sorted((run/'checkpoints').glob('step_*/complete.json'))
    if checkpoints:
        command += ['--resume',str(checkpoints[-1].parent)]
    else:
        previous = read_json(HERE/'stages'/(branch+'_train')/'state.json',{})
        if (run/'progress.json').exists() and previous.get('attempts',0) and not worker_stage.live(previous.get('process')):
            preserved = run.with_name(run.name+'.interrupted_'+utc().replace(':','').replace('+','_'))
            run.rename(preserved)
            history = read_json(HERE/'evidence/restarts.json',[])
            history.append(dict(time=utc(),branch=branch,preserved_run=str(preserved),reason='No complete recovery checkpoint; bounded restart from the same authorized initializer.'))
            atomic_json(HERE/'evidence/restarts.json',history)
        command += ['--init',identity['initial_checkpoint']]
    return command


def verify_checkpoint(branch):
    from safetensors import safe_open
    identity = binding(branch); item = spec(branch); run = Path(item['run_dir'])
    status = read_json(run/'status.json'); args = read_json(run/'train_args.json'); data = read_json(run/'data_summary.json')
    assert status['status'] == 'complete' and status['steps'] == identity['updates']
    assert data['train_windows'] == identity['windows'] and status['samples_seen'] == identity['samples']
    assert args['mode'] == 'full' and args['init'] == identity['initial_checkpoint'] and args['train_root'] == identity['train_roots']
    ck = run/'checkpoints'/('step_%06d' % status['steps']); completion = read_json(ck/'complete.json')
    assert completion['status'] == 'complete' and completion['training_identity_sha256'] == sha256(run/'training_identity.json')
    for name,digest in completion['files'].items():
        if sha256(ck/name) != digest:raise RuntimeError('Checkpoint checksum changed: '+name)
    assert {'model.safetensors','master_weights.safetensors','trainer_state.pt'} <= set(completion['files'])
    with safe_open(str(Path(identity['initial_checkpoint'])/'model.safetensors'),framework='pt',device='cpu') as before, safe_open(str(ck/'model.safetensors'),framework='pt',device='cpu') as after:
        assert set(before.keys()) == set(after.keys())
        for key in before.keys():assert before.get_slice(key).get_shape() == after.get_slice(key).get_shape()
    proof = dict(status='complete',time=utc(),trial_id='trial_1',strategy=branch,run_dir=str(run),checkpoint=str(ck),
        checkpoint_sha256={'model.safetensors':completion['files']['model.safetensors']},steps=status['steps'],
        samples_seen=status['samples_seen'],initialization=identity['initial_checkpoint'],
        training_identity_sha256=sha256(run/'training_identity.json'))
    atomic_json(HERE/'checkpoints'/(branch+'.json'),proof)
    link = HERE/'checkpoints'/branch
    if not link.exists():link.symlink_to(ck,target_is_directory=True)
    assert link.resolve() == ck


def collect_j():
    policy = f_checkpoint(); root = Path(AUTH['fresh_dagger_root'])
    if read_json(root/'summary.json',{}).get('status') != 'complete' and shutil.disk_usage(WORKSPACE/'data').free < 512*2**30:
        raise RuntimeError('Insufficient collection storage reserve; no files deleted.')
    assert read_json(HERE/'collection/plan.json')['status'] == 'complete'
    command = [SIM,'-B','-u',str(SOURCE/'tools/collect_stream_dagger.py'),'--run-dir',policy['run_dir'],
        '--weights',policy['checkpoint'],'--episodes','0','--workers','6','--out',str(root)]
    stage('J_collect',command,lambda:read_json(root/'summary.json',{}).get('status')=='complete',root/'progress.json',
        gpu=True,extra_env={'ROLEVLN_QWEN_PYTHON':POL})
    stage('J_audit',[POL,'-B','-u',str(HERE/'data_audit.py'),'J'],
        lambda:read_json(HERE/'audits/fresh_dagger.json',{}).get('status')=='complete')


def evaluate(branch, split):
    proof = read_json(HERE/'checkpoints'/(branch+'.json')); root = Path(spec(branch)['evaluation_root'])
    command = [SIM,'-B','-u',str(root/'source_v1/evaluate.py'),'--dataset','r2r','--split',split,
        '--name','phase1/'+split,'--run-dir',proof['run_dir'],'--weights',proof['checkpoint'],'--workers','6']
    env = dict(ROLEVLN_QUEUE_OUTPUT=str(root),ROLEVLN_QUEUE_CODE_ROOT=str(root/'source_v1'),ROLEVLN_NATIVE_CODE_ROOT=str(SOURCE),ROLEVLN_QWEN_PYTHON=POL)
    stage(branch+'_'+split,command,lambda:read_json(root/'phase1'/split/'complete.json',{}).get('status')=='complete',
        root/'phase1'/split/'progress.json',gpu=True,extra_env=env)
    verified = verify_split(root,split,proof)
    atomic_json(HERE/'evidence'/(branch+'_'+split+'.json'),dict(status='complete',time=utc(),checkpoint=proof,**verified))


def comparison():
    results = {}; receipts = {'F':f_checkpoint()}
    receipts.update({branch:read_json(HERE/'checkpoints'/(branch+'.json')) for branch in BRANCHES})
    roots = {'F':Path(AUTH['existing_F_evaluation'])}
    roots.update({branch:Path(spec(branch)['evaluation_root']) for branch in BRANCHES})
    for branch in ('E','F','G','J'):
        checkpoint = receipts[branch]; run = Path(checkpoint['run_dir'])
        rows = [json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
        attempt_path = (Path(AUTH['existing_F_preparation'])/'stages/continuation_train/state.json') if branch=='F' else HERE/'stages'/(branch+'_train')/'state.json'
        results[branch] = dict(checkpoint=checkpoint,training_status=read_json(run/'status.json'),
            data=read_json(run/'data_summary.json'),args=read_json(run/'train_args.json'),
            progress=read_json(run/'progress.json'),attempt_history=read_json(attempt_path),
            peak_memory_gib=max(row.get('max_mem_gib',0) for row in rows),
            evaluation={split:verify_split(roots[branch],split,checkpoint)['report'] for split in SPLITS})
    base_root = WORKSPACE/'dagger_r1_full_eval_20261031'; baseline = read_json(base_root/'completion.json')
    result = dict(status='complete',time=utc(),strategies=results,
        original_990={split:verify_split(base_root,split,baseline)['report'] for split in SPLITS},
        fresh_collection=read_json(Path(AUTH['fresh_dagger_root'])/'summary.json'),
        fairness=AUTH['fairness'],unseen_used_for_selection=False,full_RxR_evaluation_in_scope=False,
        old_checkpoints_623_990_preserved=True,trial_2_training_remains_held=True)
    atomic_json(HERE/'evidence/comparison.json',result)
    lines = ['# Four-strategy Trial 1 comparison','',AUTH['fairness'],'',
        '| Strategy | Updates in final pass | R2R seen SR | Seen SPL | Unseen SR | Unseen SPL |','|---|---:|---:|---:|---:|---:|']
    for branch in ('E','F','G','J'):
        row=results[branch];seen=row['evaluation']['val_seen']['metrics'];unseen=row['evaluation']['val_unseen']['metrics']
        lines.append(f"| {branch} | {row['training_status']['steps']} | {seen['success']:.6f} | {seen['spl']:.6f} | {unseen['success']:.6f} | {unseen['spl']:.6f} |")
    lines += ['', 'All checkpoints are fixed final-epoch outputs. No val-unseen selection, early stopping, hyperparameter tuning or automatic promotion was performed.',
        'One seed per strategy; these results do not estimate between-seed variance. Full RxR evaluation is outside this comparison.',
        'Counts, durations, collection provenance, restart history, memory and immutable checkpoint identities: evidence/comparison.json.']
    atomic_text(HERE/'RESULTS.md','\n'.join(lines)+'\n')
    atomic_json(HERE/'completion.json',dict(status='complete',time=utc(),report=str(HERE/'RESULTS.md'),checkpoint_receipts=receipts))
    trial=read_json(HERE.parent/'trial.json');trial['strategy_comparison'].update(status='complete',results=str(HERE/'RESULTS.md'))
    trial.update(status='complete_with_strategy_comparison',updated_utc=utc());atomic_json(HERE.parent/'trial.json',trial)


def main():
    check_authorization()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    lock=(HERE/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check=read_json(HERE/'evidence/implementation_checks.json',{})
    assert check.get('status')=='complete' and check['source_snapshot_sha256']==verify_source()
    f_checkpoint()
    assert read_json(HERE/'audits/existing_mixture.json',{}).get('passed')
    for branch in BRANCHES:
        check_authorization()
        if branch=='J':collect_j()
        run=Path(spec(branch)['run_dir'])
        ready=lambda:read_json(run/'status.json',{}).get('status')=='complete'
        command=training_command(branch) if not ready() else [POL,'-c','pass']
        stage(branch+'_train',command,ready,run/'progress.json',gpu=True,extra_env={'ROLEVLN_STRATEGY_BRANCH':branch})
        stage(branch+'_verify',[POL,'-B','-u',str(HERE/'pipeline.py'),'verify',branch],
            lambda:read_json(HERE/'checkpoints'/(branch+'.json'),{}).get('status')=='complete')
        for split in SPLITS:evaluate(branch,split)
    stage('comparison',[POL,'-B','-u',str(HERE/'pipeline.py'),'comparison'],
        lambda:read_json(HERE/'completion.json',{}).get('status')=='complete')


if __name__ == '__main__':
    if sys.argv[1:2]==['verify']:verify_checkpoint(sys.argv[2])
    elif sys.argv[1:]==['comparison']:comparison()
    else:main()
