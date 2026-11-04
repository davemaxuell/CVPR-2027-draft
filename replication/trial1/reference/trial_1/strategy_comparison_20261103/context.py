"""Fixed experimental assignments and immutable input bindings for E/G/J."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
QUEUE = WORKSPACE / 'queue_after_stageA'
sys.path.insert(0, str(QUEUE))
from common import read_json, atomic_json, atomic_text, sha256, utc

AUTH = read_json(HERE / 'authorization.json')
SOURCE = Path(AUTH['source_root'])
POL = '/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python'
SIM = '/home/bufsgpu/yes/envs/rolevln/bin/python'
BRANCHES = ('E', 'G', 'J')


def spec(branch):
    if branch not in BRANCHES:
        raise ValueError('Unapproved experimental branch: ' + branch)
    return read_json(HERE / 'branches' / (branch + '.json'))


def verify_source():
    for relative, digest in read_json(HERE / 'SOURCE_SNAPSHOT.json')['files'].items():
        if sha256(HERE / relative) != digest:
            raise RuntimeError('Frozen strategy source identity changed: ' + relative)
    return sha256(HERE / 'SOURCE_SNAPSHOT.json')


def check_authorization():
    if AUTH['status'] != 'authorized' or AUTH['gpu'] != 0 or AUTH['selected'] != ['E','F','G','J']:
        raise RuntimeError('Scoped four-strategy authorization is required.')
    holds = read_json(QUEUE / 'expanded_stage2/USER_JOB_HOLDS.json')['jobs']
    if holds.get('trial1_strategy_suite', {}).get('status') == 'held':
        raise RuntimeError('The four-strategy comparison is held by the user.')


def f_checkpoint():
    root = Path(AUTH['existing_F_preparation'])
    proof = read_json(root / 'evidence/checkpoint_verified.json', {})
    completion = read_json(root / 'completion.json', {})
    if completion.get('status') != 'complete' or proof.get('status') != 'complete':
        raise RuntimeError('F and its evaluations must finish before the additional strategies.')
    assert proof['run_dir'] == AUTH['existing_F_run'] and proof['steps'] == 857
    assert proof['checkpoint'] == str(Path(AUTH['existing_F_run']) / 'checkpoints/step_000857')
    assert proof['initialized_from'] == AUTH['stage_two_checkpoint']
    assert completion['checkpoint'] == proof
    return proof


def binding(branch, create=False):
    item = spec(branch)
    if branch == 'J':
        initial = f_checkpoint()
        checkpoint, digest = initial['checkpoint'], initial['checkpoint_sha256']['model.safetensors']
        audit_path = HERE / 'audits/fresh_dagger.json'
    else:
        key = 'stage_one' if branch == 'E' else 'stage_two'
        checkpoint, digest = AUTH[key + '_checkpoint'], AUTH[key + '_sha256']
        audit_path = HERE / 'audits/existing_mixture.json'
    audit = read_json(audit_path, {})
    if audit.get('status') != 'complete' or not audit.get('passed'):
        raise RuntimeError('Required data audit has not passed: ' + str(audit_path))
    assert audit['train_roots'] == item['train_roots']
    windows = audit['combined_windows']
    if item['expected_windows'] is not None:
        assert windows == item['expected_windows']
    identity = dict(branch=branch, trial_id='trial_1', suite_id=AUTH['suite_id'],
        initial_checkpoint=checkpoint, initial_model_sha256=digest,
        train_roots=item['train_roots'], validation_root=AUTH['validation_root'],
        audit=str(audit_path), audit_sha256=sha256(audit_path),
        expert_identity_sha256=sha256(HERE / 'evidence/data_identity.json'),
        windows=windows, updates=windows // 128, samples=windows // 128 * 128,
        tail_windows=windows % 128, source_snapshot_sha256=verify_source(),
        recipe=AUTH['training'], fresh_optimizer=True,
        correction_source='fresh frozen F' if branch == 'J' else 'original frozen 623')
    target = HERE / 'bindings' / (branch + '.json')
    previous = read_json(target)
    if previous and previous != identity:
        raise RuntimeError('Experimental input identity changed: ' + branch)
    if not previous:
        if not create:
            raise RuntimeError('Input binding must be created before training: ' + branch)
        atomic_json(target, identity)
    return identity


def training_args_check(args, branch, identity):
    item = spec(branch)
    assert Path(args.run_dir).resolve() == Path(item['run_dir'])
    assert args.recipe == 'streamvln_stage2' and args.mode == 'full'
    assert args.train_root == identity['train_roots'] and args.val_root == [AUTH['validation_root']]
    assert args.epochs == 1 and args.max_steps == 0 and args.source_weights is None
    assert args.batch_size * args.grad_accum == 128 and args.seed == 0
    assert args.lr == 2e-5 and args.vision_lr == 5e-6 and args.weight_decay == 0
    assert args.warmup_frac == .03 and args.schedule == 'cosine'
    assert args.augment and not args.no_grad_ckpt
    assert args.keep_final_state and args.keep_trainer_states >= 10000
    assert args.datasets == ['r2r','rxr'] and args.label_source == 'executed'
    if args.resume:
        assert not args.init and Path(args.resume).resolve().parent == Path(item['run_dir']) / 'checkpoints'
    elif args.init != identity['initial_checkpoint']:
        raise RuntimeError('Wrong initializer for strategy ' + branch)
