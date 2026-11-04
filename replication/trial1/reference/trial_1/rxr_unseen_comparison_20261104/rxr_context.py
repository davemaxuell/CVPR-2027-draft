"""Immutable evaluation protocol, model bindings and strict result verification."""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
WORKSPACE = HERE.parents[4]
QUEUE = WORKSPACE / 'queue_after_stageA'
SUITE = TRIAL / 'strategy_comparison_20261103'
F_PREP = TRIAL / 'continuation_from_990_20261103'
SOURCE = F_PREP / 'source_v1'
EVAL_CODE = HERE / 'evaluation/source_v1'
POL = '/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python'
SIM = '/home/bufsgpu/yes/envs/rolevln/bin/python'
MODELS = ('F', '990', '623', 'E', 'G', 'J')
sys.path.insert(0, str(QUEUE))
from common import atomic_json, atomic_text, read_json, sha256, utc


def verify_source():
    manifest = read_json(HERE / 'SOURCE_SNAPSHOT.json')
    for name, expected in manifest['files'].items():
        if sha256(HERE / name) != expected:
            raise RuntimeError('RxR extension source identity changed: ' + name)
    for name, expected in manifest['frozen_source_manifests'].items():
        if sha256(name) != expected:
            raise RuntimeError('Original frozen manifest changed: ' + name)
    return sha256(HERE / 'SOURCE_SNAPSHOT.json')


def authorize():
    auth = read_json(HERE / 'authorization.json')
    if auth.get('status') != 'authorized' or auth['gpu'] != 0 or auth['models'] != list(MODELS):
        raise RuntimeError('Scoped RxR comparison authorization is required.')
    holds = read_json(QUEUE / 'expanded_stage2/USER_JOB_HOLDS.json')['jobs']
    if holds.get('trial1_strategy_suite', {}).get('status') == 'held':
        raise RuntimeError('Trial 1 strategy suite is held.')


def model_spec(model):
    if model not in MODELS:
        raise ValueError(model)
    return read_json(HERE / 'models.json')[model]


def resolve_model(model):
    spec = model_spec(model)
    if model in ('623', '990'):
        proof = dict(status='complete', run_dir=spec['run_dir'], checkpoint=spec['checkpoint'],
                     checkpoint_sha256=spec['checkpoint_sha256'])
    else:
        proof = read_json(spec['checkpoint_receipt'], {})
        if proof.get('status') != 'complete' or proof.get('run_dir') != spec['run_dir']:
            raise RuntimeError('Required verified checkpoint is unavailable: ' + model)
    run = Path(proof['run_dir'])
    status = read_json(run / 'status.json', {})
    if status.get('status') != 'complete':
        raise RuntimeError('Model training is incomplete: ' + model)
    if Path(proof['checkpoint']) != run / 'checkpoints' / ('step_%06d' % status['steps']):
        raise RuntimeError('Evaluation requires the fixed final checkpoint: ' + model)
    config = read_json(run / 'stream_config.json')
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if digest != read_json(HERE / 'protocol.json')['native_stream_config_sha256']:
        raise RuntimeError('Native policy input/action configuration differs: ' + model)
    return proof


def bind_model(model):
    proof = resolve_model(model)
    root = Path(model_spec(model)['output_root'])
    expected = dict(model=model, trial_id='trial_1', run_dir=proof['run_dir'], checkpoint=proof['checkpoint'],
        checkpoint_sha256=proof['checkpoint_sha256'], protocol_sha256=sha256(HERE / 'protocol.json'),
        split_audit_sha256=sha256(HERE / 'split_audit.json'), source_snapshot_sha256=verify_source())
    for name, digest in proof['checkpoint_sha256'].items():
        if sha256(Path(proof['checkpoint']) / name) != digest:
            raise RuntimeError('Checkpoint checksum changed: ' + model)
    previous = read_json(root / 'binding.json')
    if previous and previous != expected:
        raise RuntimeError('Evaluation binding changed: ' + model)
    if not previous:
        atomic_json(root / 'binding.json', expected)
    return expected


def verify_result(model, partial=False):
    root = Path(model_spec(model)['output_root'])
    folder = root / 'phase1/val_unseen'
    binding = read_json(root / 'binding.json', {})
    proof = resolve_model(model)
    audit = read_json(HERE / 'split_audit.json')
    cfg = read_json(folder / 'config.json', {})
    report = read_json(folder / 'complete.json', {})
    if binding.get('checkpoint_sha256') != proof['checkpoint_sha256']:
        raise RuntimeError('Missing or different model binding: ' + model)
    for key, value in dict(dataset='rxr', split='val_unseen', run_dir=proof['run_dir'], weights=proof['checkpoint'],
        checkpoint_sha256=proof['checkpoint_sha256'], selected_episodes=audit['selected_episode_ids'],
        protocol_sha256=sha256(HERE / 'protocol.json'), split_audit_sha256=sha256(HERE / 'split_audit.json'),
        memory_mode='empty', force_slot=False, capture_features=False, save_frames=False).items():
        if cfg.get(key) != value:
            raise RuntimeError('Evaluation configuration mismatch: ' + model + '/' + key)
    if cfg['evaluation_sha256'] != sha256(EVAL_CODE / 'evaluate.py'):
        raise RuntimeError('Evaluation code differs.')
    sim = cfg['simulator']['habitat']
    if sim['simulator']['turn_angle'] != 15 or sim['environment']['max_episode_steps'] != 700:
        raise RuntimeError('RxR action protocol differs.')
    files = {p.stem:p for p in (folder / 'episodes').glob('*.json')}
    expected = set(audit['selected_episode_ids'])
    if not set(files) <= expected or (not partial and set(files) != expected):
        raise RuntimeError('Incomplete, unexpected or duplicate RxR episode set: ' + model)
    totals, hashes = {}, {}
    for key, path in files.items():
        row = read_json(path); reference = audit['episodes'][key]
        if row['episode_id'] != key or row['dataset'] != 'rxr' or row['split'] != 'val_unseen':
            raise RuntimeError('Episode namespace mismatch.')
        if not row['scene_id'].endswith('/' + reference['scene_id']):
            raise RuntimeError('Episode scene mismatch.')
        for name in ('language', 'role', 'instruction_id'):
            if str(row[name]) != reference[name]:
                raise RuntimeError('Episode instruction metadata mismatch: ' + name)
        if hashlib.sha256(row['instruction'].encode()).hexdigest() != reference['instruction_sha256']:
            raise RuntimeError('Episode instruction mismatch.')
        actions = row['executed_actions']; steps = row['action_steps']
        if not actions or len(actions) != len(steps) or len(actions) != row['metrics']['steps_taken']:
            raise RuntimeError('Action accounting mismatch.')
        if set(actions) - {0,1,2,3} or len(actions) > 700:
            raise RuntimeError('Invalid action space or episode length.')
        if row['termination'] != ('STOP' if actions[-1] == 0 else 'step_limit'):
            raise RuntimeError('STOP accounting mismatch.')
        for name, value in row['metrics'].items():
            if not math.isfinite(value):
                raise RuntimeError('Nonfinite measured metric.')
            totals[name] = totals.get(name, 0.) + value
        hashes[key] = sha256(path)
    identities = []
    for path in (folder / 'workers').glob('*.identity.json'):
        row = read_json(path)
        if row['weights'] != proof['checkpoint'] or any(row['weight_sha256'].get(k) != v for k,v in proof['checkpoint_sha256'].items()):
            raise RuntimeError('Worker policy differs.')
        if row.get('saved_places_slot') or row.get('stop_logit_bias') != 0:
            raise RuntimeError('Unexpected memory slot or STOP bias.')
        identities.append({k:v for k,v in row.items() if k not in ('device','seconds','nbytes')})
    if not identities or any(row != identities[0] for row in identities):
        raise RuntimeError('Missing or inconsistent policy identity.')
    if not partial:
        if report.get('status') != 'complete' or report.get('episodes') != 3669 or report.get('scenes') != 11:
            raise RuntimeError('Full RxR result is not complete.')
        if report.get('dataset') != 'rxr' or report.get('model_role') != 'trial1_' + model:
            raise RuntimeError('Result model or dataset label differs.')
        for name, total in totals.items():
            if not math.isclose(total / 3669, report['metrics'][name], rel_tol=1e-10, abs_tol=1e-10):
                raise RuntimeError('Aggregate metric mismatch: ' + name)
    return dict(status='partial_verified' if partial else 'complete', time=utc(), model=model,
        checkpoint=proof, report=report, verified_episodes=len(files), episode_sha256=hashes,
        config_sha256=sha256(folder / 'config.json'), binding_sha256=sha256(root / 'binding.json'))
