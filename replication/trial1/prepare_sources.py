#!/usr/bin/env python3
"""Prepare new 2B working copies with local bindings; no model load or job launch.

The default is a dry run. --apply writes only a new destination directory.
Historical authorizations, run statuses and checkpoint receipts are never copied.
"""
import argparse
import ast
import hashlib
import json
import re
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path

from verify_package import verify

PACKAGE = Path(__file__).resolve().parent
OLD_DATA = '/data/team_a/dave-workspace/RoleVLN/data'
OLD_POL = '/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python'
OLD_SIM = '/home/bufsgpu/yes/envs/rolevln/bin/python'
OLD_CACHE = '/data/shared/hf_cache'


def absolute_path(value):
    value = str(Path(value).expanduser().resolve())
    if any(c in value for c in "\n\r\t'\"\\"):
        raise argparse.ArgumentTypeError('Paths with quotes, backslashes or control characters are unsupported; spaces and brackets are supported.')
    return Path(value)


def prepare(args):
    verified = verify(PACKAGE)
    dest = args.destination
    if dest.exists():
        raise ValueError('Destination must be new; refusing to overwrite: ' + str(dest))
    if dest == PACKAGE or PACKAGE in dest.parents:
        raise ValueError('Create the working copy outside the immutable package.')
    if not re.fullmatch('[0-9a-fA-F]{40}', args.revision):
        raise ValueError('--revision must be a full immutable 40-character model commit SHA.')
    if not args.data_root.is_dir():
        raise ValueError('--data-root must be an existing directory holding datasets/ and scene_datasets/.')
    for p in [args.policy_python, args.sim_python]:
        if not p.is_file():
            raise ValueError('Python interpreter does not exist: ' + str(p))
    plan = {'model': 'Qwen/Qwen3.5-2B', 'revision': args.revision.lower(),
            'destination': str(dest), 'data_root': str(args.data_root),
            'policy_python': str(args.policy_python), 'sim_python': str(args.sim_python),
            'hf_home': str(args.hf_home), 'gpu': 0, 'applied': args.apply,
            'source_verification': verified,
            'next': 'Acquire the pinned model, instantiate new stage guards/receipts, and qualify before training.'}
    if not args.apply:
        return plan
    dest.mkdir(parents=True, exist_ok=False)
    for name in ['source', 'evaluation']:
        shutil.copytree(PACKAGE / name, dest / name)
    changes = {}
    swaps = [(OLD_DATA, str(args.data_root)), (OLD_POL, str(args.policy_python)),
             (OLD_SIM, str(args.sim_python)), (OLD_CACHE, str(args.hf_home))]
    for p in sorted(dest.rglob('*')):
        if not p.is_file() or p.suffix not in {'.py', '.json', '.yaml', '.yml', '.txt', '.md'}:
            continue
        before = p.read_text()
        after = before
        for old, new in swaps:
            after = after.replace(old, new)
        if p.name == 'config.py' and p.parent.name == 'stream_policy':
            fields = {
                'backbone: str = "qwen35_4b"': 'backbone: str = "qwen35_2b"',
                'repo: str = "Qwen/Qwen3.5-4B"': 'repo: str = "Qwen/Qwen3.5-2B"',
                'revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"': 'revision: str = ' + repr(args.revision.lower()),
            }
            for old, new in fields.items():
                if after.count(old) != 1:
                    raise ValueError('Unexpected model binding in ' + str(p))
                after = after.replace(old, new)
        if p.parent.parent.name == 'evaluation' and p.name == 'common.py':
            old = "str(WORKSPACE / '[OURS]' / 'baseline-v0')"
            if after.count(old) != 1:
                raise ValueError('Unexpected evaluator native-code default: ' + str(p))
            after = after.replace(old, repr(str(dest / 'source/full')))
            after = after.replace("DATA = WORKSPACE / 'data'", 'DATA = Path(' + repr(str(args.data_root)) + ')')
        if p.suffix == '.py':
            ast.parse(after, filename=str(p))
        elif p.suffix == '.json':
            json.loads(after)
        if before != after:
            p.write_text(after)
            changes[p.relative_to(dest).as_posix()] = {
                'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                'after_sha256': hashlib.sha256(after.encode()).hexdigest()}
    processed = args.data_root / 'processed/baseline_v0'
    variables = {
        'WORK': str(dest), 'PACKAGE': str(PACKAGE),
        'CODE_A': str(dest / 'source/group_a'), 'CODE_FULL': str(dest / 'source/full'),
        'EVAL_R2R': str(dest / 'evaluation/r2r'), 'EVAL_RXR': str(dest / 'evaluation/rxr'),
        'EVAL_VENDOR': str(dest / 'evaluation/vendor/vlnce'),
        'POL': str(args.policy_python), 'SIM': str(args.sim_python),
        'ROLEVLN_QWEN_PYTHON': str(args.policy_python),
        'DATA_ROOT': str(args.data_root), 'HF_HOME': str(args.hf_home),
        'HF_HUB_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '0',
        'EXPERT_A': str(processed / 'expert_groupA'), 'VAL_C': str(processed / 'expert_groupC_600'),
        'EXPERT_FULL': str(processed / 'trial_1/full_train_english_guide/expert'),
        'VAL_SEEN': str(processed / 'trial_1/full_train_english_guide/val_seen_diagnostics_600'),
        'D1': str(processed / 'trial_1_qwen35_2b/dagger_groupA'),
        'S1': str(dest / 'runs/S1'), 'S2': str(dest / 'runs/S2'), 'F': str(dest / 'runs/F'),
        'RECORDS': str(dest / 'records'), 'MODEL_REVISION': args.revision.lower(),
    }
    (dest / 'environment.sh').write_text('# Generated bindings; source only after reviewing paths. No job is launched.\n' +
        ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in variables.items()))
    for name in ['records', 'runs']:
        (dest / name).mkdir()
    plan.update(prepared_utc=datetime.now(timezone.utc).isoformat(), changes=changes,
                historical_run_state_copied=False, jobs_launched=False)
    (dest / 'preparation.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination', type=absolute_path, required=True)
    ap.add_argument('--data-root', type=absolute_path, required=True)
    ap.add_argument('--policy-python', type=absolute_path, required=True)
    ap.add_argument('--sim-python', type=absolute_path, required=True)
    ap.add_argument('--hf-home', type=absolute_path, required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--apply', action='store_true')
    print(json.dumps(prepare(ap.parse_args()), indent=2))
