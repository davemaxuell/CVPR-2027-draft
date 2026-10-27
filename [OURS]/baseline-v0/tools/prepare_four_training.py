"""Seal the predeclared scorer recipe after new-input and exact-resume checks."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scorer_four.common import (BASE_RUN, CACHE as EXTRA, FEATURES_RUN, POLICY_ID, TRIALS,
    read, sha, verify_features_protocol, write)
from scorer_four.training_common import CACHE, RUN, flags


def main():
    verify_features_protocol(); manifest = read(EXTRA / 'manifest.json'); assert manifest['status'] == 'complete'
    checks = read(RUN / 'check_training.json'); assert checks['status'] == 'passed'
    for name, digest in checks['source_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in manifest['files_sha256'].items(): assert sha(EXTRA / name) == digest, name
    recipe = read(RUN / 'recipe.json')
    assert recipe['no_E_metrics_used_for_recipe'] and recipe['trials'] == list(TRIALS)
    sources = list((ROOT / 'scorer_four').glob('*.py'))
    sources += [p for p in (ROOT / 'tools').glob('*four*training*.py')]
    sources += [ROOT / 'tools/train_four.py', ROOT / 'tools/run_four_training.py',
                ROOT / 'scorer_training/replay.py', ROOT / 'scorer_training/train.py', ROOT / 'Scorer/scorer.py']
    inputs = [RUN / 'recipe.json', RUN / 'check_training.json', EXTRA / 'manifest.json',
              FEATURES_RUN / 'result.json', FEATURES_RUN / 'verification.json', CACHE / 'scorer_config.json']
    protocol = {**recipe, 'format': 'pooled_four_WM_scorer_protocol_v1', 'world_model_trials': list(TRIALS),
        'selection_caveat': read(BASE_RUN / 'result.json')['preserved_selection_caveat'],
        'runtime': flags(), 'world_model_identity_input': False, 'groups': {'fit':'B','development':'C','evaluation':'E'},
        'E_scope': 'Initial DINO pilot is a prior evaluation. No E observations or metrics choose this repeat recipe or C deployment settings.',
        'input_sha256': {str(p): sha(p) for p in inputs}, 'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    path = RUN / 'protocol.json'
    if path.exists(): assert read(path) == protocol
    else:
        for name in protocol['source_sha256']:
            target = RUN / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
        write(path, protocol)
    print('Locked four-model scorer training', sha(path), flush=True)


if __name__ == '__main__':
    main()
