"""Freeze the predeclared repeated E pilot after four-model C calibration."""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from navigation_four.common import (CALIBRATION, EVALUATION as RUN, PRIOR_EVALUATION, TRAIN, TRIALS,
    POLICY_ID, read, sha, write)
from navigation_four.jobs import source_hashes
from tools.run_four_calibration import verify as verify_calibration


def main():
    calibration = verify_calibration(); recipe = read(CALIBRATION / 'recipe.json')
    result = read(CALIBRATION / 'result.json'); prior = read(PRIOR_EVALUATION / 'result.json')
    assert result['status'] == prior['status'] == 'complete'
    assert result['E_not_used_for_training_or_selection'] and prior['no_E_tuning']
    assert read(PRIOR_EVALUATION / 'verification.json')['status'] == 'passed'
    assert result['protocol_sha256'] == sha(CALIBRATION / 'protocol.json')
    selection = read(PRIOR_EVALUATION / 'selection.json')
    assert selection['group'] == 'E' and selection['seed'] == recipe['E_plan']['selection_seed']
    assert all(len(ids) == 112 for ids in selection['episodes'].values())
    assert len({scene for scenes in selection['episode_scenes'].values() for scene in scenes.values()}) == 11
    earliest = min(read(p)['completed_utc'] for p in (PRIOR_EVALUATION / 'evaluation/policy/episodes').glob('*.json'))
    assert recipe['declared_utc'] < earliest and read(TRAIN / 'recipe.json')['declared_utc'] < earliest
    if (RUN / 'selection.json').exists(): assert sha(RUN / 'selection.json') == sha(PRIOR_EVALUATION / 'selection.json')
    else:
        RUN.mkdir(parents=True, exist_ok=True); shutil.copy2(PRIOR_EVALUATION / 'selection.json', RUN / 'selection.json')
    for path,digest in selection['dataset_file_sha256'].items(): assert sha(path) == digest
    frozen = {}
    for variant in ('pooled','null'):
        folder = CALIBRATION / 'frozen' / variant
        for name,digest in read(folder / 'file_sha256.json').items(): assert sha(folder / name) == digest
        frozen[variant] = read(folder / 'bundle.json')
        assert frozen[variant]['policy_identity_sha256'] == POLICY_ID and frozen[variant]['STOP_bias'] == -.25
    systems = [{'name':'policy','mode':'policy','checkpoint':None,'world_model':None,'margin':None},
        {'name':'null','mode':'null','checkpoint':frozen['null']['checkpoint'],'world_model':None,'margin':frozen['null']['margin']},
        {'name':'actual','mode':'actual','checkpoint':frozen['pooled']['checkpoint'],'world_model':None,'margin':frozen['pooled']['margin']}]
    for name in TRIALS:
        systems.append({'name':name,'mode':'predicted','checkpoint':frozen['pooled']['checkpoint'],
            'world_model':name,'margin':frozen['pooled']['margin']})
    for system in systems: system['checkpoint_sha256'] = sha(system['checkpoint']) if system['checkpoint'] else None
    assert [s['name'] for s in systems] == recipe['E_plan']['systems']
    checks = read(RUN / 'check_reporting.json'); assert checks['status'] == 'passed'
    assert checks['source_sha256'] == sha(ROOT / 'tools/finalize_four_evaluation.py')
    sources = source_hashes()
    for path in (ROOT / 'tools').glob('*four*evaluation*.py'): sources[str(path.relative_to(ROOT))] = sha(path)
    sources['tools/run_four_calibration.py'] = sha(ROOT / 'tools/run_four_calibration.py')
    protocol = {'format':'four_WM_controlled_E_pilot_protocol_v1','policy_identity_sha256':POLICY_ID,'STOP_bias':-.25,
        'calibration_result_sha256':sha(CALIBRATION / 'result.json'), 'calibration_protocol_sha256':sha(CALIBRATION / 'protocol.json'),
        'recipe_sha256':sha(CALIBRATION / 'recipe.json'), 'selection_sha256':sha(RUN / 'selection.json'),
        'prior_E_result_sha256':sha(PRIOR_EVALUATION / 'result.json'), 'prior_E_verification_sha256':sha(PRIOR_EVALUATION / 'verification.json'),
        'reporting_checks_sha256':sha(RUN / 'check_reporting.json'), 'systems':systems,'workers':4,'sampling_seed':0,
        'episodes_per_system':224,'scope':recipe['E_plan']['scope'], 'RxR_protocol':prior['RxR_protocol'],
        'confidence_intervals':'Paired scene-cluster bootstrap, 2000 resamples, seed 10',
        'actual_control':'Privileged measured next views; identical pooled scorer and margin',
        'recipe_declared_before_initial_E_results':True, 'E_is_repeated_population':True, 'no_E_tuning':True,
        'source_sha256':sources}
    if (RUN / 'protocol.json').exists(): assert read(RUN / 'protocol.json') == protocol
    else:
        for name in sources:
            dest = RUN / 'source' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
        write(RUN / 'protocol.json', protocol)
    print('Locked four-model E pilot', sha(RUN / 'protocol.json'), flush=True)


if __name__ == '__main__':
    main()
