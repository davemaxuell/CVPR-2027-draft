"""Finalize the paired dynamics repeat after native and common-feature verification."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nwm_training.common import POLICY_ID, RUN, now, read, sha, write
from nwm_training.protocol import verify_protocol
from policy.experimental_reference import verify_bundle


def main():
    plan = verify_protocol(); frozen = read(RUN / 'freeze_verification.json'); quality = read(RUN / 'quality/result.json')
    assert frozen['status'] == 'passed' and quality['status'] == 'complete'
    assert frozen['protocol_sha256'] == quality['protocol_sha256'] == sha(RUN / 'protocol.json')
    assert quality['shared_samples'] == 3072 and quality['no_E_used']
    training = {name: read(RUN / name / 'result.json') for name in ('transfer','fresh')}
    for name, value in training.items():
        assert value['updates'] == plan['updates'] and value['samples_seen'] == plan['samples_per_trial']
        assert sha(value['selected_checkpoint']) == value['selected_checkpoint_sha256']
        assert sha(value['selected_model']) == value['selected_model_sha256']
        trial = plan['trials'][name]
        folder = RUN / 'frozen' / trial
        for relative, digest in read(folder / 'file_sha256.json').items(): assert sha(folder / relative) == digest
        assert quality['models'][trial]['model_sha256'] == sha(folder / 'model.pt')
    for name, report in quality['models'].items():
        assert report['samples'] == 3072 and len(report['row_sha256']) == 3072
        for file, digest in report['row_sha256'].items(): assert sha(RUN / 'quality' / name / 'rows' / file) == digest
    policy = verify_bundle(Path(plan['policy_reference']))
    assert policy['policy_identity_sha256'] == POLICY_ID and policy['decoder']['stop_logit_bias'] == -.25
    control = Path(policy['selection']['control_reference'])
    for file, digest in read(control / 'sha256.json').items(): assert sha(control / file) == digest
    verification = {'status': 'passed', 'completed_utc': now(), 'protocol_sha256': sha(RUN / 'protocol.json'),
        'native_freeze_verification_sha256': sha(RUN / 'freeze_verification.json'),
        'common_C_diagnostic_sha256': sha(RUN / 'quality/result.json'), 'C_rows_verified': 4 * 3072,
        'selected_models_and_Step2_control_preserved': True, 'no_E_used_for_fitting_or_selection': True}
    write(RUN / 'final_verification.json', verification)
    result = {'status': 'complete', 'completed_utc': now(), 'protocol_sha256': sha(RUN / 'protocol.json'),
        'selected_policy_identity_sha256': POLICY_ID, 'STOP_bias': -.25,
        'scope': 'Step 5B NWM/CDiT transfer/fresh pair for the same frozen SmolVLM2 policy',
        'training': training, 'common_C_feature_diagnostics': quality['models'],
        'frozen_bundles': {name: str(RUN / 'frozen' / name) for name in plan['trials'].values()},
        'final_verification_sha256': sha(RUN / 'final_verification.json'),
        'navigation_success_claim': False,
        'next': 'Step 7B: extend the existing measured B/C roots with both NWM/CDiT predictions; then train one pooled four-world-model scorer and matched null control'}
    write(RUN / 'result.json', result)
    with (RUN / 'common_features.csv').open('w', newline='') as file:
        writer = csv.writer(file); writer.writerow(['model','C_samples','macro_pooled_MSE','copy_current_MSE','prediction_seconds_three_candidates'])
        for name, value in quality['models'].items():
            writer.writerow([name, value['samples'], value['macro_pooled_mse'], value['copy_current_macro_pooled_mse'], value['prediction_mean_seconds_three_candidates']])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in ('transfer','fresh'):
        evaluations = [read(p)['candidate'] for p in sorted((RUN / name / 'validation').glob('step_*.json'))]
        axes[0].plot([r['step'] for r in evaluations], [r['macro_loss'] for r in evaluations], marker='o', label=name)
    axes[0].set(xlabel='Optimizer updates', ylabel='C macro native diffusion loss', title='Matched NWM/CDiT training')
    axes[0].legend(); axes[0].grid(alpha=.2)
    names = list(quality['models']); values = [quality['models'][n]['macro_pooled_mse'] for n in names]
    axes[1].barh(range(len(names)), values); axes[1].set_yticks(range(len(names)), labels=names)
    axes[1].axvline(quality['models'][names[0]]['copy_current_macro_pooled_mse'], color='black', linestyle='--', label='copy current')
    axes[1].set(xlabel='C pooled DINO feature MSE', title='Same 3,072 observed transitions'); axes[1].legend()
    figure.savefig(RUN / 'results.png', dpi=180); figure.savefig(RUN / 'results.svg'); plt.close(figure)
    lines = ['# Step 5B — matched NWM/CDiT dynamics', '', f"Completed: {result['completed_utc']}", '',
        'Both variants completed one pass over all 2,422,469 matched group-A transitions: 37,852 updates at batch 64. The NWM variant starts from the released CDiT-S/2 EMA; the fresh variant uses the same native architecture, VAE, command interface, optimizer, data order and budget.', '',
        'Checkpoint selection uses the same fixed 3,072 group-C expert-path transitions and native epsilon/learned-variance diffusion objective. EMA weights are frozen afterward. The common feature diagnostic below does not change checkpoint or sampler selection. E was not used for fitting or these choices.', '',
        '| Dynamics | Selected update | Native C diffusion loss |', '| --- | ---: | ---: |']
    for name, value in training.items(): lines.append(f"| {plan['trials'][name]} | {value['selected_step']:,} | {value['metrics']['macro_loss']:.6f} |")
    lines += ['', '| Frozen model | C pooled feature MSE |', '| --- | ---: |']
    for name, value in quality['models'].items(): lines.append(f"| {name} | {value['macro_pooled_mse']:.6f} |")
    lines += [f"| Copy current features | {quality['models'][names[0]]['copy_current_macro_pooled_mse']:.6f} |", '',
        'The common diagnostic compares 49×384 pooled DINO features on identical observations. Its scale differs from the earlier raw 196×384 DINO-only diagnostic. These expert-path transitions have no collisions; these errors do not establish recovery behavior or navigation success.', '',
        'The sampler completes 50 native ancestral diffusion steps for a fixed F/L/R batch, one sample per action, with shared real context and recorded seeds. This is an explicit compute budget reduction from the upstream 250-step evaluation. STOP is the current view and never calls the world model. Complete decoded RGB passes through the original frozen DINO encoder.', '',
        'Refined SmolVLM2 checkpoint 512 and STOP bias −0.25 remain fixed. The deliberate choice despite its failed original stopping safeguard, and the preserved Step 2 control, remain in the bundles. The VAE is frozen. Blocked movement retains the commanded action and uses the actual blocked view as its target.', '',
        '![Training and common feature diagnostics](results.png)', '',
        '[Result](result.json), [verification](final_verification.json), [protocol](protocol.json), [CSV](common_features.csv), [SVG](results.svg).', '',
        'Next: extend the existing B/C feature cache with these two models, keeping the measured labels and selected policy fixed, then refit a common four-model scorer and matched null control.', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    write(RUN / 'status.json', {'status': 'complete', 'completed_utc': now(), 'result': str(RUN / 'result.json')})
    write(ROOT / 'runs/automatic_progression.json', {'status': 'active', 'authorization': 'User requested automatic progression on 2026-10-27',
        'completed_trial': 'SmolVLM2 initial DINO pilot; NWM/CDiT dynamics repeat', 'current_step': '7B',
        'next': result['next'], 'updated_utc': now(), 'policy_identity_sha256': POLICY_ID})
    print('NWM/CDiT pair complete and frozen', flush=True)


if __name__ == '__main__':
    main()
