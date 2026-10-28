"""Fix the matched NWM/CDiT training recipe before any CE fitting."""
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from nwm_training.common import INDEX, POLICY_ID, RUN, read, sha, verify_assets, verify_index, write
from policy.experimental_reference import verify_bundle


def main():
    assets = verify_assets(); data = verify_index()
    cpu = read(RUN / 'check_cpu.json'); gpu = read(RUN / 'check_gpu.json')
    assert cpu['status'] == gpu['status'] == 'passed'
    for report in (cpu, gpu):
        for name, digest in report['source_sha256'].items(): assert sha(ROOT / name) == digest, name
    predecessor = ROOT / 'runs/step10_smolvlm2_refined512_dino_pair_v1/result.json'
    assert read(predecessor)['status'] == 'complete'
    assert gpu['predecessor_result_sha256'] == sha(predecessor)
    reference = ROOT / 'runs/step4_smolvlm2_refined512/reference'
    policy = verify_bundle(reference)
    assert policy['policy_identity_sha256'] == POLICY_ID and policy['decoder']['stop_logit_bias'] == -.25
    order_path = INDEX.parent / 'sample_order.npy'; order = np.load(order_path)
    assert np.array_equal(np.sort(order), np.arange(data['train_samples']))
    names = list((ROOT / 'nwm_training').glob('*.py'))
    names += [p for p in (ROOT / 'tools').glob('*.py') if 'nwm' in p.name]
    names += [ROOT / n for n in ('wm/common.py', 'wm/training.py', 'wm/features.py', 'wm/dino.py',
        'wm/frozen.py', 'Scorer/world_models.py', 'Scorer/interfaces.py', 'policy/experimental_reference.py',
        'manifest/scene_partition.json', 'env/nwm_requirements.txt')]
    names += [ROOT / name for name in assets['source_sha256']]
    names += list((ROOT / 'env/nwm_dependencies/diffusers').rglob('*.py'))
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(names))}
    inputs = [RUN / 'assets/selection.json', RUN / 'check_cpu.json', RUN / 'check_gpu.json',
        INDEX / 'manifest.json', order_path, predecessor]
    plan = {'format': 'matched_NWM_CDiT_CE_protocol_v1', 'scope': 'SmolVLM2 refinement512: NWM transfer / fresh CDiT repeat',
        'policy_identity_sha256': POLICY_ID, 'STOP_bias': -.25, 'policy_reference': str(reference),
        'policy_selection_caveat': policy['selection'],
        'trials': {'transfer': 'nwm_transfer', 'fresh': 'cdit_fresh_dynamics'},
        'architecture': assets['architecture'], 'trainable_parameters': 49227680,
        'transfer': 'All released step100000 EMA tensors; fresh source optimizer', 'VAE_frozen': True,
        'training_group': 'A', 'validation_group': 'C', 'samples_per_trial': data['train_samples'],
        'validation_samples': data['validation_samples'], 'validation_limit': data['validation_limit'],
        'epochs': 1, 'batch_size': 64, 'updates': math.ceil(data['train_samples'] / 64), 'seed': 5,
        'sample_order': str(order_path), 'data_workers': 8,
        'optimizer': {'name': 'AdamW', 'lr': 1e-4, 'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': 0.,
                      'schedule': 'constant, matching the pinned native NWM recipe', 'clip_grad_norm': 1.},
        'EMA_decay': .9999, 'EMA_initialization': 'Copy initial model in each trial',
        'objective': 'Pinned native diffusion training_losses: 1000-step linear schedule, epsilon MSE and learned-range variance term',
        'posterior': 'Frozen FP32 VAE means/clamped logvars stored FP16; fresh FP32 posterior sample for each of five frames per example',
        'conditioning': {'real_context_frames': 4, 'command_order': ['FORWARD','LEFT','RIGHT'],
                         'command_table': assets['command_table'], 'relative_time': 1 / 128,
                         'blocked_motion': 'Keep nominal command; observed blocked view is target'},
        'precision': {'parameters': 'FP32', 'dynamics_autocast': 'BF16', 'VAE': 'FP32',
                      'matmul_TF32': False, 'cudnn_TF32': False, 'deterministic_algorithms': True,
                      'MHA_fastpath': False, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
        'eval_every': 4096, 'validation_randomness_seed': 5005,
        'selection': 'Minimum C dataset/action macro native loss of EMA, earliest-update tie; include initialization',
        'cache_batch_size': gpu['selected_cache_batch_size'], 'sampling_steps': gpu['selected_sampling_steps'],
        'sampling': 'Complete native ancestral epsilon/learned-variance sampler, clip_denoised false, one sample per movement; STOP identity',
        'candidate_sampling_batch': 'F/L/R together in fixed order, shared sampled real context; independent noise per candidate row',
        'common_encoder_decoded_images': 'Batch one with native DINO precision flags, identical weights/preprocessing to the existing scorer cache',
        'sampling_budget_rationale': gpu['sampling_budget_rationale'],
        'training_profile': gpu['training_profile'], 'budget_rationale': 'Exactly one full pass over the same matched group-A transitions for both variants; fixed before fitting',
        'no_E_for_fitting_selection_or_budget_choice': True,
        'E_predecessor_use': 'Completion dependency only; metrics are not read to choose this recipe',
        'next': 'Freeze both models, extend the existing B/C feature cache, then refit one pooled four-model scorer with a matched null control',
        'input_sha256': {str(p): sha(p) for p in inputs}, 'source_sha256': sources}
    path = RUN / 'protocol.json'
    if path.exists(): assert read(path) == plan
    else:
        for name in sources:
            destination = RUN / 'source' / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        write(path, plan)
    print('Locked NWM/CDiT repeat:', sha(path), flush=True)


if __name__ == '__main__':
    main()
