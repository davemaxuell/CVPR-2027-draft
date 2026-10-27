"""Freeze and verify each selected DINO model, then report the matched comparison."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.experimental_reference import verify_bundle
from wm.common import CORPUS, REFERENCE, ROOT, RUN, now, read, sha, write
from wm.features import CACHE, FeatureSamples, IndexedImages
from wm.frozen import FrozenDinoWorldModel
from wm.training import evaluate


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target)
    else:
        shutil.copy2(source, target)


def main():
    plan = read(RUN / 'plan.json')
    source = read(RUN / 'source_sha256.json')
    for relative, expected in source.items():
        assert sha(ROOT / relative) == sha(RUN / 'source' / relative) == expected
    policy = verify_bundle(REFERENCE)
    assert policy['policy_identity_sha256'] == plan['selected_policy_identity_sha256']
    assert policy['decoder']['stop_logit_bias'] == -.25
    control = Path(policy['selection']['control_reference'])
    for relative, expected in read(control / 'sha256.json').items():
        assert sha(control / relative) == expected
    assert not policy['selection']['original_stopping_safeguard_passed']
    cache = read(CACHE / 'manifest.json')
    seal = read(RUN / 'cache_seal.json')
    stat = (CACHE / 'features.npy').stat()
    assert [stat.st_size, stat.st_mtime_ns] == seal['features_size_mtime_ns']
    assert sha(CACHE / 'manifest.json') == seal['cache_manifest_sha256']
    shared = RUN / 'frozen/encoder'
    assets = RUN / 'assets'
    names = ['dinov2_vits14_pretrain.pth', 'dinov2_native_provenance.json', 'encoder_identity.json', 'encoder_weight_equivalence.json']
    names += [str(p.relative_to(assets)) for p in (assets / 'dinov2_source').rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')]
    for name in names:
        copy_verified(assets / name, shared / name)
    write(shared / 'file_sha256.json', {name: sha(shared / name) for name in names})
    runtime_files = ['wm/common.py', 'wm/dino.py', 'wm/frozen.py', 'Scorer/world_models.py', 'Scorer/interfaces.py', 'third_party/dino_wm/models/vit.py']
    results, checks = {}, {}
    loader = DataLoader(FeatureSamples('validation'), batch_size=plan['batch_size'], num_workers=2,
                        pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(3005))
    features = np.load(CACHE / 'features.npy', mmap_mode='r')
    for trial in ['transfer', 'fresh']:
        result = read(RUN / trial / 'result.json')
        assert result['status'] == 'complete'
        assert result['samples_seen'] == plan['samples_per_trial'] and result['updates'] == plan['updates']
        assert result['plan_sha256'] == sha(RUN / 'plan.json')
        assert result['encoder_identity_sha256'] == cache['encoder_identity_sha256']
        assert sha(RUN / trial / 'selected_model.pt') == result['model_sha256']
        name = 'dinowm_transfer' if trial == 'transfer' else 'dinov2_fresh_dynamics'
        target = RUN / 'frozen' / name
        copy_verified(RUN / trial / 'selected_model.pt', target / 'model.pt')
        config = {'format': 'frozen_ce_dino_dynamics_v1', 'trial': name, 'selected_step': result['selected_step'],
                  'selected_policy_identity_sha256': plan['selected_policy_identity_sha256'],
                  'policy_reference': str(REFERENCE), 'policy_decoder': policy['decoder'],
                  'encoder_identity_sha256': cache['encoder_identity_sha256'], 'encoder_assets': '../encoder',
                  'history_frames': 3, 'minimum_real_observations': 4, 'raw_visual_shape': [196, 384],
                  'pooled_visual_shape': [49, 384], 'condition_widths': {'zero_proprio': 10, 'learned_command': 10},
                  'action_indices': {'0': 'FORWARD', '1': 'LEFT', '2': 'RIGHT'}, 'STOP': 'identity current features; no dynamics call',
                  'precision': {'parameters': 'float32', 'dynamics_autocast': 'bfloat16', 'encoder_compute': 'float32', 'encoded_features': 'float16'},
                  'selection': result['metrics'], 'model_sha256': result['model_sha256'],
                  'plan_sha256': result['plan_sha256'], 'runtime_source_sha256': {p: source[p] for p in runtime_files}}
        write(target / 'config.json', config)
        write(target / 'file_sha256.json', {name: sha(target / name) for name in ['model.pt', 'config.json']})
        frozen = FrozenDinoWorldModel(target, expected_policy_identity=policy['policy_identity_sha256'], load_encoder=(trial == 'transfer'))
        assert not any(p.requires_grad for p in frozen.model.parameters()) and not frozen.model.training
        metrics, _ = evaluate(frozen.model, loader)
        assert abs(metrics['macro_mse'] - result['metrics']['macro_mse']) < 1e-7
        batch = next(iter(loader))
        x, a = batch['features'][:2].cuda(), batch['actions'][:2].cuda()
        pred = frozen.predict(x[:, :3], a[:, :2], a[:, 2], observations_seen=4, encoder_identity=frozen.encoder_identity_sha256)
        assert pred.shape == (2, 196, 384) and torch.isfinite(pred).all()
        for kwargs in [{'observations_seen': 3, 'encoder_identity': frozen.encoder_identity_sha256},
                       {'observations_seen': 4, 'encoder_identity': 'wrong encoder'}]:
            try:
                frozen.predict(x[:, :3], a[:, :2], a[:, 2], **kwargs)
            except AssertionError:
                pass
            else:
                raise AssertionError('Frozen model accepted incompatible context/encoder')
        try:
            FrozenDinoWorldModel(target, expected_policy_identity='wrong policy')
        except AssertionError:
            pass
        else:
            raise AssertionError('Frozen model accepted different policy identity')
        encoder_check = None
        if trial == 'transfer':
            # Re-encode source bytes from throughout the cache with the packaged encoder.
            image_set = IndexedImages()
            sample_ids = sorted(set(np.random.default_rng(5).choice(len(image_set), 64, replace=False).tolist()))
            pixels = torch.stack([image_set[i][1] for i in sample_ids]).cuda()
            with torch.inference_mode():
                actual = frozen.encoder.encoder.forward_features(pixels)['x_norm_patchtokens'].float().cpu()
            expected = torch.tensor(np.array(features[sample_ids], copy=True)).float()
            mse = (actual - expected).square().mean().item()
            assert mse < 5e-7
            assert not any(p.requires_grad for p in frozen.encoder.parameters())
            encoder_check = {'reencoded_samples': len(sample_ids), 'mse_vs_stored_float16': mse, 'trainable_parameters': 0}
        checks[name] = {'status': 'passed', 'frozen_parameters': True, 'reload_validation_macro_mse': metrics['macro_mse'],
                        'policy_encoder_and_warmup_guards': True, 'encoder_check': encoder_check}
        results[trial] = result
        del frozen
        torch.cuda.empty_cache()
    verification = {'status': 'passed', 'completed_utc': now(), 'trials': checks,
                    'matched_samples_and_update_budget': True, 'policy_reference_and_STOP_bias_unchanged': True,
                    'Step2_control_preserved': True, 'failed_stopping_safeguard_selection_preserved': True,
                    'encoder_identity_sha256': cache['encoder_identity_sha256'],
                    'selected_policy_identity_sha256': policy['policy_identity_sha256'],
                    'runtime_checks': read(RUN / 'check_runtime.json')['status'],
                    'training_restart_and_sample_mapping_checks': read(RUN / 'check_training.json')}
    write(RUN / 'final_verification.json', verification)
    result = {'status': 'complete', 'completed_utc': now(), 'scope': 'Step 5A: DINO transfer and matched fresh dynamics',
              'trials': results, 'training_samples_per_trial': plan['samples_per_trial'], 'validation_samples': 3072,
              'selected_policy_identity_sha256': policy['policy_identity_sha256'], 'encoder_identity_sha256': cache['encoder_identity_sha256'],
              'verification': 'final_verification.json', 'remaining_world_model_families': 'NWM/CDiT pair deferred to the later repeat in the agreed sequence'}
    write(RUN / 'result.json', result)
    lines = ['# Step 5A complete — DINO transfer and fresh dynamics', '', f'Completed: {now()}', '',
             'Both trials used the same frozen native DINOv2-S/14 encoder, new CE action embedding initialization, group-A samples, shuffled order, optimizer, learning-rate schedule and full-pass training budget. Only predictor initialization differed.',
             f"Each trial trained on {plan['samples_per_trial']:,} matched transitions for {plan['updates']:,} optimizer updates. Checkpoints were selected by mean feature-prediction MSE across six dataset/action groups in 3,072 fixed group-C expert-path samples.", '',
             '| Trial | Initial C MSE | Selected update | Selected C MSE | Copy-current C MSE |', '| --- | ---: | ---: | ---: | ---: |']
    for trial, row in results.items():
        initial = read(RUN / trial / 'evaluations/step_000000.json')['macro_mse']
        lines.append(f"| {trial} | {initial:.6f} | {row['selected_step']:,} | {row['metrics']['macro_mse']:.6f} | {row['metrics']['copy_current_macro_mse']:.6f} |")
    lines += ['', 'The transferred predictor is the author’s released PointMaze checkpoint, loaded strictly. Its three-frame, 404-channel, six-layer architecture matches the fresh control. Wall’s released one-frame predictor is incompatible with this fixed interface. Source action/proprio encoders, decoder and optimizer were excluded.',
              'Release provenance: the [author’s mirror](https://huggingface.co/datasets/gaoyuezhou/dino_wm) archive matched the original OSF SHA-256. Exact source/weight identities are in `assets/download_provenance.json` and `assets/transfer_selection.json`.',
              'All 175 native DINO encoder tensors equal the previously pinned Hugging Face weights. Native position interpolation is retained. Features were computed in float32 and stored as float16 after a precision check; both trials share the same immutable cache.',
              'Selected bundles: `frozen/dinowm_transfer/` and `frozen/dinov2_fresh_dynamics/`, with shared `frozen/encoder/`. Load through `wm.frozen.FrozenDinoWorldModel` with the selected policy identity. Require four actual observations before prediction; the DINO model receives the last three. STOP uses current features without a dynamics call.',
              'The navigation policy remains refined SmolVLM2 checkpoint 512 with STOP logit bias −0.25. Its deliberate selection despite the failed stopping safeguard is retained; the original Step 2 control is unchanged.',
              '[Final verification](final_verification.json) passed: strict checkpoint loading, sample/action alignment, group separation, native/SDPA output and gradient parity, optimizer restart, frozen encoder/model state, and packaged model validation replay.',
              'Limit: validation uses held-out expert paths and contains no blocked forward examples. Group-A training retains those failures. Feature prediction error and command sensitivity do not establish downstream navigation success or benefit from scorer overrides.',
              'The DINO pair is ready for Step 6 scorer-label collection. NWM/CDiT remains the later paired repeat; no group-B labels, group-E final evaluation, or scorer training were performed in this step.', '']
    (RUN / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
