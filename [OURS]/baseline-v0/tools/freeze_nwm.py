"""Package selected EMA weights, shared VAE and encoder identity; verify native replay."""
import gc
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

from nwm_training.common import CACHE, INDEX, POLICY_ID, RUN, VAE_DIRECTORY, now, read, sha, write
from nwm_training.data import IndexedVAEImages, PosteriorSamples
from nwm_training.engine import configure, diffusion, evaluate
from nwm_training.frozen import FrozenNwmWorldModel
from nwm_training.protocol import verify_cache, verify_protocol
from policy.experimental_reference import verify_bundle


def copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): assert sha(destination) == sha(source), destination
    else: shutil.copy2(source, destination)


def main():
    configure(); plan = verify_protocol(); verify_cache()
    vae = RUN / 'frozen/vae'
    for name in ('config.json', 'diffusion_pytorch_model.safetensors'):
        copy_exact(VAE_DIRECTORY / name, vae / name)
    write(vae / 'file_sha256.json', {n: sha(vae / n) for n in ('config.json', 'diffusion_pytorch_model.safetensors')})
    encoder = ROOT / 'runs/step5_smolvlm2_refined512/frozen/encoder'
    shared = {str(vae / n): digest for n, digest in read(vae / 'file_sha256.json').items()}
    shared.update({str(encoder / n): digest for n, digest in read(encoder / 'file_sha256.json').items()})
    for path, digest in shared.items(): assert sha(path) == digest, path
    encoder_id = read(encoder / 'encoder_identity.json')['sha256']
    reports = {}; checkpoint_count = 0
    for trial, name in plan['trials'].items():
        trained = read(RUN / trial / 'result.json')
        assert trained['status'] == 'complete' and trained['protocol_sha256'] == sha(RUN / 'protocol.json')
        assert trained['updates'] == plan['updates'] and trained['samples_seen'] == plan['samples_per_trial']
        assert sha(trained['selected_model']) == trained['selected_model_sha256']
        evaluations = []
        for path in sorted((RUN / trial / 'checkpoints').glob('*.json')):
            record = read(path); assert sha(path.with_suffix('.pt')) == record['checkpoint_sha256']
            assert np.isfinite(record['macro_loss']); evaluations.append(record); checkpoint_count += 1
        assert [e['step'] for e in evaluations] == [0, *range(4096, plan['updates'], 4096), plan['updates']]
        selected = min(evaluations, key=lambda r: (r['macro_loss'], r['step']))
        assert selected['step'] == trained['selected_step']
        folder = RUN / 'frozen' / name
        copy_exact(trained['selected_model'], folder / 'model.pt')
        config = {'format': 'frozen_NWM_CDiT_CE_v1', 'trial': name, 'selected_update': selected['step'],
            'policy_identity_sha256': POLICY_ID, 'STOP_bias': -.25, 'selection_caveat': plan['policy_selection_caveat'],
            'protocol_sha256': sha(RUN / 'protocol.json'), 'source_checkpoint': trained['selected_checkpoint'],
            'source_checkpoint_sha256': trained['selected_checkpoint_sha256'], 'state': 'EMA',
            'encoder_identity_sha256': encoder_id, 'DINO_encoder_directory': str(encoder),
            'VAE_directory': str(vae), 'latent_scale': .18215, 'VAE_parameter_precision': 'FP32',
            'posterior_context': 'sample from FP16 rounded mean/logvar using explicit shared context seed',
            'minimum_real_observations': 4, 'candidate_order': ['FORWARD','LEFT','RIGHT'],
            'candidate_batch_size': 3, 'samples_per_candidate': 1, 'sampling_steps': plan['sampling_steps'],
            'command_table': plan['conditioning']['command_table'], 'relative_time': 1 / 128,
            'precision': plan['precision'], 'common_DINO_runtime_flags': 'FP32 batch1, matmul TF32 false, cuDNN TF32 true, deterministic algorithms false',
            'shared_files_sha256': shared, 'runtime_source_sha256': plan['source_sha256']}
        if (folder / 'config.json').exists(): assert read(folder / 'config.json') == config
        else: write(folder / 'config.json', config)
        write(folder / 'file_sha256.json', {n: sha(folder / n) for n in ('model.pt', 'config.json')})
        frozen = FrozenNwmWorldModel(name)
        assert all(not p.requires_grad for module in (frozen.model, frozen.vae, frozen.encoder) for p in module.parameters())
        loader = DataLoader(PosteriorSamples('validation'), batch_size=64, num_workers=2, pin_memory=True,
                            generator=torch.Generator().manual_seed(3005))
        report, _ = evaluate(frozen.model, diffusion(), loader)
        assert report == trained['metrics']
        samples = np.load(INDEX / 'validation_samples.npy', mmap_mode='r')
        moments = np.load(CACHE / 'posteriors.npy', mmap_mode='r')
        context = torch.from_numpy(np.array(moments[samples[0, :4]], copy=True))[None]
        before = context.clone(); rng = torch.get_rng_state().clone()
        predictions = frozen.predict_movements(context, context_seed=505, sampling_seed=506)
        repeated = frozen.predict_movements(context, context_seed=505, sampling_seed=506)
        assert torch.equal(predictions, repeated) and torch.equal(context, before)
        assert torch.equal(torch.get_rng_state(), rng)
        try: frozen.predict_movements(context[:, 1:], context_seed=505, sampling_seed=506)
        except AssertionError: pass
        else: raise AssertionError('Insufficient context must be rejected')
        reports[name] = {'selected_step': selected['step'], 'native_selected_validation_exact': True,
            'parameters_frozen': True, 'complete_pipeline_prediction_repeat_exact': True,
            'real_context_unchanged': True, 'insufficient_context_rejected': True,
            'features_shape': list(predictions.shape), 'config_sha256': sha(folder / 'config.json'),
            'model_sha256': sha(folder / 'model.pt')}
        del frozen, predictions, repeated, loader; gc.collect(); torch.cuda.empty_cache()
    reference = verify_bundle(Path(plan['policy_reference']))
    assert reference['policy_identity_sha256'] == POLICY_ID and not reference['selection']['original_stopping_safeguard_passed']
    control = Path(reference['selection']['control_reference'])
    for name, digest in read(control / 'sha256.json').items(): assert sha(control / name) == digest
    verify_protocol()
    record = {'status': 'passed', 'completed_utc': now(), 'protocol_sha256': sha(RUN / 'protocol.json'),
        'checkpoint_files_hashed': checkpoint_count, 'trials': reports, 'policy_and_Step2_control_preserved': True,
        'selected_policy_identity_sha256': POLICY_ID, 'STOP_bias': -.25, 'no_E_used': True}
    write(RUN / 'freeze_verification.json', record); print(record, flush=True)


if __name__ == '__main__':
    main()
