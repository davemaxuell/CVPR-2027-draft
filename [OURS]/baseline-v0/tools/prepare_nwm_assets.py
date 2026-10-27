"""CPU-only strict validation and extraction of already authorized model assets."""
import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from nwm_training.common import (INDEX, NWM_FILE, NWM_REVISION, NWM_SHA, POLICY_ID, RUN,
                                 VAE_DIRECTORY, VAE_REVISION, now, read, sha, verify_index, write)
from nwm_training.model import build_vae, command_table, native_model


def main():
    torch.set_num_threads(4)
    index = verify_index()
    assert sha(NWM_FILE) == NWM_SHA
    checkpoint = torch.load(NWM_FILE, map_location='cpu', weights_only=False)
    assert checkpoint['train_steps'] == 100000
    model = native_model()
    state = {name.removeprefix('_orig_mod.'): value for name, value in checkpoint['ema'].items()}
    assert len(state) == len(checkpoint['ema'])
    model.load_state_dict(state, strict=True)
    assert all(torch.isfinite(value).all() for value in state.values())
    assert sum(p.numel() for p in model.parameters()) == 49227680
    assert model.pos_embed.shape == (5, 196, 384) and model.out_channels == 8
    extracted = RUN / 'assets/nwm_cdit_s_ema.pt'
    extracted.parent.mkdir(parents=True, exist_ok=True)
    if extracted.exists():
        saved = torch.load(extracted, map_location='cpu', weights_only=True)
        assert saved.keys() == state.keys() and all(torch.equal(saved[k], state[k]) for k in state)
    else:
        temporary = extracted.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(extracted)
    vae = build_vae()
    assert all(torch.isfinite(v).all() for v in vae.state_dict().values())
    files = [NWM_FILE, extracted, VAE_DIRECTORY / 'config.json', VAE_DIRECTORY / 'diffusion_pytorch_model.safetensors']
    sources = [ROOT / 'third_party/nwm/models.py', ROOT / 'third_party/nwm/train.py',
               ROOT / 'third_party/nwm/isolated_nwm_infer.py', ROOT / 'third_party/nwm/datasets.py',
               ROOT / 'third_party/nwm/misc.py', ROOT / 'third_party/nwm/config/data_config.yaml']
    sources += sorted((ROOT / 'third_party/nwm/diffusion').glob('*.py'))
    record = {'status': 'verified_cpu', 'verified_utc': now(), 'selected_policy_identity_sha256': POLICY_ID,
        'source_model': 'facebook/nwm', 'source_revision': NWM_REVISION, 'source_file': str(NWM_FILE),
        'source_checkpoint_step': 100000, 'state_selected': 'ema', 'extracted_ema': str(extracted),
        'strict_tensor_count': len(state), 'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'architecture': 'native CDiT-S/2, depth12 width384 heads6, four contexts, 28x28 latent, epsilon+learned-variance',
        'transfer_scope': 'All 219 EMA tensors including learned positions/action/time embeddings; discard source optimizer/scaler',
        'fresh_scope': 'Exact native architecture and initialization, same trainable parameters and conditioning',
        'vae': {'repo': 'stabilityai/sd-vae-ft-ema', 'revision': VAE_REVISION, 'directory': str(VAE_DIRECTORY),
                'frozen': True, 'parameters': sum(p.numel() for p in vae.parameters()), 'latent_scale': .18215},
        'command_order': ['FORWARD', 'LEFT', 'RIGHT'], 'command_table': command_table().tolist(),
        'command_definition': 'Nominal local [forward metres, left metres, left-positive yaw radians]; x/y divided by .25 then source min/max normalized. Collision never changes command.',
        'relative_time': '1/128 for one CE command; no physical seconds assumption',
        'preprocessing': 'RGB float32 /255; antialiased tensor Resize(shortest side224), CenterCrop224, normalize mean/std .5; same CE crop as DINO and NWMVisualBridge',
        'native_preprocessing_difference': 'The official NWM source uses 4:3 crop then square resize. The agreed CE crop adaptation is kept identical in both paired trials.',
        'index_manifest_sha256': sha(INDEX / 'manifest.json'), 'training_samples': index['train_samples'],
        'validation_samples': index['validation_samples'], 'unique_images': index['unique_images'],
        'no_GPU_used': True, 'no_E_data_used': True, 'inference_and_training_validation': 'pending GPU preflight after current DINO evaluation',
        'versions': {name: importlib.metadata.version(name) for name in ('torch','torchvision','diffusers','timm','safetensors')},
        'files_sha256': {str(p): sha(p) for p in files},
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    path = RUN / 'assets/selection.json'
    if path.exists():
        prior = read(path); record['verified_utc'] = prior['verified_utc']; assert prior == record
    else:
        write(path, record)
    print('Verified NWM EMA and frozen VAE on CPU:', sha(path), flush=True)


if __name__ == '__main__':
    main()
