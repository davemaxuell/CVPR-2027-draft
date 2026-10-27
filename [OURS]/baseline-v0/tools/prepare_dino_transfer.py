"""Extract only strict predictor tensors from the hash-verified author's release."""
import importlib
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Scorer.world_models import build_dino_predictor
from wm.common import ROOT, RUN, now, read, sha, write


def compare_encoders(assets):
    native = torch.load(assets / 'dinov2_vits14_pretrain.pth', map_location='cpu', weights_only=True)
    pinned = read(ROOT / 'manifest/model_revisions.json')['dino_encoder']
    hf = load_file(str(Path(pinned['path']) / 'model.safetensors'))
    mapped = {'cls_token': hf['embeddings.cls_token'], 'mask_token': hf['embeddings.mask_token'],
              'pos_embed': hf['embeddings.position_embeddings'],
              'patch_embed.proj.weight': hf['embeddings.patch_embeddings.projection.weight'],
              'patch_embed.proj.bias': hf['embeddings.patch_embeddings.projection.bias'],
              'norm.weight': hf['layernorm.weight'], 'norm.bias': hf['layernorm.bias']}
    for i in range(12):
        prefix = f'encoder.layer.{i}.'
        for suffix in ['weight', 'bias']:
            mapped[f'blocks.{i}.attn.qkv.{suffix}'] = torch.cat([hf[prefix + f'attention.attention.{part}.{suffix}'] for part in ['query', 'key', 'value']], dim=0)
            mapped[f'blocks.{i}.attn.proj.{suffix}'] = hf[prefix + f'attention.output.dense.{suffix}']
            for name in ['norm1', 'norm2', 'mlp.fc1', 'mlp.fc2']:
                mapped[f'blocks.{i}.{name}.{suffix}'] = hf[prefix + f'{name}.{suffix}']
        for j in (1, 2):
            mapped[f'blocks.{i}.ls{j}.gamma'] = hf[prefix + f'layer_scale{j}.lambda1']
    assert set(mapped) == set(native)
    for key in native:
        assert torch.equal(native[key], mapped[key]), key
    return {'status': 'passed', 'all_native_tensors_equal_pinned_HF_weights': True, 'tensor_count': len(native),
            'HF_repository': pinned['repo'], 'HF_revision': pinned['revision'],
            'HF_weights_sha256': sha(Path(pinned['path']) / 'model.safetensors'),
            'native_weights_sha256': sha(assets / 'dinov2_vits14_pretrain.pth'),
            'native_interpolation': {'offset': 0.1, 'antialias': False},
            'reason_for_native_source': 'Preserve source interpolation; installed HF implementation uses size-based interpolation instead of native offset 0.1'}


def main():
    assets = RUN / 'assets'
    archive = read(assets / 'download_provenance.json')
    assert archive['matches_osf_sha256']
    assert sha(archive['archive_path']) == archive['sha256']
    # Allow only the reviewed local architecture and installed torch/accelerate types.
    # weights_only never falls back to unrestricted pickle loading.
    sys.path.insert(0, str(ROOT / 'third_party/dino_wm'))
    native_predictor = build_dino_predictor(ROOT / 'third_party/dino_wm/models/vit.py')
    comparisons = {}
    for name in ['point_maze', 'pusht', 'wall_single']:
        path = assets / 'outputs' / name / 'checkpoints/model_latest.pth'
        cfg = yaml.safe_load((path.parents[1] / 'hydra.yaml').read_text())
        globals_ = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        allowed = []
        for identifier in globals_:
            assert identifier.startswith(('torch.nn.', 'torch.optim.', 'accelerate.', 'models.vit.', 'models.vqvae.', 'models.proprio.')) or identifier in ('builtins.dict', 'collections.defaultdict'), identifier
            module, attribute = identifier.rsplit('.', 1)
            allowed.append(getattr(importlib.import_module(module), attribute))
        with torch.serialization.safe_globals(allowed):
            saved = torch.load(path, map_location='cpu', weights_only=True)
        state = saved['predictor'].state_dict()
        compatible = set(state) == set(native_predictor.state_dict()) and all(state[k].shape == v.shape for k, v in native_predictor.state_dict().items())
        comparisons[name] = {'checkpoint_sha256': sha(path), 'epoch': saved['epoch'], 'history': cfg['num_hist'],
                             'position_shape': list(state['pos_embedding'].shape), 'predictor_parameters': sum(v.numel() for v in state.values()),
                             'strictly_compatible': compatible, 'predictor_config': cfg['predictor'], 'source_config_sha256': sha(path.parents[1] / 'hydra.yaml')}
        if name == 'point_maze':
            assert compatible and cfg['num_hist'] == 3 and cfg['concat_dim'] == 1
            native_predictor.load_state_dict(state, strict=True)
            torch.save(state, assets / 'point_maze_predictor.pt')
        del saved, state
    encoder = compare_encoders(assets)
    write(assets / 'encoder_weight_equivalence.json', encoder)
    report = {'status': 'selected', 'recorded_utc': now(), 'task': 'point_maze',
              'rationale': 'Exact three-frame predictor compatibility; navigation through obstacles is a closer source-task match than object pushing. Chosen before CE validation; no claim of measured superiority.',
              'source_checkpoint': str(assets / 'outputs/point_maze/checkpoints/model_latest.pth'),
              'source_checkpoint_sha256': comparisons['point_maze']['checkpoint_sha256'],
              'extracted_predictor': 'point_maze_predictor.pt', 'extracted_predictor_sha256': sha(assets / 'point_maze_predictor.pt'),
              'comparisons': comparisons, 'transferred_modules': ['predictor'],
              'excluded_modules': ['source action encoder', 'source proprio encoder', 'image decoder', 'optimizers'],
              'CE_command_embedding': 'fresh three-action embedding, width 10, shared initialization with fresh control',
              'CE_proprio': '10 zero-valued channels', 'policy_identity_sha256': read(RUN.parent / 'step4_smolvlm2_refined512/reference/reference.json')['policy_identity_sha256']}
    write(assets / 'transfer_selection.json', report)
    print(__import__('json').dumps(report, indent=2), flush=True)
    print(__import__('json').dumps(encoder, indent=2), flush=True)


if __name__ == '__main__':
    main()
