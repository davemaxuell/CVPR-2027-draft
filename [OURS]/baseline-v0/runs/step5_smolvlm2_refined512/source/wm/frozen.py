"""Frozen DINO dynamics tied to one declared navigation policy and visual encoder."""
from pathlib import Path

import torch

from wm.common import ROOT, read, sha
from wm.dino import build_dynamics, build_encoder, preprocessing


class FrozenDinoWorldModel:
    def __init__(self, bundle, *, expected_policy_identity, device='cuda:0', load_encoder=False):
        self.bundle = Path(bundle).resolve()
        for relative, expected in read(self.bundle / 'file_sha256.json').items():
            assert sha(self.bundle / relative) == expected, ('Frozen world-model file changed', relative)
        self.config = read(self.bundle / 'config.json')
        assert self.config['selected_policy_identity_sha256'] == expected_policy_identity
        assert self.config['minimum_real_observations'] == 4
        for relative, expected in self.config['runtime_source_sha256'].items():
            assert sha(ROOT / relative) == expected
        self.device = torch.device(device)
        self.model = build_dynamics('fresh', device=device)
        self.model.load_state_dict(torch.load(self.bundle / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        self.model.requires_grad_(False).eval()
        self.policy_identity_sha256 = expected_policy_identity
        self.encoder_identity_sha256 = self.config['encoder_identity_sha256']
        self.encoder = None
        if load_encoder:
            assets = (self.bundle / self.config['encoder_assets']).resolve()
            for relative, expected in read(assets / 'file_sha256.json').items():
                assert sha(assets / relative) == expected
            assert read(assets / 'encoder_identity.json')['sha256'] == self.encoder_identity_sha256
            self.encoder = build_encoder(device, assets=assets)

    @torch.inference_mode()
    def encode_images(self, images):
        assert self.encoder is not None, 'Load the shared frozen encoder to encode RGB'
        pixels = torch.stack([preprocessing()(image) for image in images]).to(self.device)
        with torch.autocast(self.device.type, enabled=False):
            values = self.encoder.encoder.forward_features(pixels)['x_norm_patchtokens']
        # Same quantization convention as the shared fixed-feature training cache.
        return values.to(torch.float16)

    @torch.inference_mode()
    def predict(self, real_patches, executed_actions, candidate, *, observations_seen, encoder_identity):
        assert observations_seen >= self.config['minimum_real_observations'], 'Use the base policy during warm-up'
        assert encoder_identity == self.encoder_identity_sha256
        assert real_patches.shape[1:] == (3, 196, 384)
        self.model.eval()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            return self.model(real_patches, executed_actions, candidate)
