"""Pinned native DINO features and strictly compatible CE dynamics."""
import sys
import types

import torch
from torch.nn import functional as F
from torchvision import transforms

from Scorer.world_models import DinoCommandDynamics, DinoRGBEncoder, build_dino_predictor
from wm.common import ROOT, RUN, read, sha


def preprocessing():
    # Tensor resize/crop matches DinoRGBEncoder's declared source convention.
    return transforms.Compose([transforms.ToTensor(), transforms.Resize(224, antialias=True),
                               transforms.CenterCrop(224), transforms.Normalize([.5] * 3, [.5] * 3),
                               transforms.Resize(196, antialias=True)])


def build_encoder(device='cuda:0', assets=None):
    assets = assets or RUN / 'assets'
    provenance = read(assets / 'dinov2_native_provenance.json')
    for relative, expected in provenance['source_sha256'].items():
        assert sha(assets / 'dinov2_source' / relative) == expected
    weights = assets / 'dinov2_vits14_pretrain.pth'
    assert sha(weights) == provenance['weight_sha256']
    directory = str(assets / 'dinov2_source')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from dinov2.hub.backbones import dinov2_vits14
    native = dinov2_vits14(pretrained=False)
    native.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    assert native.interpolate_offset == .1 and not native.interpolate_antialias
    return DinoRGBEncoder(native).requires_grad_(False).eval().to(device)


def sdpa_attention(self, x):
    batch, tokens, _ = x.shape
    qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
    q, k, v = [value.reshape(batch, tokens, self.heads, -1).transpose(1, 2) for value in qkv]
    allowed = self.bias[:, :, :tokens, :tokens].bool()
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                            dropout_p=self.dropout.p if self.training else 0.0)
    return self.to_out(output.transpose(1, 2).reshape(batch, tokens, -1))


def enable_sdpa(model):
    for attention, _ in model.predictor.transformer.layers:
        attention.forward = types.MethodType(sdpa_attention, attention)
    return model


def build_dynamics(initialization='fresh', device='cuda:0', *, seed=5, sdpa=True, assets=None):
    assets = assets or RUN / 'assets'
    torch.manual_seed(seed)
    model = DinoCommandDynamics(build_dino_predictor(ROOT / 'third_party/dino_wm/models/vit.py'))
    if initialization == 'transfer':
        selection = read(assets / 'transfer_selection.json')
        path = assets / selection['extracted_predictor']
        assert sha(path) == selection['extracted_predictor_sha256']
        model.load_transferred_predictor(torch.load(path, map_location='cpu', weights_only=True))
    else:
        assert initialization == 'fresh'
    # Exactly the same new CE command embedding in both initialization trials.
    generator = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        model.command_embedding.weight.copy_(torch.randn(model.command_embedding.weight.shape, generator=generator))
    if sdpa:
        enable_sdpa(model)
    return model.to(device)
