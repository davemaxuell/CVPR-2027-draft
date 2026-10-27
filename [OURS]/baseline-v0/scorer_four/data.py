"""Reuse full actual prefixes and labels; append immutable NWM/CDiT futures."""
import numpy as np
import torch

from scorer_features.dataset import FrozenScorerDataset as BaseDataset
from scorer_training.data import ScorerData as BaseTrainingData
from scorer_four.common import ADDITIONS, CACHE, FEATURES_RUN, read, sha, verify_features_protocol


def verified_extension():
    verify_features_protocol()
    manifest = read(CACHE / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['protocol_sha256'] == sha(FEATURES_RUN / 'protocol.json')
    assert sha(FEATURES_RUN / 'verification.json') == manifest['verification_sha256']
    assert read(FEATURES_RUN / 'verification.json')['status'] == 'passed'
    return manifest


class FrozenScorerDataset(BaseDataset):
    def __init__(self, group):
        verified_extension(); super().__init__(group)
        for name in ADDITIONS:
            self.predictions[name] = np.load(CACHE / name / 'future.npy', mmap_mode='r')

    def futures(self, index, representation):
        if representation not in ADDITIONS: return super().futures(index, representation)
        value = np.array(self.predictions[representation][self.roots[index]['id']], copy=True)
        return torch.from_numpy(value)[None]


class ScorerData(BaseTrainingData):
    def __init__(self, device='cuda:0'):
        verified_extension(); super().__init__(device)
        extra = torch.stack([torch.from_numpy(np.load(CACHE / name / 'future.npy')) for name in ADDITIONS]).to(self.device)
        self.predictions = torch.cat((self.predictions, extra), dim=0)
        assert self.predictions.shape == (4, 1280, 4, 49, 384)
        assert not self.predictions.requires_grad
