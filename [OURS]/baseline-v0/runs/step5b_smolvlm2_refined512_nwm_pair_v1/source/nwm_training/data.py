"""Shared observed images and posterior moments, with no future pose inputs."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from nwm_training.common import CACHE, INDEX
from nwm_training.model import LATENT_SCALE, preprocessing
from wm.features import IndexedImages


class IndexedVAEImages(IndexedImages):
    def __init__(self, start=0, end=None):
        super().__init__(start=start, end=end, index=INDEX)
        self.transform = preprocessing()


def posterior_moments(vae, pixels):
    distribution = vae.encode(pixels).latent_dist
    assert tuple(distribution.mean.shape[1:]) == (4, 28, 28)
    moments = torch.cat([distribution.mean, distribution.logvar], dim=1)
    assert torch.isfinite(moments).all()
    return moments


def sample_posterior(moments, *, generator=None):
    assert moments.shape[-3:] == (8, 28, 28)
    mean, logvar = moments.float().chunk(2, dim=-3)
    noise = torch.randn(mean.shape, dtype=torch.float32, device=mean.device, generator=generator)
    return (mean + (.5 * logvar.clamp(-30., 20.)).exp() * noise) * LATENT_SCALE


class PosteriorSamples(Dataset):
    def __init__(self, split='train', *, order=None, start=0, stop=None, cache=CACHE):
        assert split in ('train', 'validation')
        self.samples = np.load(INDEX / f'{split}_samples.npy', mmap_mode='r')
        self.cache, self.order, self.start = cache, order, start
        self.stop = len(self.samples) if stop is None else stop
        self.moments, self.pid = None, None

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, slot):
        slot += self.start
        if self.order is not None:
            slot = int(self.order[slot])
        if self.moments is None or self.pid != os.getpid():
            self.moments = np.load(self.cache / 'posteriors.npy', mmap_mode='r')
            self.pid = os.getpid()
        row = self.samples[slot]
        # Four actual context frames and the measured target; the target never
        # appears in conditioning(). No poses/displacements enter this object.
        value = np.array(self.moments[row[:5]], copy=True)
        return {'moments': torch.from_numpy(value), 'action': int(row[8]),
                'dataset': int(row[9]), 'source': int(row[10]), 'collided': bool(row[11]),
                'blocked': bool(row[12]), 'slot': int(slot)}
