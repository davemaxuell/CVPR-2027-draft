"""Hash-checked RGB loading and compact fixed-encoder features."""
import hashlib
import io
import os
import sqlite3

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from wm.common import DATA, RUN, read
from wm.dino import preprocessing

CACHE = DATA / 'wm_dino_features_refined512_v1'


class IndexedImages(Dataset):
    def __init__(self, start=0, end=None, index=None):
        self.index = index or RUN / 'data_index'
        self.manifest = read(self.index / 'manifest.json')
        self.start = start
        self.end = self.manifest['unique_images'] if end is None else end
        self.transform = preprocessing()
        self.connection, self.pid = None, None

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, slot):
        number = self.start + slot
        if self.connection is None or self.pid != os.getpid():
            self.connection = sqlite3.connect(f"file:{self.index / 'images.sqlite'}?mode=ro&immutable=1", uri=True)
            self.pid = os.getpid()
        expected, path = self.connection.execute('SELECT sha256,path FROM images WHERE id=?', (int(number),)).fetchone()
        with open(path, 'rb') as file:
            raw = file.read()
        assert hashlib.sha256(raw).hexdigest() == expected, ('Source image changed', path)
        with Image.open(io.BytesIO(raw)) as image:
            assert image.mode == 'RGB' and image.size == (640, 480)
            pixels = self.transform(image)
        return number, pixels


class FeatureSamples(Dataset):
    def __init__(self, split='train', *, order=None, start=0, stop=None, cache=None, index=None):
        assert split in ('train', 'validation')
        self.cache = cache or CACHE
        self.index = index or RUN / 'data_index'
        self.samples = np.load(self.index / f'{split}_samples.npy', mmap_mode='r')
        self.order = order
        self.start = start
        self.stop = len(self.samples) if stop is None else stop
        self.features, self.pid = None, None

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, slot):
        slot += self.start
        if self.order is not None:
            slot = int(self.order[slot])
        if self.features is None or self.pid != os.getpid():
            self.features = np.load(self.cache / 'features.npy', mmap_mode='r')
            self.pid = os.getpid()
        row = self.samples[slot]
        # Column zero is the extra fourth context view reserved for CDiT.
        frame_ids = row[1:5]
        value = np.array(self.features[frame_ids], copy=True)
        return {'features': torch.from_numpy(value), 'actions': torch.tensor(row[6:9], dtype=torch.long),
                'dataset': int(row[9]), 'source': int(row[10]), 'collided': bool(row[11]),
                'blocked': bool(row[12]), 'slot': int(slot)}
