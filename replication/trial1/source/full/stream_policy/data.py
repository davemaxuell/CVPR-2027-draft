"""Training windows from rendered trajectories: expert renders and StreamVLN DAgger rounds.

Root layout: <root>/<dataset>/<episode_id>/{meta.json, rgb/0000.jpg, ...}; frame t is the view before
action t. Expert renders (tools/render_expert.py) are used when their replay succeeded, was not
truncated and stayed within `max_deviation` metres of the stored path, as in trial_0. DAgger rounds
(tools/collect_stream_dagger.py) contribute their kept episodes. One sample is one dialogue window
(layout.windows), as in StreamVLN's VLNActionDataset.

As in StreamVLN, trajectories with fewer than 4 actions are skipped (3 two-action RxR episodes in group A).
Targets are the executed actions (StreamVLN), with a kept DAgger episode's final frame labelled STOP.
label_source="expert" instead supervises the expert's label at every step while the prompt keeps the
executed actions; steps without a valid label are left unsupervised.
"""
import glob
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .layout import fixed_conjunction, layout, windows

STOP = 0
MIN_ACTIONS = 4  # VLNActionDataset @ e48f6ff7: `if actions_len < 4: continue`


def streamvln_augmentation():
    """StreamVLN's data_augmentation transforms (streamvln_train.py @ e48f6ff7), applied per image."""
    from torchvision.transforms import v2
    return v2.Compose([v2.ColorJitter(brightness=0.2, saturation=0.2), v2.RandomPosterize(bits=4),
                       v2.RandomAdjustSharpness(sharpness_factor=1.5), v2.RandomAutocontrast()])


class TrajectoryWindows(Dataset):
    def __init__(self, roots, cfg, inputs, datasets=("r2r", "rxr"), label_source="executed", max_deviation=0.5,
                 max_episodes=0, augment=False, fixed_conjunctions=False):
        if label_source not in ("executed", "expert"):
            raise ValueError("label_source must be 'executed' or 'expert'")
        self.cfg, self.inputs, self.label_source = cfg, inputs, label_source
        self.fixed_conjunctions = fixed_conjunctions  # validation: the closed-loop choice instead of random
        self.augment = streamvln_augmentation() if augment else None
        self.episodes, self.index, self.excluded = [], [], {}
        for root in roots:
            for name in datasets:
                paths = sorted(glob.glob(os.path.join(root, name, "*", "meta.json")))
                if max_episodes:
                    paths = paths[:max_episodes]
                skipped = {"failed_or_truncated": [], "drift": [], "not_kept": [], "shorter_than_4": []}
                for path in paths:
                    with open(path) as f:
                        m = json.load(f)
                    episode = self._episode(m, os.path.dirname(path), skipped, max_deviation)
                    if episode is None:
                        continue
                    self.episodes.append(episode)
                    e = len(self.episodes) - 1
                    self.index += [(e, start, stop) for start, stop in windows(len(episode["executed"]), cfg)]
                self.excluded[f"{os.path.basename(os.path.normpath(root))}/{name}"] = skipped

    def _episode(self, m, folder, skipped, max_deviation):
        kind = m.get("data_kind", "expert_render")
        executed = list(m["executed_action"])
        if kind == "expert_render":
            if m["metrics"]["success"] < 1 or m["truncated"]:
                skipped["failed_or_truncated"].append(m["episode_id"])
                return None
            if m["max_location_deviation_m"] >= max_deviation:
                skipped["drift"].append(m["episode_id"])
                return None
            if len(executed) != m["num_frames"] or executed[-1] != STOP:
                raise ValueError(f"expert render is not frame-aligned and STOP-terminated: {folder}")
            targets = list(m["expert_action"]) if self.label_source == "expert" else list(executed)
        elif kind == "streamvln_dagger":
            if not m["keep"]:
                skipped["not_kept"].append(m["episode_id"])
                return None
            if m["num_frames"] != len(executed):
                raise ValueError(f"DAgger episode frames and actions differ: {folder}")
            if self.label_source == "expert":
                targets = list(m["expert_action"])
            else:
                targets = executed[:-1] + [STOP]
        else:
            raise ValueError(f"unsupported data_kind {kind!r}: {folder}")
        if len(executed) < MIN_ACTIONS:
            skipped["shorter_than_4"].append(m["episode_id"])
            return None
        return {"dir": folder, "dataset": m["dataset"], "episode_id": str(m["episode_id"]), "kind": kind,
                "instruction": m["instruction"]["text"], "executed": executed, "targets": targets}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, start, stop = self.index[i]
        ep = self.episodes[e]
        conjunction = (fixed_conjunction(self.cfg, start) if self.fixed_conjunctions
                       else lambda turn: random.choice(self.cfg.conjunctions))
        pieces = layout(self.cfg, ep["instruction"], start, stop, conjunction)
        sample = self.inputs.encode(pieces, ep["executed"], ep["targets"])
        images = []
        for frame, kind in sample["images"]:
            with Image.open(os.path.join(ep["dir"], "rgb", f"{frame:04d}.jpg")) as image:
                image = self.inputs.resize(image.convert("RGB"), kind)
            images.append(self.augment(image) if self.augment else image)
        sample["pixels"] = images
        sample["key"] = (ep["dataset"], ep["episode_id"], start)
        return sample

    def action_counts(self):
        counts = np.zeros(4, dtype=np.int64)
        for ep in self.episodes:
            counts += np.bincount([a for a in ep["targets"] if a is not None], minlength=4)
        return counts

    def summary(self):
        by_kind = {}
        for ep in self.episodes:
            row = by_kind.setdefault(f"{ep['kind']}/{ep['dataset']}", {"episodes": 0, "steps": 0})
            row["episodes"] += 1
            row["steps"] += len(ep["executed"])
        return {"windows": len(self), "episodes": len(self.episodes), "by_source": by_kind,
                "target_action_counts": dict(zip(["STOP", "FORWARD", "LEFT", "RIGHT"], self.action_counts().tolist())),
                "excluded": self.excluded}


class Sources(Dataset):
    """Windows of several roots, read by (draw, index) keys from MixtureSampler.

    Each sample's augmentation and conjunctions are seeded from (seed, draw), its position in the
    training stream, instead of from worker state, so the stream depends on the seed alone and a
    resumed run continues it exactly.
    """

    def __init__(self, sources, seed):
        self.sources, self.seed = sources, seed
        self.offsets = np.concatenate([[0], np.cumsum([len(s) for s in sources])]).astype(np.int64)

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, key):
        draw, index = key
        state = int(np.random.SeedSequence([self.seed, draw]).generate_state(1)[0])
        random.seed(state)
        torch.manual_seed(state)
        s = int(np.searchsorted(self.offsets, index, side="right") - 1)
        sample = self.sources[s][index - int(self.offsets[s])]
        sample["draw"] = draw
        return sample


class MixtureSampler(Sampler):
    """Reproducible, resumable (draw, index) stream over the concatenation of `sizes` sources.

    Without weights: a fresh seeded permutation of every window each pass (StreamVLN's shuffled
    concatenation). With weights: each draw picks a source by weight, then a window uniformly.
    `draw` counts positions in the stream; `start` skips that many, so a resumed run continues it.
    """

    def __init__(self, sizes, seed, weights=None, start=0, block=4096):
        if not sizes or min(sizes) <= 0:
            raise ValueError("every source must be nonempty")
        if weights is not None and (len(weights) != len(sizes) or min(weights) < 0 or sum(weights) <= 0):
            raise ValueError("weights must be nonnegative, one per source")
        self.sizes, self.seed, self.start, self.block = list(sizes), seed, start, block
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64) / sum(weights)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def __iter__(self):
        if self.weights is None:
            total = int(self.offsets[-1])
            epoch, offset = divmod(self.start, total)
            while True:
                picks = np.random.default_rng([self.seed, epoch]).permutation(total)
                yield from ((epoch * total + p, int(i)) for p, i in enumerate(picks[offset:], offset))
                epoch, offset = epoch + 1, 0
        block, offset = divmod(self.start, self.block)
        while True:
            rng = np.random.default_rng([self.seed, block])
            source = rng.choice(len(self.sizes), size=self.block, p=self.weights)
            picks = self.offsets[source] + (rng.random(self.block) * np.asarray(self.sizes)[source]).astype(np.int64)
            yield from ((block * self.block + p, int(i)) for p, i in enumerate(picks[offset:], offset))
            block, offset = block + 1, 0

    def __len__(self):
        return 2 ** 62
