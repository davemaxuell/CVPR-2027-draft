"""Per-step SFT samples from rendered trajectories (tools/render_expert.py layout).

Each sample is one (episode, t): the prompt uses executed actions before t and the frames chosen
by the policy window up to t; the target is expert_action[t]. Episodes whose expert replay did
not succeed, was truncated, or drifted at least `max_deviation` metres from the stored expert path
(RxR-CE only: those average ~27 wall collisions) are excluded and listed in `excluded`.
"""
import glob
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .common import PolicyConfig, past_frame_indices


class ExpertSteps(Dataset):
    def __init__(self, roots, cfg: PolicyConfig, datasets=("r2r", "rxr"), max_episodes=0, max_deviation=0.5):
        self.cfg = cfg
        self.episodes, self.excluded = [], {}
        for root in roots:
            for name in datasets:
                metas = sorted(glob.glob(os.path.join(root, name, "*", "meta.json")))
                if max_episodes:
                    metas = metas[:max_episodes]
                skipped = {"failed_or_truncated": [], "drift": []}
                for path in metas:
                    with open(path) as f:
                        m = json.load(f)
                    if m["metrics"]["success"] < 1 or m["truncated"]:
                        skipped["failed_or_truncated"].append(m["episode_id"])
                        continue
                    if m["max_location_deviation_m"] >= max_deviation:
                        skipped["drift"].append(m["episode_id"])
                        continue
                    self.episodes.append({"dir": os.path.dirname(path), "dataset": name,
                                          "instruction": m["instruction"]["text"],
                                          "executed": m["executed_action"], "expert": m["expert_action"]})
                self.excluded[f"{os.path.basename(os.path.normpath(root))}/{name}"] = skipped
        lengths = np.array([len(e["expert"]) for e in self.episodes], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        e = int(np.searchsorted(self.offsets, index, side="right") - 1)
        t = int(index - self.offsets[e])
        ep = self.episodes[e]
        frames = past_frame_indices(t, self.cfg) + [t]
        images = [Image.open(os.path.join(ep["dir"], "rgb", f"{i:04d}.jpg")).convert("RGB") for i in frames]
        return {"instruction": ep["instruction"], "executed_actions": ep["executed"][:t], "t": t,
                "images": images, "target": ep["expert"][t], "dataset": ep["dataset"]}

    def action_counts(self):
        counts = np.zeros(4, dtype=np.int64)
        for ep in self.episodes:
            counts += np.bincount(ep["expert"], minlength=4)
        return counts
