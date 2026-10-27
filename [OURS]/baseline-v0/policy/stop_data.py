"""Audited label-only views and reproducible STOP-focused minibatches.

Frame indices and executed-action histories always refer to the original rollout.
Sampling changes the training distribution, never the causal context of a sample.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .common import past_frame_indices


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StopSteps(Dataset):
    def __init__(self, roots, cfg, group="A"):
        self.cfg, self.episodes, self.provenance = cfg, [], {}
        pools, targets, distances = defaultdict(list), [], []
        self.offsets = [0]
        seen = set()
        partition = Path(__file__).resolve().parents[1] / "manifest/scene_partition.json"
        allowed = set(json.loads(partition.read_text())["groups"][group]["scenes"])
        for root in map(Path, roots):
            report = json.loads((root / "summary.json").read_text())
            settings = json.loads((root / "settings.json").read_text())
            assert report["status"] == "complete" and settings["group"] == group
            assert settings["stop_radius_m"] == 2.0
            for filename, expected in report["artifacts_sha256"].items():
                assert sha(root / filename) == expected, (root, filename)
            self.provenance[str(root.resolve())] = sha(root / "summary.json")
            root_targets = np.load(root / "targets.npy")
            root_distances = np.load(root / "goal_distances.npy")
            assert len(root_targets) == len(root_distances)
            base, expected_offset = self.offsets[-1], 0
            for line in (root / "episodes.jsonl").read_text().splitlines():
                entry = json.loads(line)
                path = Path(entry["meta"])
                assert path not in seen, path
                seen.add(path)
                assert entry["scene"] in allowed and entry["kind"] == settings["kind"]
                assert entry["offset"] == expected_offset
                assert sha(path) == entry["meta_sha256"], path
                meta = json.loads(path.read_text())
                assert str(meta["episode_id"]) == entry["episode_id"] and meta["scene"] == entry["scene"]
                assert len(meta["executed_action"]) == len(meta["expert_action"]) == entry["length"]
                assert all(a in (0, 1, 2, 3) for a in meta["executed_action"])
                if entry["kind"] == "policy":
                    collection = path.parents[2] / "collection_config.json"
                    signature = json.loads(collection.read_text())
                    assert signature["controller"] == "policy" and signature["group"] == "A" and signature["beta"] == 0
                    assert signature["recovery_rule"]["name"] == "goal_first_stop_2m_ordered_waypoints_v2"
                    assert sha(collection) == settings["collection_config_sha256"] == meta["collection_signature_sha256"]
                    assert len(meta["distances_to_goal"]) == entry["length"] + 1
                start = entry["offset"]
                for category in ("stop", "hard", "navigation"):
                    times = np.asarray(entry[category], dtype=np.int64)
                    assert len(times) == len(set(times.tolist()))
                    assert ((times >= 0) & (times < entry["length"])).all()
                    labels, d = root_targets[start + times], root_distances[start + times]
                    if category == "stop":
                        assert (labels == 0).all() and (d < 2).all()
                    else:
                        assert (labels > 0).all() and (labels <= 3).all() and (d >= 2).all()
                    pools[(entry["kind"], category, entry["dataset"])].extend((base + start + times).tolist())
                self.episodes.append({**entry, "dir": path.parent, "instruction": meta["instruction"]["text"],
                                      "executed": meta["executed_action"]})
                expected_offset += entry["length"]
                self.offsets.append(base + expected_offset)
            assert expected_offset == len(root_targets)
            targets.append(root_targets)
            distances.append(root_distances)
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        self.targets = np.concatenate(targets)
        self.distances = np.concatenate(distances)
        self.pools = {key: np.asarray(values, dtype=np.int64) for key, values in pools.items()}
        assert self.offsets[-1] == len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.offsets, index, side="right") - 1)
        t = int(index - self.offsets[episode])
        ep = self.episodes[episode]
        target = int(self.targets[index])
        assert target in (0, 1, 2, 3)
        images = []
        for frame in past_frame_indices(t, self.cfg) + [t]:
            with Image.open(ep["dir"] / "rgb" / f"{frame:04d}.jpg") as image:
                images.append(image.convert("RGB"))
        return {"instruction": ep["instruction"], "executed_actions": ep["executed"][:t], "t": t,
                "images": images, "target": target, "dataset": ep["dataset"]}

    def summary(self):
        return {"episodes": len(self.episodes), "original_states": len(self), "provenance": self.provenance,
                "pool_sizes": {"/".join(key): len(value) for key, value in self.pools.items()},
                "pool_actions": {"/".join(key): np.bincount(self.targets[value], minlength=4).tolist()
                                 for key, value in self.pools.items()}}


class StopSampler(Sampler):
    # The four odd category counts use opposite dataset allocations in each pair.
    # Each update has 32 R2R + 32 RxR; parity alternates across updates.
    CATEGORIES = (("expert", "stop", 5), ("policy", "stop", 1),
                  ("expert", "hard", 5), ("policy", "hard", 5),
                  ("expert", "navigation", 32), ("policy", "navigation", 16))
    BATCH = 64

    def __init__(self, pools, seed=3, start=0):
        self.pools, self.seed, self.start = pools, seed, start
        for kind, category, _ in self.CATEGORIES:
            for dataset in ("r2r", "rxr"):
                if not len(pools.get((kind, category, dataset), [])):
                    raise ValueError(f"Empty required sampling pool: {kind}/{category}/{dataset}")

    def block(self, block):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, block]))
        picks, odd = [], 0
        for kind, category, size in self.CATEGORIES:
            r2r = size // 2
            if size % 2:
                r2r += (block + odd) % 2
                odd += 1
            for dataset, count in (("r2r", r2r), ("rxr", size - r2r)):
                pool = self.pools[(kind, category, dataset)]
                picks.extend(pool[rng.integers(len(pool), size=count)].tolist())
        assert len(picks) == self.BATCH
        rng.shuffle(picks)
        return picks

    def __iter__(self):
        block, offset = divmod(self.start, self.BATCH)
        while True:
            yield from self.block(block)[offset:]
            block, offset = block + 1, 0

    def __len__(self):
        return 2 ** 62


def validation_picks(data, count=2000):
    """Fixed balanced probe; a learning diagnostic, not a success-rate estimate."""
    picks = []
    for dataset in ("r2r", "rxr"):
        for category, quota in (("stop", count // 4), ("hard", count // 8), ("navigation", count // 8)):
            pool = data.pools[("expert", category, dataset)]
            picks.extend(pool[np.linspace(0, len(pool) - 1, min(quota, len(pool))).astype(int)].tolist())
    return list(dict.fromkeys(picks))
