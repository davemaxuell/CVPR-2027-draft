"""DAgger samples retain failed rollouts and use actual actions for prompt history."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .common import past_frame_indices


class DaggerSteps(Dataset):
    def __init__(self, roots, cfg, allowed_scenes):
        self.cfg, self.episodes = cfg, []
        self.invalid_labels = 0
        self.metadata_hashes = {}
        import hashlib
        for root in map(Path, roots):
            config_bytes = (root / "collection_config.json").read_bytes()
            config = json.loads(config_bytes)
            config_hash = hashlib.sha256(config_bytes).hexdigest()
            if config["group"] != "A" or config["controller"] != "policy" or config["beta"] != 0.0:
                raise ValueError("DAgger training requires group-A policy rollouts")
            summary = json.loads((root / "summary.json").read_text())
            if summary["status"] != "complete":
                raise ValueError("DAgger collection is incomplete")
            for path in sorted(root.glob("*/*/meta.json")):
                if path.parent.name.startswith("."):
                    continue
                content = path.read_bytes()
                m = json.loads(content)
                if m["group"] != "A" or m["scene"] not in allowed_scenes or m["data_kind"] != "dagger":
                    raise ValueError(f"Invalid DAgger provenance: {path}")
                if m["collection_signature_sha256"] != config_hash:
                    raise ValueError(f"DAgger episode configuration mismatch: {path}")
                executed, labels = m["executed_action"], m["expert_action"]
                if len(executed) != len(labels) or len(labels) != m["num_actions"]:
                    raise ValueError(f"Misaligned actions/labels: {path}")
                if any(a not in (0, 1, 2, 3) for a in executed):
                    raise ValueError(f"Invalid executed action: {path}")
                if any(a not in (None, 0, 1, 2, 3) for a in labels):
                    raise ValueError(f"Invalid expert label: {path}")
                valid = [i for i, a in enumerate(labels) if a is not None]
                self.invalid_labels += len(labels) - len(valid)
                self.metadata_hashes[str(path)] = hashlib.sha256(content).hexdigest()
                # Success and timeout are deliberately not filters: failures are the point of DAgger.
                if valid:
                    self.episodes.append({"dir": path.parent, "dataset": m["dataset"],
                                          "instruction": m["instruction"]["text"],
                                          "executed": executed, "expert": labels, "valid": valid})
        self.offsets = np.concatenate([[0], np.cumsum([len(e["valid"]) for e in self.episodes])]).astype(np.int64)

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        e = int(np.searchsorted(self.offsets, index, side="right") - 1)
        ep = self.episodes[e]
        t = ep["valid"][int(index - self.offsets[e])]
        images = []
        for frame in past_frame_indices(t, self.cfg) + [t]:
            with Image.open(ep["dir"] / "rgb" / f"{frame:04d}.jpg") as image:
                images.append(image.convert("RGB"))
        return {"instruction": ep["instruction"], "executed_actions": ep["executed"][:t], "t": t,
                "images": images, "target": ep["expert"][t], "dataset": ep["dataset"]}

    def action_counts(self):
        return np.bincount([ep["expert"][t] for ep in self.episodes for t in ep["valid"]], minlength=4)


class BalancedSourceSampler(Sampler):
    """Exactly half expert / half DAgger per optimizer update; reproducible on resume.

    Each source is sampled uniformly with replacement. DAgger data is much smaller
    than the original expert corpus; equal source weight prevents it being diluted.
    """
    def __init__(self, expert_size, dagger_size, batch_size, seed, start=0):
        if min(expert_size, dagger_size) <= 0 or batch_size % 2:
            raise ValueError("Both sources must be nonempty and the effective batch must be even")
        self.expert_size, self.dagger_size = expert_size, dagger_size
        self.batch_size, self.seed, self.start = batch_size, seed, start

    def __iter__(self):
        block, offset = divmod(self.start, self.batch_size)
        half = self.batch_size // 2
        while True:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, block]))
            picks = np.concatenate([rng.integers(self.expert_size, size=half),
                                    self.expert_size + rng.integers(self.dagger_size, size=half)])
            rng.shuffle(picks)
            yield from picks[offset:].tolist()
            block, offset = block + 1, 0

    def __len__(self):
        return 2 ** 62
