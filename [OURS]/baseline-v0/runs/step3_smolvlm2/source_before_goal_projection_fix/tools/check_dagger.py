"""Check leakage-sensitive DAgger data/history, recovery progress and sampler resume."""
import itertools
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
from PIL import Image

from policy.dagger_data import BalancedSourceSampler, DaggerSteps
from policy.smolvlm2 import default_config
from sim.recovery import RouteExpert


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "collection_config.json").write_text(json.dumps({"group": "A", "controller": "policy", "beta": 0.0}))
        (root / "summary.json").write_text(json.dumps({"status": "complete"}))
        episode = root / "r2r" / "1"
        (episode / "rgb").mkdir(parents=True)
        for t in range(5):
            Image.new("RGB", (8, 8), (t * 40, 0, 0)).save(episode / "rgb" / f"{t:04d}.jpg")
        (episode / "meta.json").write_text(json.dumps({"group": "A", "data_kind": "dagger", "scene": "sceneA",
            "dataset": "r2r", "instruction": {"text": "Follow the corridor"}, "num_actions": 4,
            "expert_action": [2, None, 3, 0], "executed_action": [1, 2, 1, 3],
            "collection_signature_sha256": hashlib.sha256((root / "collection_config.json").read_bytes()).hexdigest(),
            "metrics": {"success": 0}, "termination": "step_limit"}))
        data = DaggerSteps([root], default_config(), {"sceneA"})
        assert len(data) == 3 and data.invalid_labels == 1 and len(data.episodes) == 1
        sample = data[1]
        assert sample["t"] == 2 and sample["target"] == 3
        assert sample["executed_actions"] == [1, 2] and len(sample["images"]) == 3
        assert np.asarray(sample["images"][-1])[0, 0, 0] > 70
        assert data[2]["target"] == 0 and data[2]["executed_actions"] == [1, 2, 1]
        try:
            DaggerSteps([root], default_config(), {"sceneC"})
        except ValueError:
            pass
        else:
            raise AssertionError("Cross-group DAgger data accepted")
    full = list(itertools.islice(iter(BalancedSourceSampler(103, 17, 64, 1)), 256))
    for start in range(0, 256, 64):
        assert sum(i < 103 for i in full[start:start + 64]) == 32
    resumed = list(itertools.islice(iter(BalancedSourceSampler(103, 17, 64, 1, start=91)), 165))
    assert resumed == full[91:]

    class FakeFollower:
        def __init__(self, *args, **kwargs):
            self.action, self.resets = 1, 0
            self._follower = self
        def _build_follower(self):
            pass
        def reset(self):
            self.resets += 1
        def get_next_action(self, target):
            return self.action

    position = np.array([0., 0., 0.])
    sim = SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda p, island_index: np.array(p), get_island=lambda p: 0),
        get_agent_state=lambda: SimpleNamespace(position=position),
        geodesic_distance=lambda a, b: np.linalg.norm(np.array(a) - np.array(b)))
    episode = SimpleNamespace(reference_path=[[0, 0, 0], [4, 0, 0], [0, 0, 0]],
                              goals=[SimpleNamespace(position=[0, 0, 0])])
    with patch("sim.recovery.ShortestPathFollower", FakeFollower):
        expert = RouteExpert(sim, episode)
        action, detail = expert.query()
        assert action == 1 and detail["route_index"] == 1, "Must follow the route even when initially near its final goal"
        expert.followers[False].action = 0
        action, detail = expert.query()
        assert action is None and detail["error"] == "unexpected_follower_stop"
        expert.followers[False].action = 1
        position[:] = [4, 0, 0]
        action, detail = expert.query()
        assert action == 1 and detail["route_index"] == 2
        position[:] = [0, 0, 0]
        action, detail = expert.query()
        assert action == 0 and detail["route_index"] == 2
        assert expert.followers[False].resets == 2
    print(json.dumps({"status": "passed", "checks": ["failed rollouts retained", "invalid labels skipped without renumbering time",
          "executed-action history is separate from expert target", "group-A enforcement", "exact balanced sampling and resume",
          "ordered reference progress without nearest-point jumps", "no silent STOP on follower error"]}, indent=2))


if __name__ == "__main__":
    main()
