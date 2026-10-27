"""Load an immutable STOP-study reference, including its selected decoder bias."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import PolicyConfig
from .smolvlm2 import SmolVLM2Policy


class StopReference(SmolVLM2Policy):
    def __init__(self, bundle, device="cuda"):
        bundle = Path(bundle).resolve()
        metadata = json.loads((bundle / "reference.json").read_text())
        assert metadata["format"] == "smolvlm2_stop_reference_v1"
        assert metadata["confirmation_passed"]
        for relative, expected in json.loads((bundle / "file_sha256.json").read_text()).items():
            assert hashlib.sha256((bundle / relative).read_bytes()).hexdigest() == expected, relative
        root = Path(__file__).resolve().parents[1]
        for relative, expected in metadata["runtime_source_sha256"].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, f"Runtime source drift: {relative}"
        self.stop_bias = float(metadata["decoder"]["stop_logit_bias"])
        assert np.isfinite(self.stop_bias)
        self.reference_identity = metadata
        super().__init__(PolicyConfig.load(bundle / "policy_config.json"), adapter=str(bundle / "adapter"), device=device)
        self.model.requires_grad_(False)
        self.model.eval()

    def action_scores(self, batch, return_context=False):
        result = super().action_scores(batch, return_context=return_context)
        raw = result[0] if return_context else result
        scores = raw.clone()
        scores[:, 0] += self.stop_bias
        return (scores, result[1]) if return_context else scores
