"""Versioned experimental policy: saved preprocessing and mandatory STOP bias."""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from .common import ACTION_WORDS, PolicyConfig
from .smolvlm2 import END_OF_TURN, SmolVLM2Inputs, SmolVLM2Policy


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    metadata = json.loads((bundle / "reference.json").read_text())
    if metadata["format"] != "smolvlm2_experimental_reference_v1":
        raise ValueError("Unsupported experimental reference format")
    for relative, expected in json.loads((bundle / "file_sha256.json").read_text()).items():
        assert sha256(bundle / relative) == expected, ("Bundle drift", relative)
    root = Path(__file__).resolve().parents[1]
    for relative, expected in metadata["runtime_source_sha256"].items():
        assert sha256(root / relative) == expected, ("Runtime source drift", relative)
    identity = metadata["identity_fields"]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == metadata["policy_identity_sha256"]
    assert metadata["decoder"] == identity["decoder"]
    assert metadata["decoder"]["action_order"] == ACTION_WORDS
    assert metadata["decoder"]["stop_logit_bias"] == -0.25
    assert metadata["selection"]["authority"] == "explicit_user_selection_2026-10-26"
    assert not metadata["selection"]["original_stopping_safeguard_passed"]
    assert sha256(bundle / "adapter/adapter_model.safetensors") == identity["adapter_sha256"]
    assert sha256(bundle / "policy_config.json") == identity["policy_config_sha256"]
    for name, expected in identity["processor_sha256"].items():
        assert sha256(bundle / "processor" / name) == expected
    return metadata


class ReferenceInputs(SmolVLM2Inputs):
    """Use the archived processor with the native evaluation prompt builder."""
    def __init__(self, bundle):
        bundle = Path(bundle)
        self.cfg = PolicyConfig.load(bundle / "policy_config.json")
        self.processor = AutoProcessor.from_pretrained(str(bundle / "processor"), local_files_only=True)
        self.processor.image_processor.do_image_splitting = self.cfg.image_splitting
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        self.targets = [f" {word}{END_OF_TURN}" for word in ACTION_WORDS]
        self.target_ids = [self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in self.targets]
        self.first_token = [ids[0] for ids in self.target_ids]
        assert len(set(self.first_token)) == 4


class ExperimentalReference(SmolVLM2Policy):
    """action_scores always includes the locked STOP correction, exactly once."""
    def __init__(self, bundle, device="cuda:0"):
        self.bundle = Path(bundle).resolve()
        self.reference_identity = verify_bundle(self.bundle)
        self.identity_sha256 = self.reference_identity["policy_identity_sha256"]
        self.stop_bias = self.reference_identity["decoder"]["stop_logit_bias"]
        cfg = PolicyConfig.load(self.bundle / "policy_config.json")
        super().__init__(cfg, adapter=str(self.bundle / "adapter"), device=device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.inputs = ReferenceInputs(self.bundle)
        assert self.first_token.tolist() == self.inputs.first_token

    @torch.no_grad()
    def raw_action_scores(self, batch, return_context=False):
        return super().action_scores(batch, return_context=return_context)

    def action_scores(self, batch, return_context=False):
        result = self.raw_action_scores(batch, return_context=return_context)
        raw = result[0] if return_context else result
        scores = raw.clone()
        scores[:, 0] += self.stop_bias
        return (scores, result[1]) if return_context else scores
