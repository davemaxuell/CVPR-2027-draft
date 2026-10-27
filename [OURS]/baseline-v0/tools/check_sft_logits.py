"""Compare selective action-token logits/loss/gradients against full-vocabulary output."""
import contextlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.data import ExpertSteps
from policy.smolvlm2 import SmolVLM2Inputs, SmolVLM2Policy, default_config


def check(dtype):
    cfg = default_config()
    inputs = SmolVLM2Inputs(cfg)
    data = ExpertSteps(["/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupC_600"],
                       cfg, max_episodes=3)
    steps = [data[i] for i in [0, 15, 47]]
    batch = {k: v.cuda() for k, v in inputs(steps).items()}
    policy = SmolVLM2Policy(cfg, lora_dropout=0.0, dtype=dtype)
    policy.model.eval()
    labels = batch["labels"]
    mask = labels[:, 1:] != -100
    context = (patch("policy.smolvlm2.torch.autocast", lambda *a, **k: contextlib.nullcontext())
               if dtype == torch.float32 else contextlib.nullcontext())
    with context:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = policy.model(**policy.model_inputs(batch), use_cache=False).logits
        reference = torch.nn.functional.cross_entropy(out[:, :-1][mask].float(), labels[:, 1:][mask], reduction="sum")
        reference.backward()
        params = policy.trainable_parameters()
        grads = [p.grad.detach().clone() for p in params]
        policy.model.zero_grad(set_to_none=True)
        reference_loss = float(reference.detach())
        ends = batch["lengths"] - (labels != -100).sum(1) - 1
        ref_scores = out[torch.arange(3, device="cuda"), ends][:, policy.first_token].detach().float()
        del out, reference
        loss, tokens = policy.loss_sum(batch)
        loss.backward()
        grad_relative = (sum(float((p.grad-g).square().sum()) for p, g in zip(params, grads)) ** .5 /
                         sum(float(g.square().sum()) for g in grads) ** .5)
        eval_loss, n, scores = policy.evaluate(batch)
        score_error = float((scores-ref_scores).abs().max())
        loss_error = abs(float(loss.detach())-reference_loss)
        # bf16 GEMM shape changes affect backward reduction rounding. fp32 proves
        # the underlying loss and gradient are equivalent without that rounding.
        assert loss_error < (1e-4 if dtype == torch.bfloat16 else 2e-4), loss_error
        assert score_error < (1e-4 if dtype == torch.bfloat16 else 2e-4), score_error
        assert grad_relative < (.03 if dtype == torch.bfloat16 else 2e-4), grad_relative
        assert tokens == n and abs(float(eval_loss)-float(loss.detach())) < 1e-5
    return {"dtype": str(dtype), "reference_loss": reference_loss, "selective_loss": float(loss.detach()),
            "target_tokens": tokens, "max_score_error": score_error, "relative_gradient_error": grad_relative}


if __name__ == "__main__":
    torch.manual_seed(0)
    report = [check(torch.bfloat16), check(torch.float32)]
    print(json.dumps(report, indent=2))
    (ROOT / "runs/step2_smolvlm2/check_selective_logits.json").write_text(json.dumps(report, indent=2))
