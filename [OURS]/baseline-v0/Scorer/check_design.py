"""Synthetic CPU checks of the reference design; not CE experiments."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch

from labels import utility_target
from scorer import FORWARD, START, STOP, NavigationScorer, pool_dino_patches, scorer_loss, select_action


def main():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    checks, counts = [], {}
    batch, length, candidates = 2, 7, 4
    actions = torch.arange(candidates).expand(batch, -1)
    valid = torch.tensor([[True] * length, [True] * 4 + [False] * 3])
    current = torch.randn(batch, 49, 384, requires_grad=True)
    future = torch.randn(batch, candidates, 49, 384, requires_grad=True)
    previous = torch.full((batch,), START, dtype=torch.long)
    elapsed = torch.zeros(batch, 1)

    for policy_width in (960, 896, 2048):
        model = NavigationScorer(policy_width)
        text = torch.randn(batch, length, policy_width, requires_grad=True)
        context_feature = torch.randn(batch, policy_width, requires_grad=True)
        context = model.observe(text, valid, current, context_feature, previous, elapsed)
        outputs = model.score(context, future, actions)
        assert all(value.shape == (batch, candidates) and torch.isfinite(value).all() for value in outputs.values())
        target = torch.tensor([[0.8, 0.15, 0.2, 0.1], [0.1, 0.9, 0.4, 0.05]])
        success = (target > 0.6).float()
        collisions = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.]])
        losses = scorer_loss(outputs, target, success, collisions, actions)
        losses["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
        for external in (text, current, context_feature, future):
            assert external.grad is None, "A frozen input received a gradient"
        for module in (model.policy_projection, model.visual_projection, model.memory_gru, model.blocks, model.utility_head):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count < 5_000_000
        counts[str(policy_width)] = count
        checks.append(f"width_{policy_width}: shapes, finite trainable gradients, frozen feature inputs, fewer than 5M parameters")

    model.eval()
    with torch.no_grad():
        context = model.observe(text, valid, current, context_feature, previous, elapsed)
        before_memory, before_tokens = context.memory.clone(), context.tokens.clone()
        expected = model.score(context, future, actions)
        assert torch.equal(context.memory, before_memory) and torch.equal(context.tokens, before_tokens)
        checks.append("candidate scoring leaves actual memory and context unchanged")

        permutation = torch.tensor([3, 1, 0, 2])
        permuted = model.score(context, future[:, permutation], actions[:, permutation])
        for key in expected:
            torch.testing.assert_close(permuted[key], expected[key][:, permutation], rtol=1e-5, atol=1e-6)
        checks.append("candidate order only permutes scores")

        changed = future.detach().clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 10
        independent = model.score(context, changed, actions)
        for key in expected:
            torch.testing.assert_close(independent[key][:, [0, 2, 3]], expected[key][:, [0, 2, 3]])
        checks.append("changing one candidate cannot change other candidate scores")

        changed[:, STOP] = 12345
        stop = model.score(context, changed, actions)
        for key in expected:
            torch.testing.assert_close(stop[key][:, STOP], expected[key][:, STOP])
        checks.append("STOP always uses the current view, ignoring a supplied future")

        changed_text = text.detach().clone()
        changed_text[~valid] = torch.randn_like(changed_text[~valid]) * 100
        padded_context = model.observe(changed_text, valid, current, context_feature, previous, elapsed)
        padded_scores = model.score(padded_context, future, actions)
        torch.testing.assert_close(context.memory, padded_context.memory)
        for key in expected:
            torch.testing.assert_close(expected[key], padded_scores[key])
        extra_text = torch.cat((text, torch.randn(batch, 3, model.policy_width)), dim=1)
        extra_valid = torch.cat((valid, torch.zeros(batch, 3, dtype=torch.bool)), dim=1)
        extra = model.observe(extra_text, extra_valid, current, context_feature, previous, elapsed)
        torch.testing.assert_close(context.memory, extra.memory, rtol=1e-5, atol=1e-6)
        extra_scores = model.score(extra, future, actions)
        for key in expected:
            torch.testing.assert_close(expected[key], extra_scores[key], rtol=1e-5, atol=1e-6)
        checks.append("masked text content and additional padding do not change real memory or scores")

        other_view = current.detach().flip(1)
        first_b = model.observe(text, valid, other_view, context_feature, previous, elapsed)
        after_ab = model.observe(text, valid, other_view, context_feature, previous, elapsed, context.memory)
        after_ba = model.observe(text, valid, current, context_feature, previous, elapsed, first_b.memory)
        assert (after_ab.memory - after_ba.memory).abs().max() > 1e-6
        checks.append("real memory is order sensitive (structural check, not semantic competence)")

    old_memory = torch.randn(batch, 8, 256, requires_grad=True)
    next_context = model.observe(text, valid, current, context_feature, previous, elapsed, old_memory)
    model.score(next_context, future, actions)["utility"].sum().backward()
    assert old_memory.grad is not None and old_memory.grad.abs().sum() > 0
    checks.append("gradient flows through real memory until the caller truncates BPTT")

    values = torch.tensor([[0.3, 0.31, 0.2, 0.1], [0.3, 0.8, 0.2, 0.1]])
    selected = select_action(values, actions, torch.tensor([0, 0]), margin=0.05)
    assert selected.tolist() == [0, 1]
    tied = select_action(torch.ones_like(values), actions, torch.tensor([2, 3]), margin=0)
    assert tied.tolist() == [2, 3]
    other_ties = select_action(torch.tensor([[0.1, 0.8, 0.8, 0.1]]),
                              torch.tensor([[3, 2, 1, 0]]), torch.tensor([0]), margin=0)
    assert other_ties.tolist() == [1], "Other ties must use canonical action order"
    checks.append("override margin and ties preserve the base action as specified")

    constant = pool_dino_patches(torch.ones(2, 196, 384))
    assert constant.shape == (2, 49, 384) and torch.equal(constant, torch.ones_like(constant))
    checks.append("DINO patch pooling preserves shape and constant inputs")

    failed, completed = [], []
    for success in (0, 1):
        for ndtw in (0.0, 0.2, 0.7, 1.0):
            for steps in (1, 100, 500):
                for collisions in (0, steps):
                    value = utility_target(success, ndtw, steps, collisions, 500)
                    assert -0.0300001 <= value <= 1.0000001
                    (completed if success else failed).append(value)
    assert min(completed) > max(failed)
    try:
        utility_target(1, 0.8, 501, 0, 500)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid action count accepted")
    checks.append("utility range, realized success preference, and invalid-count rejection")

    directory = Path(__file__).resolve().parent
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
        "device_used": "cpu", "cuda_available": torch.cuda.is_available(), "seed": 23,
        "trainable_parameter_counts_by_policy_hidden_width": counts,
        "checks_passed": checks, "all_passed": True,
        "scope": "Synthetic tensor, gradient, memory, action-selection, and label checks only. No model weights loaded, CE rollout, navigation training, or GPU memory measurement.",
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ("scorer.py", "labels.py", "interfaces.py", "runtime.py", "world_models.py", "config.json", "check_design.py")},
    }
    (directory / "check_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
