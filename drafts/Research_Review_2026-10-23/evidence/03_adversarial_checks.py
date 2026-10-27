"""Independent bounded A3 checks. No CE training, weights, or source mutation."""
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parents[1] / "Research Approach" / "Scorer"
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in ORIGINAL.iterdir() if p.is_file()}

def load_original(name):
    spec = importlib.util.spec_from_file_location(name, ORIGINAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

labels = load_original("labels")

# The corrected example respects the original utility support, with common
# action cost explicitly incorporated. These are abstract attainable labels,
# not simulated CE trajectories.
B, n = 500, 2
case = [(0.8, 1), (0.9, 1), (0.1, 0)]
support = []
for value, success in case:
    ndtw = (value - 0.7 * success + 0.01 * n / B) / 0.3
    measured = labels.utility_target(success, ndtw, n, 0, B)
    assert 0 <= ndtw <= 1 and abs(measured - value) < 1e-12
    support.append(dict(utility=value, success=success, ndtw=ndtw,
                        remaining_actions=n, max_episode_actions=B))
safe, risky = 0.8, [0.1, 0.9]
q_risky = sum(risky) / 2
sample_selector = sum(q_risky if z > safe else safe for z in risky) / 2
assert abs(sample_selector - 0.65) < 1e-12

# Measured max can invent headroom under stochastic outcomes. Three movement
# actions; STOP can be inferior and deterministic. No training is involved.
oracle_mean = sum(max(z) for z in itertools.product(risky, repeat=3)) / 8
assert abs(oracle_mean - 0.8) < 1e-12
common_seed_max = sum(max(z, 1.0-z) for z in risky) / 2
assert abs(common_seed_max - 0.9) < 1e-12

# Aliasing: equally likely latent roots reverse the correct movement action.
alias_Q = [[0.9, 0.1], [0.1, 0.9]]
bayes_Q = [sum(row[a] for row in alias_Q) / 2 for a in range(2)]
bayes_mse = sum((v - 0.5) ** 2 for row in alias_Q for v in row) / 4
alias_regret = 0.9 - max(bayes_Q)
assert abs(bayes_mse - 0.16) < 1e-12 and abs(alias_regret - 0.4) < 1e-12

# Proper mean-gap loss versus per-sample gap loss: full +/-1 stochastic model
# matches the target distribution, yet squared independent sample matching
# prefers deterministic zero in a relaxed scalar model class.
target = [-1.0, 1.0]
native_sample_loss = sum((d-y)**2 for d in target for y in target) / 4
collapsed_sample_loss = sum(y*y for y in target) / 2
cross_product = sum((d1-y)*(d2-y) for d1 in target for d2 in target for y in target) / 8
assert native_sample_loss == 2 and collapsed_sample_loss == 1 and cross_product == 1

# Exhaustive score-margin check for four candidates, including cases in which
# the base is retained. This checks both safety and delta+2epsilon regret.
eps, margin = 0.05, 0.1
margin_checks, overrides = 0, 0
for Q in itertools.product([0.1, 0.5, 0.9], repeat=4):
    for errors in itertools.product([-eps, eps], repeat=4):
        pred = [q+e for q,e in zip(Q,errors)]
        best = max(range(4), key=pred.__getitem__)
        selected = best if pred[best] - pred[0] > margin else 0
        assert Q[selected] >= Q[0] - 1e-12
        assert max(Q) - Q[selected] <= margin + 2 * eps + 1e-12
        margin_checks += 1
        overrides += int(selected != 0)

# Independent stochastic history-tree check with four actions, stochastic base,
# arbitrary path-terminal rewards, and exact Q_ref. STOP is action 3 and exits.
rng = random.Random(31823)
history_checks = 0
minimum_gain = float("inf")
for trial in range(50):
    cache = {}
    def tree(h, depth):
        global history_checks
        if depth == 0:
            terminal = 0.7 * rng.randrange(2) + 0.3 * rng.random()
            return terminal, terminal
        qs, mus = [], []
        for a in range(4):
            if a == 3:
                terminal = 0.7 * rng.randrange(2) + 0.3 * rng.random()
                qs.append(terminal - 0.01/3)
                mus.append(terminal - 0.01/3)
            else:
                p = rng.random()
                children = [tree(h+(a,z), depth-1) for z in range(2)]
                cost = 0.01/3 + 0.02/3 * rng.randrange(2)
                qs.append(p*children[0][0] + (1-p)*children[1][0] - cost)
                mus.append(p*children[0][1] + (1-p)*children[1][1] - cost)
        weights = [rng.random() for _ in range(4)]
        weights = [w/sum(weights) for w in weights]
        ref = sum(w*q for w,q in zip(weights,qs))
        action = max(range(4), key=qs.__getitem__)
        improved = mus[action]
        assert improved >= ref - 1e-12
        history_checks += 1
        return ref, improved
    ref, improved = tree((), 3)
    minimum_gain = min(minimum_gain, improved-ref)

# Original scorer input gradient is intentionally stopped. Frozen parameter
# status alone would not stop input Jacobians (demonstrated separately).
import torch
torch.set_num_threads(2)
torch.manual_seed(31823)
scorer = load_original("scorer")
model = scorer.NavigationScorer(896)
context = model.observe(torch.randn(1,3,896), torch.ones(1,3,dtype=torch.bool),
                        torch.randn(1,64,384), torch.randn(1,896),
                        torch.tensor([scorer.START]), torch.zeros(1,1))
future = torch.randn(1,4,64,384,requires_grad=True)
actions = torch.arange(4)[None]
out = model.score(context, future, actions)["utility"]
grad_future = torch.autograd.grad(out.sum(), future, allow_unused=True)[0]
assert grad_future is None
with torch.no_grad():
    changed = future.detach().clone()
    changed[:,0] = torch.randn_like(changed[:,0])
    finite_difference = (model.score(context,changed,actions)["utility"]-out).abs()
    assert finite_difference[0,0] > 0
    assert torch.equal(finite_difference[0,1:], torch.zeros(3))
frozen = torch.nn.Linear(2,1,bias=False).requires_grad_(False)
with torch.no_grad():
    frozen.weight.copy_(torch.tensor([[2.0,3.0]]))
x = torch.ones(1,2,requires_grad=True)
input_grad = torch.autograd.grad(frozen(x).sum(),x)[0]
assert torch.equal(input_grad,torch.tensor([[2.0,3.0]]))

# Re-execute A2 without editing its report or result. Its output path is
# redirected by executing unchanged source with a review-local __file__.
a2_source = OUT / "02_formal_checks.py"
replay = OUT / "03_a2_replay"
replay.mkdir(exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(a2_source.read_text(encoding="utf-8"),str(a2_source),"exec"),
         {"__file__":str(replay / "02_formal_checks.py"),"__name__":"__main__"})
a2_result = json.loads((replay / "02_formal_checks.json").read_text(encoding="utf-8"))
assert a2_result == json.loads((OUT/"02_formal_checks.json").read_text(encoding="utf-8"))

after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
         for p in ORIGINAL.iterdir() if p.is_file()}
assert before == after
result = dict(
    status="passed", scope="Bounded synthetic mathematics and original CPU scorer; no CE integration/training/model downloads",
    utility_compatible_example=support,
    sampling=dict(safe_Q=safe,risky_Q=q_risky,single_sample_selector_return=sample_selector),
    noisy_oracle=dict(three_equal_Q=q_risky,expected_max_three_rollouts=oracle_mean,
                      spurious_lift=oracle_mean-q_risky,common_seed_anticorrelated_max=common_seed_max),
    aliasing=dict(Bayes_Q=bayes_Q,minimum_MSE=bayes_mse,minimum_expected_regret=alias_regret),
    gap_variance=dict(correct_stochastic_sample_loss=native_sample_loss,
                      collapsed_loss=collapsed_sample_loss,two_sample_product_loss=cross_product),
    four_action_margin=dict(cases=margin_checks,overrides=overrides,epsilon=eps,margin=margin),
    independent_stochastic_policy_improvement=dict(trees=50,history_checks=history_checks,
                                                   minimum_root_gain=minimum_gain),
    gradient=dict(torch=torch.__version__,future_gradient_is_none=grad_future is None,
                  future_change_score_difference=finite_difference[0,0].item(),
                  frozen_layer_input_gradient=input_grad.tolist()),
    A2_replay_exact_match=True, original_hashes_unchanged=True,
    original_sha256=before)
(OUT / "03_adversarial_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
