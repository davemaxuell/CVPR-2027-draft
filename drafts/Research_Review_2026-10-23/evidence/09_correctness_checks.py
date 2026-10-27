"""A9 bounded independent centered-loss and scorer-input-gradient checks.
No CE efficacy claim; original files loaded read-only and never imported for bytecode.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import types
import torch

torch.set_num_threads(2)
torch.manual_seed(91023)
OUT = Path(__file__).resolve().parent
ORIGINAL = OUT.parents[1] / "Research Approach" / "Scorer" / "scorer.py"
source = ORIGINAL.read_text(encoding="utf-8")
before = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
K = 4
s = torch.randn(7,K,dtype=torch.float64,requires_grad=True)
y = torch.randn(7,K,dtype=torch.float64)
e = s-y
pe = e-e.mean(dim=1,keepdim=True)
loss = pe.square().mean()
pair_sum = sum(((s[:,a]-s[:,b])-(y[:,a]-y[:,b])).square()
               for a in range(K) for b in range(a+1,K))
pair_loss = pair_sum.mean()/(K*K)
assert torch.allclose(loss,pair_loss,atol=1e-14,rtol=1e-14)
grad = torch.autograd.grad(loss,s)[0]
analytic = 2*pe/(s.shape[0]*K)
assert torch.allclose(grad,analytic,atol=1e-14,rtol=1e-14)
shifted = (s+torch.arange(7,dtype=s.dtype)[:,None])-y
assert torch.allclose((shifted-shifted.mean(1,keepdim=True)).square().mean(),loss)

# Exhaustive finite examples also include score ties and the margin deployment rule.
checks = 0
worst_greedy_slack = float("inf")
worst_margin_slack = float("inf")
for true in itertools.product([0.,0.5,1.],repeat=4):
    q = torch.tensor(true,dtype=torch.float64)
    for scores in itertools.product([0.,0.5,1.],repeat=4):
        v = torch.tensor(scores,dtype=torch.float64)
        centered = v-q-(v-q).mean()
        bound = (2.0**0.5)*centered.norm().item()
        best = int(v.argmax())
        regret = (q.max()-q[best]).item()
        margin = 0.5
        selected = best if v[best]-v[0] > margin else 0
        margin_regret = (q.max()-q[selected]).item()
        assert regret <= bound+1e-12
        assert margin_regret <= margin+bound+1e-12
        worst_greedy_slack = min(worst_greedy_slack,bound-regret)
        worst_margin_slack = min(worst_margin_slack,margin+bound-margin_regret)
        checks += 1

# Only hypothetical differentiable change: remove explicit predicted-future detach.
# This is an in-memory copied source, not an implemented project change.
needle = "context.current_patches[:, None], future_patches.detach(),"
assert source.count(needle) == 1
patched_source = source.replace(needle,"context.current_patches[:, None], future_patches,")
module = types.ModuleType("a9_scorer_copy")
sys.modules[module.__name__] = module
exec(compile(patched_source,str(ORIGINAL)+" [A9 in-memory future-gradient variant]","exec"),module.__dict__)
model = module.NavigationScorer(896).requires_grad_(False).eval()
ctx = model.observe(torch.randn(1,3,896),torch.ones(1,3,dtype=torch.bool),
                    torch.randn(1,64,384),torch.randn(1,896),
                    torch.tensor([module.START]),torch.zeros(1,1))
future = torch.randn(1,4,64,384,requires_grad=True)
labels = torch.tensor([[0.9,0.1,0.4,0.2]],requires_grad=True)
scores = model.score(ctx,future,torch.arange(4)[None])["utility"]
residual = scores-labels.detach()
task = (residual-residual.mean(1,keepdim=True)).square().mean()
task.backward()
movement_norms = future.grad[0,:3].flatten(1).norm(dim=1)
stop_norm = future.grad[0,3].norm().item()
assert torch.isfinite(future.grad).all() and (movement_norms > 0).all()
assert stop_norm == 0 and labels.grad is None
assert all(p.grad is None for p in model.parameters())
after = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
assert before == after

result = {
 "status":"passed",
 "scope":"Independent CPU algebra, finite score checks and in-memory copied scorer route; no integration/training/CE evidence",
 "torch":torch.__version__,
 "proposal_v1_sha256":"4CBC57E6781D83BF3C3A88F9030C54EFBD39A45B8B98535EB68D5DD163D77F9F",
 "centered_vs_pair_loss_abs_error":abs(loss.item()-pair_loss.item()),
 "gradient_max_abs_error":(grad-analytic).abs().max().item(),
 "pair_mean_to_centered_multiplier_K4":3/8,
 "greedy_and_margin_cases":checks,
 "minimum_greedy_bound_slack":worst_greedy_slack,
 "minimum_margin_bound_slack":worst_margin_slack,
 "input_gradient_variant":{
   "movement_future_gradient_norms":movement_norms.tolist(),
   "stop_future_gradient_norm":stop_norm,
   "frozen_parameter_gradients_absent":True,
   "label_gradients_absent":True
 },
 "original_scorer_unchanged":before==after,
 "original_scorer_sha256":before
}
(OUT/"09_correctness_checks.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))

