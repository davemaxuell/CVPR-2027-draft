"""Self-contained finite arithmetic examples; no imported research project code."""
from pathlib import Path
import math,json
def softmax(xs):
 m=max(xs);v=[math.exp(x-m) for x in xs];return [x/sum(v) for x in v]
p=[.6,.4];u=[.2,.8];g=1.5
arithmetic=[g*a+(1-g)*b for a,b in zip(p,u)]
log_guidance=softmax([g*math.log(a)+(1-g)*math.log(b) for a,b in zip(p,u)])
assert max(abs(a-b) for a,b in zip(arithmetic,log_guidance))>1e-3
outside_simplex=[2*a-b for a,b in zip([.8,.2],[.4,.6])]
assert min(outside_simplex)<0
future_a=[3.,1.];future_b=[2.,0.];shift_b=[x+10 for x in future_b]
assert all(abs(a-b)<1e-12 for a,b in zip(softmax(future_b),softmax(shift_b)))
score_before=[2+1.2*max(future_a),1+1.2*max(future_b)]
score_after=[2+1.2*max(future_a),1+1.2*max(shift_b)]
assert score_before[0]>score_before[1] and score_after[0]<score_after[1]
out=dict(executed='2026-10-22',setup='Python standard library; two explicit finite examples; no model/simulator',
 cag=dict(conditional=p,unconditional=u,gamma=g,arithmetic=arithmetic,normalized_log_guidance=log_guidance,extrapolation_outside_simplex=outside_simplex,
 interpretation='Arithmetic combination of probabilities is not generally normalized log-density guidance. This is not a test of the released CAG action/logit branches and does not refute empirical results.'),
 uln=dict(future_policy_before=softmax(future_b),future_policy_after=softmax(shift_b),branch_scores_before=score_before,branch_scores_after=score_after,
 interpretation='Cross-state raw-logit comparison is sensitive to additive state-dependent offsets, even when within-state action probabilities are unchanged. No claim that trained ULN logits exhibit this failure frequency.'),
 evidence_type='EXECUTED FINITE CHECK; not benchmark validation or general performance proof')
(Path(__file__).parent/'finite_check_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
