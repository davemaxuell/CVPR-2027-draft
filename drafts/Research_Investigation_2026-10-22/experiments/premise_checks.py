"""Small mathematical counterexamples, not navigation benchmark experiments."""
from pathlib import Path
import json, math, random, platform

def softmax(z):
    largest=max(z);e=[math.exp(v-largest) for v in z];return [x/sum(e) for x in e]
def lse(z):
    largest=max(z);return largest+math.log(sum(math.exp(v-largest) for v in z))
def sigmoid(x):return 1/(1+math.exp(-x))

rng=random.Random(261022)
errors=[]
for _ in range(1000):
    z=[rng.uniform(-10,10) for _ in range(rng.randrange(2,51))]
    joint=softmax(z)
    stop=sigmoid(z[0]-lse(z[1:]))
    conditional=softmax(z[1:])
    factored=[stop]+[(1-stop)*p for p in conditional]
    errors.append(max(abs(a-b) for a,b in zip(joint,factored)))
assert max(errors)<1e-12

frontiers=[dict(movement_candidates=n,stop_logit=1,movement_logit=0,
                stop_probability=math.exp(1)/(math.exp(1)+n)) for n in [1,5,20]]
assert frontiers[0]['stop_probability']>frontiers[-1]['stop_probability']

# A straight 10 m reference path, with successful termination at 8 m under a 3 m
# success radius. The usual SPL definition still returns 1, despite an unvisited
# 2 m suffix. This disproves equivalence to exact reference-path replay.
shortest=10.0;executed=8.0;endpoint_error=2.0;radius=3.0
spl=float(endpoint_error<radius)*shortest/max(shortest,executed)
assert spl==1.0

gaussian=[]
for alpha in [3.0,3.8]:
    sigma=.75;positions=list(range(1,8))
    raw=[math.exp(-((k-alpha)**2)/(2*sigma**2)) for k in positions]
    weights=[v/sum(raw) for v in raw]
    gaussian.append(dict(alpha=alpha,sigma=sigma,weights=weights,
                         floor_weight=weights[math.floor(alpha)-1]))
assert all(v>0 for item in gaussian for v in item['weights'])

# Mean pooling and log determinants: elementary properties, not measured model
# failures. No semantically meaningful embedding distribution is assumed here.
a=[0.0]*512;b=[0.0]*512;b[0]=1;b[1]=-1
assert a!=b and sum(a)/512==sum(b)/512
logdet_scaled_identity=512*math.log(.1)
assert logdet_scaled_identity<0

result=dict(date='2026-10-22',python=platform.python_version(),seed=261022,
            evidence_type='EXECUTED TOY NUMERICAL CHECK',benchmark_performance_tested=False,
            softmax_factorization=dict(trials=1000,max_absolute_error=max(errors),
              interpretation='A categorical policy always factors into a stop Bernoulli and conditional movement distribution. A new head needs a substantive feature, objective, calibration or decision-rule difference.'),
            frontier_normalization=frontiers,
            spl_counterexample=dict(shortest_path=shortest,executed_length=executed,endpoint_error=endpoint_error,success_radius=radius,spl=spl,exact_reference_replay=False),
            gaussian_window=gaussian,
            mean_pool_counterexample=dict(dimensions=512,distinct_vectors=True,equal_scalar_mean=True),
            logdet_counterexample=dict(dimensions=512,diagonal=.1,logdet=logdet_scaled_identity),
            limitations=['No trained navigator, simulator, checkpoint or benchmark was executed.',
              'These finite numerical checks illustrate algebraic identities/counterexamples; they do not establish empirical prevalence.',
              'The scaling experiment fixes logits artificially and does not prove frontier count is the cause of real stopping failures.',
              'Non-injective pooling alone does not show failure on the learned embedding distribution.'])
out=Path(__file__).with_name('premise_checks_results.json')
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
