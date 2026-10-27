"""Small self-contained logic checks; no downloaded project code is executed."""
import itertools,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[2]
def normalized(xs):
 m=sum(xs)/len(xs);s=math.sqrt(sum((x-m)**2 for x in xs)/len(xs));return [(x-m)/s for x in xs]
r=[-2.,0.,3.,5.];base=normalized(r)
discount_check=max(abs(a-b) for a,b in zip(base,normalized([.5**7*x for x in r])))
N=4;population=[-1.,1.]; vals=[];unbiased=[]
for xs in itertools.product(population,repeat=N):
 m=sum(xs)/N;v=sum((x-m)**2 for x in xs)/N
 vals.append(m*m-v/N);unbiased.append(m*m-(v*N/(N-1))/N)
map_values=[10.,min(10.,2.),min(min(10.,2.),9.)]
out={'scope':'Executed isolated arithmetic, not a benchmark or execution of repository code.',
'worldvln_discount':{'rewards':r,'positive_scale':.5**7,'max_normalized_advantage_difference':discount_check,'interpretation':'A shared positive reward scale cancels under per-group standardization without epsilon. This does not establish how the unreleased trainer groups or weights rewards.'},
'cvaml_variance':{'N':N,'distribution':'Independent equally likely -1/+1 values; true mean 0 and variance 1; fixed scalar value function. Enumerated all 16 samples.', 'expected_squared_mean_minus_population_variance_over_N':sum(vals)/len(vals),'expected_squared_mean_minus_unbiased_variance_over_N':sum(unbiased)/len(unbiased),'interpretation':'The population-variance correction leaves 1/N^2 = 0.0625 in this restricted case. The real policy-dependent critic introduces additional dependencies not simulated here.'},
'wmnav_merge':{'input_scores':[10,2,9],'ordinary_map_values':map_values,'goal_override_value':10,'interpretation':'An ordinary minimum update cannot raise a low score. The inspected goal-mask override is a distinct exception.'},
'navq_ties':{'graph_edges':[['s','a'],['s','b'],['a','g'],['b','g']],'shortest_paths_to_g':[['s','a','g'],['s','b','g']],'interpretation':'Two equal-length paths show shortest paths need not be unique. A stored tie-broken path can still define an operational target.'}}
(R/'evidence/02/logic_check_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
