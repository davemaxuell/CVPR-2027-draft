"""Independent, isolated finite checks for assignment 08.

No downloaded project module, model, checkpoint, simulator, or benchmark is executed.
Analytical identities are stated in the report; numerical checks are not proofs.
Run: python experiments/08/logic_checks.py (from any working directory).
"""
from pathlib import Path
import hashlib
import itertools
import json
import math
import platform
import sys

import numpy as np
import scipy
from scipy.stats import beta, t

OUT = Path(__file__).resolve().parent

def softmax(x):
    e=np.exp(np.asarray(x)-np.max(x)); return e/e.sum()

def set_loss(z, members):
    return -math.log(float(softmax(z)[members].sum()))

def numerical_grad(f,z,eps=1e-6):
    result=[]
    for j in range(len(z)):
        d=np.zeros(len(z)); d[j]=eps
        result.append((f(z+d)-f(z-d))/(2*eps))
    return np.array(result)

def product_graph_check():
    # Physical graph J--A and J--G. State q records whether A has been visited.
    graph={'J':{'A':1.,'G':1.},'A':{'J':1.},'G':{'J':1.}}
    states=list(itertools.product(graph,[0,1]))
    def update(q,v): return int(q or v=='A')
    # Bellman relaxation with only (G,1) terminal.
    v={s:(0. if s==('G',1) else math.inf) for s in states}
    for _ in range(30):
        new={s:min([v[s]]+[cost+v[(n,update(s[1],n))] for n,cost in graph[s[0]].items()]) for s in states}
        if new==v: break
        v=new
    def actions(s):
        return [n for n,c in graph[s[0]].items() if abs(c+v[(n,update(s[1],n))]-v[s])<1e-12]
    assert actions(('J',0))==['A'] and actions(('J',1))==['G']
    # Going to G first is inefficient but recoverable if the agent does not stop.
    cost_via_G=1+v[('G',0)]
    assert cost_via_G==5. and v[('J',0)]==3.
    geometric_teacher=['G','STOP']
    state_after_geometric=update(0,'G')
    geometric_teacher_accepts=(state_after_geometric==1)
    assert not geometric_teacher_accepts
    # An implementation using only endpoint G misses A on macro path J,A,J,G.
    q=0
    for node in ['A','J','G']: q=update(q,node)
    assert q==1 and update(0,'G')==0
    # Excluding A from the available actions leaves empty optimal action set.
    native_menu=['G']
    feasible=[a for a in actions(('J',0)) if a in native_menu]
    assert not feasible
    # Epsilon=2 admits J->A when q=1; A->J is exact optimal. Their repetition loops.
    slack_j_a=1+v[('A',1)]-v[('J',1)]
    assert slack_j_a==2
    return {'values':{str(k):val for k,val in v.items()},'before_A':actions(('J',0)),
            'after_A':actions(('J',1)),'macro_all_event_state':q,'endpoint_only_state':0,
            'restricted_menu_optimal_actions':feasible,'epsilon_loop_threshold':slack_j_a,
            'optimal_before_A_cost':v[('J',0)],'cost_of_valid_completion_after_G_first':cost_via_G,
            'G_first_viable_with_budget_5':cost_via_G<=5.,'G_first_viable_with_budget_3':cost_via_G<=3.,
            'geometric_teacher_continuation':geometric_teacher,'geometric_teacher_accepted':geometric_teacher_accepts,
            'scope':'Constructed deterministic graph; no navigation policy executed.'}

def set_loss_check():
    z=np.array([.2,-.5,1.1]); members=[0,1]; p=softmax(z); mass=p[members].sum()
    analytic=p.copy(); analytic[members]-=p[members]/mass
    numerical=numerical_grad(lambda zz:set_loss(zz,members),z)
    assert np.max(np.abs(analytic-numerical))<1e-8
    almost_collapsed=np.array([1-1e-8-1e-12,1e-8,1e-12])
    loss=-math.log(almost_collapsed[:2].sum())
    all_actions_gradient=p-p/p.sum()
    assert np.max(np.abs(all_actions_gradient))<1e-15 and loss<1e-10
    return {'analytic_gradient':analytic.tolist(),'finite_difference_gradient':numerical.tolist(),
            'max_absolute_gradient_difference':float(np.max(np.abs(analytic-numerical))),
            'two_valid_one_invalid_distribution':almost_collapsed.tolist(),'loss':loss,
            'scope':'The set loss encourages total acceptable mass, not diversity or uniformity within the set.'}

def quota_and_witness_check():
    scores=np.array([.95,.8,.6,.3,.2,.1]); bins=[0,0,0,1,1,1]; quota={0:2,1:1}
    selected=[0,1,3]
    feasible=[]
    for subset in itertools.combinations(range(6),3):
        if all(sum(bins[i]==k for i in subset)==q for k,q in quota.items()):
            feasible.append((float(scores[list(subset)].sum()),subset))
    best=max(feasible)
    assert tuple(selected)==best[1]
    # Two required evidence units; union-mass supervision can use only the first.
    alpha=np.array([1-1e-12,1e-12,0.])
    union_loss=-math.log(alpha[:2].sum())
    two_required_groups_loss=-math.log(alpha[0])-math.log(alpha[1])
    assert union_loss==0. and two_required_groups_loss>27
    # Missing-label mechanism: within identical E, 50 positive and 50 negative cases.
    # Only 5 of the positives and 45 of the negatives receive a known label.
    true_positive_rate=.5; labelled_positive_rate=5/(5+45)
    assert labelled_positive_rate==.1
    return {'quota_best_subset':list(best[1]),'quota_best_sum':best[0],
            'union_witness_loss':union_loss,'required_group_loss':two_required_groups_loss,
            'true_positive_rate_identical_E':true_positive_rate,
            'masked_BCE_optimum_under_selective_labels':labelled_positive_rate,
            'scope':'Masked BCE estimates the labelled conditional population without a missingness/support assumption.'}

def event_statistics(sequences):
    events='AB'; n=len(sequences); H=len(sequences[0]); one=np.zeros((H,2)); p=np.zeros(2); joint=np.zeros((2,2))
    for seq in sequences:
        for i,char in enumerate(seq):
            for k,event in enumerate(events): one[i,k]+=(char==event)/n
        for k,a in enumerate(events):
            p[k]+=(a in seq)/n
            for l,b in enumerate(events):
                joint[k,l]+=any(seq[i]==a and seq[j]==b for i in range(H) for j in range(i+1,H))/n
    return one,p,joint

def joint_event_check():
    left=['AB0','B0A','0AB']; right=['A0B','0BA','BA0']
    ltime,lp,lj=event_statistics(left); rtime,rp,rj=event_statistics(right)
    assert np.array_equal(ltime,rtime) and np.array_equal(lp,rp)
    assert math.isclose(lj[0,1],2/3) and math.isclose(rj[0,1],1/3)
    discounted=[]
    for gamma in [0.,.3,.5,.7,1.]:
        weights=gamma**np.arange(3)
        l=weights@ltime; r=weights@rtime
        assert np.array_equal(l,r)
        discounted.append({'gamma':gamma,'left':l.tolist(),'right':r.tolist()})
    # Necessary pairwise bounds alone do not imply a coherent trajectory law.
    # With A and B known to occur exactly once at distinct times, J_AB + J_BA = 1.
    incoherent={'pA':1.,'pB':1.,'J_AB':1.,'J_BA':1.}
    assert incoherent['J_AB']<=min(incoherent['pA'],incoherent['pB'])
    return {'left_uniform_sequences':left,'right_uniform_sequences':right,
            'timewise_marginals':ltime.tolist(),'ever_event_probabilities':lp.tolist(),
            'left_J':lj.tolist(),'right_J':rj.tolist(),'discounted_additive_targets':discounted,
            'pairwise_bound_passes_but_single_occurrence_order_is_incoherent':incoherent,
            'scope':'Symbolic event representation only. Does not prove collisions for actual rich caption embeddings or learnability from navigation inputs.'}

def attribution_check():
    table=[{'identity_correct':d,'arrival_correct':r,'success':int(d and r)} for d,r in itertools.product([0,1],repeat=2)]
    assert any(x['identity_correct']==1 and x['success']==0 for x in table)
    # Two equally common hidden causes give identical observable E and S=0.
    unavoidable_error=.5
    # A calibrated constant q=.9 can become wrong on a selected witness subgroup.
    errors=np.array([1]*90+[0]*10); q=np.full(100,.9); witness=np.array([0]*90+[1]*10)
    assert math.isclose(float(errors.mean()),float(q.mean()),abs_tol=1e-12)
    selected=(q>=.9)&(witness==1)
    return {'success_truth_table':table,'identical_observation_equal_cause_Bayes_error':unavoidable_error,
            'global_q':.9,'global_identity_error_rate':float(errors.mean()),
            'witness_selected_identity_error_rate':float(errors[selected].mean()),
            'scope':'Global calibration is insufficient after a correlated witness gate. No claim that this distribution occurs in PSG-Nav.'}

def uagm_check():
    # Input independently transcribed from actual page 19 Table 11 rendered in evidence/08.
    rows=[('DUET','SR',72.22,.21,78.32,.13,3.42e-12),
          ('BEVBert','SR',75.82,.25,78.32,.13,8.91e-11),
          ('VER','SR',76.37,.18,78.32,.13,1.58e-10),
          ('DUET','SPL',60.41,.27,66.47,.17,6.60e-15),
          ('BEVBert','SPL',64.14,.22,66.47,.17,7.30e-10),
          ('VER','SPL',65.07,.21,66.47,.17,2.90e-13)]
    result=[]
    for name,metric,mx,sx,my,sy,reported in rows:
        gap=my-mx
        # Each displayed two-decimal mean or SD permits +/- .005 rounding.
        min_sd=abs(sx-sy)-.01; max_sd=sx+sy+.01
        max_t=math.sqrt(5)*(gap+.01)/min_sd
        min_t=math.sqrt(5)*(gap-.01)/max_sd
        min_p=float(2*t.sf(max_t,df=4)); max_p=float(2*t.sf(min_t,df=4))
        required_t=float(t.isf(reported/2,df=4)); required_sd=math.sqrt(5)*gap/required_t
        # One-sided would only halve p. Bound remains violated.
        assert reported<min_p/2
        result.append({'baseline':name,'metric':metric,'mean_gap':gap,'reported_p':reported,
                       'rounded_difference_SD_interval':[min_sd,max_sd],
                       'paired_t_interval':[min_t,max_t],'possible_two_sided_p_interval':[min_p,max_p],
                       'minimum_one_sided_p':min_p/2,'reported_p_incompatible_with_five_run_interpretation':True,
                       'difference_SD_required_for_reported_p':required_sd})
    return {'assumptions':['Five paired run-level observations','Displayed plus/minus values are SDs of these same five values',
                           'Ordinary paired t test with df=4','Means and SDs rounded to nearest .01'],
            'rows':result,'scope':'Reporting consistency only. A different statistical unit or protocol needs clarification; no inference about misconduct or underlying benchmark validity.'}

def feedtta_check():
    result=[]
    for alpha,p in [(-.2,.01),(-.2,.05),(-.001,.01),(-.01,.05)]:
        den=alpha*p+1-p
        main_mean=p*alpha+(1-p)/den
        appendix_mean=(p*alpha+1-p)/den
        main_abs=p*abs(alpha)+(1-p)/den
        appendix_abs=(p*abs(alpha)+1-p)/den
        assert den>0 and main_mean>1 and math.isclose(appendix_mean,1)
        assert main_abs>1 and appendix_abs>1
        result.append({'alpha':alpha,'p':p,'D':den,'main_eq5_mean_multiplier':main_mean,
                       'appendix_mean_multiplier':appendix_mean,'main_abs_multiplier':main_abs,
                       'appendix_abs_multiplier':appendix_abs,'printed_approx_abs_multiplier':1-p*p-alpha*p})
    return {'rows':result,'scope':'Scalar multiplier algebra; no conclusion about empirical plasticity or forgetting.'}

def atena_check():
    records=[]
    for prob in [.51,.8,.95]:
        z=np.array([math.log(prob/(1-prob)),0.])
        standard=prob*(1-prob)*math.log((1-prob)/prob)
        for lam in [.1,.5,.9,.99,.9999]:
            q=lam+(1-lam)*prob
            analytic=(1-lam)*prob*(1-prob)*math.log((1-q)/q)
            def loss(zz):
                pp=softmax(zz); qq=(1-lam)*pp; qq[0]+=lam
                return float(-np.sum(qq*np.log(qq)))
            numeric=numerical_grad(loss,z)[0]
            assert abs(analytic-numeric)<1e-8
            records.append({'selected_probability':prob,'lambda':lam,'standard_logit_gradient':standard,
                            'mixture_logit_gradient':analytic,'finite_difference':float(numeric),
                            'magnitude_ratio':abs(analytic/standard)})
    assert any(r['magnitude_ratio']<1 for r in records) and any(r['magnitude_ratio']>1 for r in records)
    return {'rows':records,'scope':'Binary policy with fixed selected action 0. Does not compare full neural-network gradient norms or benchmark adaptation.'}

def cpnav_check():
    n=500; v=225; coverage_target=.55
    a=n+1-v; b=v
    upper=float(beta.ppf(.99,a,b)); lower=float(beta.ppf(.01,a,b))
    chance=float(beta.sf(coverage_target,a,b))
    assert upper>=coverage_target and chance<.99 and lower<coverage_target
    # Exact order-statistic argument: with continuous iid scores, conditional
    # coverage at rank k has Beta(k,n+1-k) distribution over calibration sets.
    # Keeping a goal branch in a set is not enough to execute it in a finite budget.
    retained=True; budget=2; high_score_bad_branch_cost=3
    goal_reached=retained and budget>high_score_bad_branch_cost
    assert retained and not goal_reached
    return {'n':n,'v':v,'beta_parameters':[a,b],'target':coverage_target,
            'upper_99_percentile':upper,'lower_1_percentile':lower,
            'probability_conditional_coverage_at_least_target':chance,
            'oracle_path_retained':retained,'goal_reached_in_constructed_budget':goal_reached,
            'scope':'Checks the printed Beta criterion and a constructed finite-budget search counterexample; no CPNAV model run.'}

def main():
    checks={'P1_product_graph':product_graph_check(),'P1_set_loss':set_loss_check(),
            'P2_quota_witness_missing_labels':quota_and_witness_check(),'P3_joint_events':joint_event_check(),
            'P4_attribution':attribution_check(),'UAGM_statistics':uagm_check(),
            'FeedTTA_scaling':feedtta_check(),'ATENA_gradient':atena_check(),'CPNAV_beta_and_budget':cpnav_check()}
    record={'reviewer':'assignment_08','cutoff':'2026-10-22','python':sys.version,'platform':platform.platform(),
            'numpy':np.__version__,'scipy':scipy.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'evidence_type':'EXECUTED ISOLATED NUMERICAL/FINITE CHECKS','project_code_executed':False,
            'benchmark_executed':False,'all_assertions_passed':True,'checks':checks}
    (OUT/'logic_results.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'checks':len(checks),'all_assertions_passed':True,'results':str(OUT/'logic_results.json')}))

if __name__=='__main__': main()
