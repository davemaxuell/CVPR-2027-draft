"""Original finite checks; no downloaded project code is imported or executed."""
from pathlib import Path
import json, numpy as np
from scipy.stats import beta
out={}
n=500; v=225; target=.55
out['cpnav_quantile_direction']={'assumptions':'iid continuous episode scores; k=n+1-v order statistic; example values, not author calibration data','n':n,'v':v,'target_coverage':target,'upper_99_percent_quantile':float(beta.ppf(.99,n+1-v,v)),'lower_1_percent_quantile':float(beta.ppf(.01,n+1-v,v)),'probability_coverage_at_least_target':float(beta.sf(target,n+1-v,v)),'interpretation':'An upper .99 quantile passing the target does not establish .99 confidence of coverage exceeding that target.'}
e2=np.array([.01,4.]);r2=np.array([.01,0.])
out['hep_loss_reduction']={'residual_squared_norms':e2.tolist(),'unexplained_squared_norms':r2.tolist(),'ratio_of_sums':float(r2.sum()/e2.sum()),'mean_of_ratios':float((r2/e2).mean()),'interpretation':'The two objectives weight patches differently; this is not a benchmark result.'}
mu=np.array([1.,0.]);goal=np.array([0.,1.]);e=goal-mu;u=e/np.linalg.norm(e);ridge=.001
r=e-u*(u@e)/(u@u+ridge)
out['hep_task_direction_counterexample']={'unit_prediction':mu.tolist(),'unit_goal':goal.tolist(),'basis':u.tolist(),'cosine_distance':float(1-mu@goal),'orthogonal_score_with_ridge':float(r@r),'interpretation':'If a goal-relevant difference lies inside a learned nuisance subspace, orthogonal scoring suppresses it. This is a constructed case, not evidence that the trained model learns this subspace.'}
z=np.array([.3,-.2]);w=np.array([.35,.35]);soft=lambda a:np.exp(a-a.max())/np.exp(a-a.max()).sum()
out['dart_equal_prior_cancellation']={'original_weights':soft(z).tolist(),'with_equal_floored_memory_priors':soft(z+np.log(w)).tolist(),'max_abs_difference':float(np.abs(soft(z)-soft(z+np.log(w))).max()),'interpretation':'Equal multiplicative memory priors cancel within a cell; a singleton cell is likewise unaffected.'}
out['retained_path_budget_counterexample']={'tree':'Root has admissible decoy D with score .9 and goal G with score .8; D requires one action and return requires one action; budget is one action. Both retained at threshold .5; highest-score search selects D.','oracle_path_retained':True,'goal_reached_within_budget':False,'interpretation':'Set inclusion alone is insufficient for budgeted success; additional search/budget assumptions are necessary.'}
Path('evidence/06/math_checks_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
