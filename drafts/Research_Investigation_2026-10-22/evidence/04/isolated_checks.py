"""Self-contained algebra checks; no fetched project code is imported or executed."""
import math,json,pathlib,datetime
out={"scope":"Executed finite numerical checks of printed formulas, not navigation benchmarks or general empirical validation.","date":"2026-10-22","tests":[]}
for p,a in [(.05,-.2),(.01,-.2),(.1,-.01)]:
    d=1-p+a*p
    literal_mean=p*a+(1-p)/d
    literal_eav=p*abs(a)+(1-p)/d
    appendix_mean=(p*a+1-p)/d
    appendix_eav=(p*abs(a)+1-p)/d
    out['tests'].append({'name':'FeedTTA printed Eq5 versus AppendixB.1','p':p,'alpha':a,'literal_mean_multiplier':literal_mean,'literal_expected_absolute_multiplier':literal_eav,'appendix_mean_multiplier':appendix_mean,'appendix_expected_absolute_multiplier':appendix_eav,'interpretation':'Appendix scaling is unbiased if applied to all coordinates; neither negative-alpha variant shrinks expected absolute gradient when denominator is positive.'})
for p,lam in [(.9,.4),(.9,.99),(.6,.4)]:
    q=lam+(1-lam)*p
    standard=p*(1-p)*math.log((1-p)/p)
    mixture=(1-lam)*p*(1-p)*math.log((1-q)/q)
    out['tests'].append({'name':'ATENA binary selected-logit entropy gradient','selected_probability':p,'lambda':lam,'standard_gradient':standard,'mixture_gradient':mixture,'magnitude_ratio':abs(mixture/standard),'interpretation':'Larger log-ratio gap does not universally imply a larger parameter gradient because the chain rule includes1-lambda.'})
out['tests'].append({'name':'FSTTA covariance rank bound','gradient_count_M':3,'parameter_dimension_D':768,'maximum_centered_covariance_rank':2,'minimum_zero_eigenvalues_exact_arithmetic':766,'interpretation':'Printed inverse-eigenvalue formula needs a pseudoinverse, floor or regularizer. Released fast path clamps eigenvalues to1e-6.'})
path=pathlib.Path(__file__).with_name('isolated_check_results.json');path.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
