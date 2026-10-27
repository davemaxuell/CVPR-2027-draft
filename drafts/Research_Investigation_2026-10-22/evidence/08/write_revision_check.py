"""Record the focused final logic review of immutable input bytes."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
paths=['proposals/final_proposals.md','proposals/final_proposals.json',
       'proposals/Revision_Record.md','proposals/rejected_ideas.md',
       'evaluations/08_logic.md','evaluations/08_findings.json']
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
markdown=(ROOT/paths[0]).read_text(encoding='utf8')
records=json.loads((ROOT/paths[1]).read_text(encoding='utf8'))
fields=['research_hypothesis','target_bottleneck','proposed_mechanism','formal_formulation',
        'novelty_argument','benefits_tradeoffs','minimal_implementation','experimental_plan',
        'falsification_criteria','risks_prerequisites']
mismatches=[{'proposal':p['id'],'field':f} for p in records for f in fields if p.get(f,'') not in markdown]
assert [p['id'] for p in records]==['P1','P2','P4'] and not mismatches
result={
 'assignment':'08','review_stage':'focused final revision recheck','date':'2026-10-22',
 'decision':'Bounded acceptance as conditional diagnostic studies; no residual material logical blocker',
 'input_sha256':hashes,
 'consistency_check':{'retained_ids':['P1','P2','P4'],'substantive_fields_checked':30,'mismatches':mismatches},
 'findings':[
  {'id':'08-R-P1','proposal':'P1','status':'Material objections resolved in formulation',
   'resolved':['Current-state terminal predicate C(v,q)','Complete-continuation semantic audit, separate from efficiency',
               'Declared-budget feasibility','Common primitive-neighbor/STOP pilot and node-landmark grammar',
               'Native evolving-menu state and complete termination/return contract',
               'Ambiguous action/state labels and transferable state encoding'],
   'residual_limits':['Natural semantic error prevalence untested','Annotation reliability untested',
                      'Deployable state sufficiency and incremental benefit untested'],
   'implementation_clarification':'Spell out c(STOP)=0, V(success terminal)=0, V(failure terminal)=infinity. Wording clarification is sufficient; keep STOP absorbing.',
   'remaining_material_logic_blocker':False},
  {'id':'08-R-P2','proposal':'P2','status':'Material objections resolved or explicitly bounded',
   'resolved':['OR applies to complete alternative evidence bundles','Conjunctive group variant separately specified',
               'Quota feasibility and quality-floor tradeoff','Unresolved status and masked-label population limits',
               'Sampling, route-family, annotation-effort and exposure controls'],
   'residual_limits':['Complete evidence may be unavailable','Encoder may ignore required bundle elements',
                      'Selection and downstream utility untested'],
   'remaining_material_logic_blocker':False},
  {'id':'08-R-P4','proposal':'P4','status':'Material objections resolved or explicitly bounded',
   'resolved':['Outcome-only failure is not identity evidence','Observable identification and missing-label limits',
               'Restricted controller-invariance assumptions','Joint correlated-gate evaluation',
               'Source/paper separation and no-write/quarantine controls'],
   'residual_limits':['Natural harmful-write incidence untested','Deployable identifying evidence not established',
                      'Simple alternatives may suffice','Stale-memory tradeoff untested'],
   'remaining_material_logic_blocker':False},
  {'id':'08-R-P3','proposal':'P3','status':'Rejected as an original method; diagnostic appendix accurately scoped',
   'resolved':['Removed from retained proposal set','Corrected matched-marginal counterexample',
               'Strict-order monitor convention','NavQ sum and policy/probability limits'],
   'residual_limits':['No actual NavQ collision, learnability or performance established','Novelty search not repeated by08'],
   'remaining_material_logic_blocker':False}
 ],
 'new_executed_work':'Document/JSON consistency and SHA256 hashing only',
 'new_project_code_executed':False,'new_navigation_benchmarks':False,'new_literature_search':False,
 'acceptance_does_not_establish':['Originality','Natural prevalence','Successful training','Navigation gains',
                                  'Learned-policy convergence','Calibrated deployment risk'],
 'report':'evaluations/08_revision_check.md'
}
(ROOT/'evaluations/08_revision_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'decision':result['decision'],'fields_checked':30,'mismatches':mismatches,'hashes':hashes},indent=2))
