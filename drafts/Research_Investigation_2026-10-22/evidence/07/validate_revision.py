"""Validate revision artifacts/references; no scientific or empirical certification."""
from pathlib import Path
import hashlib
import json
import re

root=Path(__file__).resolve().parents[2]
def read(rel): return json.loads((root/rel).read_text(encoding='utf-8'))
def sha(rel): return hashlib.sha256((root/rel).read_bytes()).hexdigest()
errors=[]
records=read('proposals/final_proposals.json')
if [r['id'] for r in records] != ['P1','P2','P4']: errors.append('Final retained IDs differ from P1/P2/P4.')
for rel,expected in read('evidence/07/reviewed_draft_hashes.json').items():
    if sha(rel)!=expected: errors.append('Reviewed draft modified: '+rel)
claims={x['claim_id'] for x in read('evidence/claims.json')}
matrices={x['bottleneck_id'] for n in (5,6) for x in read(f'followups/{n:02d}_matrix.json')}
novelty_sources={x['id'] for x in read('evidence/09/sourcecards.json')['sources']}
review_findings={x['id'] for x in read('evaluations/09_findings.json')['findings']}
review_findings.update(x['id'] for r in read('evaluations/10_findings.json')['findings'] for x in r['critical_findings'])
for r in records:
    for key,allowed in [('source_claim_ids',claims),('source_matrix_ids',matrices),('novelty_source_ids',novelty_sources),('review_finding_ids',review_findings)]:
        for value in r[key]:
            if value not in allowed: errors.append(f"{r['id']}: missing {key} {value}")
    for key in ['research_hypothesis','target_bottleneck','proposed_mechanism','formal_formulation','novelty_argument','benefits_tradeoffs','minimal_implementation','experimental_plan','falsification_criteria','risks_prerequisites']:
        if not r.get(key): errors.append(f"{r['id']}: missing {key}")
link_count=0
for rel in ['proposals/final_proposals.md','proposals/rejected_ideas.md','proposals/Revision_Record.md','assignments/07_summary.md']:
    path=root/rel
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        if target.startswith(('http://','https://','#')): continue
        link_count+=1
        if not (path.parent/target.split('#',1)[0]).exists(): errors.append(f'{rel}: missing {target}')
review_statuses={
 '08':read('evaluations/08_findings.json')['completion_status'],
 '09':read('evaluations/09_findings.json')['status'],
 '10':read('evaluations/10_findings.json')['status'],
}
if any(s.lower()!='complete' for s in review_statuses.values()): errors.append('Independent review not complete.')
scope={
 'cutoff':'2026-10-22',
 'revision_status':'Complete for coordinator and focused reviewer recheck; acceptance not self-certified',
 'retained_ids':['P1','P2','P4'],'rejected_original_method_id':'P3',
 'review_statuses':review_statuses,
 'review_inputs':{rel:sha(rel) for rel in ['evaluations/08_logic.md','evaluations/08_findings.json','evaluations/09_novelty.md','evaluations/09_findings.json','evaluations/10_feasibility.md','evaluations/10_findings.json','experiments/Initial_Experiment_Plan.md']},
 'direct_additional_primary_sections_read':[
  {'source':'S09-LAW','pages':[3,4],'path':'evidence/09/sources/law.txt'},
  {'source':'S09-CVS','pages':[3,4,5],'path':'evidence/09/sources/cvs.txt'},
  {'source':'S09-NRM','pages':[5,6],'path':'evidence/09/sources/noisy_rm.txt'},
  {'source':'S09-HIMECO','pages':[5,6],'path':'evidence/09/sources/himeco.txt'},
 ],
 'other_source_scope':'Other new primary/code findings are attributed to independent 08–10; no claim of independent full rereading by 07.',
 'new_scientific_experiments_executed':False,
 'validation_only':'Script builds records and checks reference integrity; no source/model/benchmark execution.',
 'preserved_drafts':read('evidence/07/reviewed_draft_hashes.json'),
}
(root/'evidence/07/revision_scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
# The scope file is generated here; existence is checked after generation.
errors=[e for e in errors if 'missing ../evidence/07/revision_scope.json' not in e]
result={'scope':'Artifact integrity only','retained_proposals':len(records),'A_to_J_sections_per_proposal':10,'local_links_checked':link_count,'independent_reviews_complete':review_statuses,'reviewed_draft_hashes_unchanged':not any('draft modified' in e for e in errors),'errors':errors,'passed':not errors}
(root/'evidence/07/revision_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
raise SystemExit(bool(errors))
