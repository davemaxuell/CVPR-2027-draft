"""Build the revised proposal inventory without modifying the reviewed draft."""
from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parents[2]
old = {p['id']: p for p in json.loads((root/'proposals/proposals.json').read_text(encoding='utf-8'))}
text = (root/'proposals/final_proposals.md').read_text(encoding='utf-8')
section_keys = ['research_hypothesis','target_bottleneck','proposed_mechanism','formal_formulation','novelty_argument','benefits_tradeoffs','minimal_implementation','experimental_plan','falsification_criteria','risks_prerequisites']
ids = ['P1','P2','P4']
names = ['Recovery labels','Route validity','Negative memory']
refs = {
 'P1': ['S09-LAW','S09-NRM','S09-OCD','S09-GCR'],
 'P2': ['S09-CVS','S09-CART','S09-FCA'],
 'P4': ['S09-HIMECO','S09-EMBODISKILL','S09-DOCTOR','S09-CANTEST','S09-PSG'],
}
review_ids = {
 'P1': ['09-P1-01','09-P1-02','09-P1-03','10-P1-01','10-P1-02','10-P1-03'],
 'P2': ['09-P2-01','09-P2-02','10-P2-01','10-P2-02'],
 'P4': ['09-P4-01','09-P4-02','09-P4-03','10-P4-01','10-P4-02','10-P4-03'],
}
records=[]
for rank,(pid,name) in enumerate(zip(ids,names),1):
    block = re.search(r'^## '+pid+r' — .*?\n(.*?)(?=^## |\Z)',text,re.M|re.S).group(1)
    parts = re.findall(r'^### ([A-J])\. [^\n]+\n(.*?)(?=^### |\Z)',block,re.M|re.S)
    assert [p[0] for p in parts] == list('ABCDEFGHIJ'), (pid,[p[0] for p in parts])
    record = {
      'id':pid,'name':name,'cutoff':'2026-10-22',
      'status':'Revised conditional empirical study; coordinator acceptance and focused recheck pending',
      'provisional_rank':rank,
      'method_novelty':'Low; broad mechanism established. No original-method certification.',
      'study_originality_confidence':'Low to moderate, bounded by reviewed literature' if pid!='P4' else 'Lower confidence in residual empirical gap; conditional contamination study',
      'logical_status':'Revised against final 08 and 10 findings; assumptions explicit, no learned-policy or deployment guarantee',
      'empirical_status':'No proposed-system training or navigation benchmark; natural prevalence and utility untested',
      'implementation_feasibility':'Diagnostic conditional on missing assets/labels; model work gated by the diagnostic',
      'scientific_value':'Potentially an independently validated protocol, audit or dataset plus a controlled intervention; conference-level method value not established',
      'source_claim_ids':old[pid]['source_claim_ids'],
      'source_bottleneck_ids':old[pid]['source_bottleneck_ids'],
      'source_matrix_ids':old[pid]['source_matrix_ids'],
      'additional_source_ids':old[pid]['additional_source_ids'],
      'novelty_source_ids':refs[pid],
      'review_finding_ids':review_ids[pid],
      'logic_review_record':f'evaluations/08_findings.json:proposals[id={pid}]',
      'feasibility_review_record':f'evaluations/10_findings.json:findings[proposal_id={pid}]',
      'novelty_review':'evaluations/09_novelty.md',
      'detail_path':'proposals/final_proposals.md',
      'original_review_target':'proposals/draft_proposals.md',
    }
    record.update({key:body.strip() for key,(_,body) in zip(section_keys,parts)})
    records.append(record)
(root/'proposals/final_proposals.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

snapshot = {}
for rel in ['proposals/draft_proposals.md','proposals/proposals.json']:
    snapshot[rel] = hashlib.sha256((root/rel).read_bytes()).hexdigest()
archive=root/'evidence/07/reviewed_draft_hashes.json'
if archive.exists():
    assert json.loads(archive.read_text(encoding='utf-8')) == snapshot, 'Reviewed draft changed during revision'
else:
    archive.write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps({'retained':ids,'rejected_method':'P3','A_to_J_sections_per_record':10,'reviewed_drafts_preserved':True}))
