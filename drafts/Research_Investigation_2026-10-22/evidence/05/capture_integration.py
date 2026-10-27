"""Record the original dossiers and shared snapshots actually incorporated by05."""
from pathlib import Path
import json, hashlib
R=Path(__file__).resolve().parents[2]
E=R/'evidence/05'
scope=json.loads((E/'original_scope.json').read_text(encoding='utf-8'))
records=[]
for item in scope:
    p=R/item['dossier_read']
    records.append({'paper_id':item['paper_id'],'dossier_path':item['dossier_read'],
                    'sha256_at_integration':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'direct_original_pdf_pages_by05':item['primary_pdf_pages_read_by_05']})
shared=[]
for paper in ['navnuances','uln']:
    p=R/'repositories/04'/paper/'source.json'
    x=json.loads(p.read_text(encoding='utf-8-sig'))
    shared.append({'paper_id':paper,'source_manifest':str(p.relative_to(R)),'repo_url':x['repo_url'],'commit':x['commit'],
                   'direct05_audit':f'evidence/05/{paper}_code_audit.md','source_only':True,'project_executed':False})
out={'date':'2026-10-22','cutoff':'2026-10-22',
     'completion_scope':'Assignment05 follow-up audit, incorporating04 relevant final source sections; unrelated04 work may continue',
     'originals':records,'shared_original_snapshots_directly_inspected':shared,
     'cross_assignment_source':{'source_id':'FU06-reinsvln','catalog':'evidence/06/sources.json','direct05_pages':[3],
                                'other_source_scope':'Attributed to06 full source card, not claimed as direct05 review'},
     'final04_deltas':['VLN-Trans latent conditioning and unavailable raw SyFiS builder','GELA actual supervised losses and augmented-data launcher',
                      'FGR2R dataset/chunker-only release','GSA persistent graph with episode-state reset','NAP old release and action-set pruning',
                      'slow4fast mocked reflection and placeholder embeddings','M-Track, NavNuances and ULN shared05 audits integrated by04']}
(E/'integration_records.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Captured {len(records)} dossier hashes and {len(shared)} shared original source audits.')
