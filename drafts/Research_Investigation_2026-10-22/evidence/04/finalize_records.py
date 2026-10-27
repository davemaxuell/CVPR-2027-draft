import json,re,pathlib
from build import ROOT,NOTES,render
n=json.loads(NOTES.read_text(encoding='utf8'))
n['landmark_rxr']['implementation']=n['landmark_rxr']['implementation'].replace('Supplement review pending.','The official nine-page supplement was reviewed, including model details and the annotation datasheet.')
n['mtrack']['implementation']=n['mtrack']['implementation'].replace('Assignment 05 is independently inspecting the official runtime milestone implementation; pinned paths and findings will be incorporated. Main paper personally reviewed here; no code execution.','The main paper was reviewed by assignment 04; the completed independent runtime audit by assignment 05 is incorporated below.')

# Normalize the explicit prose, preserving model names, identifiers, paths and URLs.
prefixes='batch|Batch|lr|Table|Figures|Figure|Fig|Eqs|Eq|SupplementTable|Supplement|Appendix|and|is|are|uses|for|at|SR|SPL|OSR|RGSPL|GELA|reaches|alone|ELSA|LBP|three|rises|within|roughly|direction|numerical|landmark|to|stays|score|by|Only|rates|least|becomes|gives|length|worsens|with|lines|than|base|RL|SGR|episodes|contiguous|above|case|falls|width|SAL|Tent|about|not|decay|cells|temperature|while|tau|offset|At|reward|line|volume'
def clean(s):
    s=re.sub(r'\b('+prefixes+r')(?=\d)',r'\1 ',s)
    s=re.sub(r'(?<=\d)(points|pages|PDF|SR|SPL)\b',r' \1',s)
    s=re.sub(r'\.py(?=\d)', '.py:',s)
    s=s.replace('a120-degree','a 120-degree').replace('a2D-input','a 2D-input').replace('two3090s','two 3090s').replace('normalized4D','normalized 4D').replace('a(1−λ)','a (1−λ)').replace('all7pages','all 7 pages').replace('uses2026','uses 2026').replace('supplementary5981','supplementary 5981').replace('includes1-lambda','includes 1-lambda').replace('to1e-6','to 1e-6')
    s=s.replace('.. Follow-ups:', '. Follow-ups:')
    return s
for pid,v in n.items():
    for k in ['scope','problem','method','novelty','results','implementation','repo_status','venue']:
        if isinstance(v.get(k),str):v[k]=clean(v[k])
    v['associated']=[clean(x) if isinstance(x,str) else x for x in v.get('associated',[])]
    for x in v.get('sources',[]):x['locator']=clean(x.get('locator','Primary source'))
    for b in v.get('bottlenecks',[]):
        for k,val in b.items():
            if isinstance(val,str):b[k]=clean(val)
        kind=b['kind'].lower()
        b['kind']='insufficiently tested' if kind in ['evaluation','evidence quality','assumption and insufficiently tested'] else 'implementation-specific' if kind=='implementation and evaluation' else 'methodological' if kind=='methodological and insufficiently tested' else kind

for file in ['manifest.json','extra_manifest.json','zip_manifest.json','closure_manifest.json']:
    for item in json.loads((ROOT/'evidence/04/supplements'/file).read_text(encoding='utf8')):
        pid=item.get('paper_id') or {'vsi_supp':'vsi_bench','explore_confident_full':'explore_confident'}.get(item.get('id'))
        if pid not in n or item.get('error') or item.get('status')==404:continue
        if not any(s['url']==item['url'] for s in n[pid].get('sources',[])):
            n[pid].setdefault('sources',[]).append({'url':item['url'],'locator':'Official supplementary or complete author version; reviewed scope listed in associated material'})

for pid,v in n.items():
    repo_path=ROOT/'repositories/04'/pid/'source.json'
    if repo_path.exists():
        repo=json.loads(repo_path.read_text(encoding='utf8'))
        for f in v.get('findings',[]):
            if f['evidence_type']=='INSPECTED CODE':f['source']=repo.get('repo_url','')+'/tree/'+repo.get('commit','')
    if pid=='environment_review':
        for f in v['findings']:f['source']='Supplied internal environment review; primary-source status not established'

NOTES.write_text(json.dumps(n,ensure_ascii=False,indent=2),encoding='utf8')
render()

results=json.loads((ROOT/'assignments/04_results.json').read_text(encoding='utf8'))
required={'paper_id','short_name','full_title','authors','year','venue_or_status','bibliography_sources','main_pdf_pages_read','associated_files_reviewed','repo_url','repo_commit','code_files_inspected','code_executed','repository_status','dossier_path','key_findings','bottlenecks','followup_sources','access_limitations','completion_status'}
errors=[]
if len(results)!=33 or len({r['paper_id'] for r in results})!=33:errors.append('Inventory count or uniqueness failed')
allowed={'Resolved within a clearly specified setting','Partially addressed','Shifted into another cost or limitation','Still unresolved in the reviewed literature','Uncertain because evidence is insufficient'}
for r in results:
    pid=r['paper_id']
    if required-set(r):errors.append(pid+': missing schema keys')
    if not r['key_findings'] or not r['bottlenecks']:errors.append(pid+': empty evidence or bottleneck record')
    if not r['authors'] and pid!='environment_review':errors.append(pid+': no authors')
    if r['code_executed']:errors.append(pid+': fetched-code execution incorrectly flagged')
    if r['code_files_inspected'] and not r['repo_commit']:errors.append(pid+': source inspection lacks commit')
    for b in r['bottlenecks']:
        if b['status'] not in allowed:errors.append(pid+': noncanonical bottleneck status')
    text=(ROOT/r['dossier_path']).read_text(encoding='utf8')
    if re.search(r'pending|in progress|will be incorporated|exact title to verify',text,re.I):errors.append(pid+': stale completion prose')
    if len(text.split())<400:errors.append(pid+': dossier too short for substantive review')
out={'date':'2026-10-22','entries':len(results),'unique_ids':len({r['paper_id'] for r in results}),'dossiers_present':sum((ROOT/r['dossier_path']).exists() for r in results),'bottleneck_count':sum(len(r['bottlenecks']) for r in results),'pinned_repositories':sum(bool(r['repo_commit']) for r in results),'records_with_inspected_source':sum(bool(r['code_files_inspected']) for r in results),'errors':errors,'scope':'Artifact consistency validation only; no model or navigation result established.'}
(ROOT/'evidence/04/validation.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
