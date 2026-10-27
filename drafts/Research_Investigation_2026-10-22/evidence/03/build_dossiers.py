from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
papers={e['id']:e for e in json.loads((ROOT/'assignments'/'03_papers.json').read_text(encoding='utf-8'))}
records=[]
audit_path=ROOT/'evidence'/'03'/'code_audits.json'
audits=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
for path in sorted((ROOT/'evidence'/'03').glob('notes_*.json')):
 for n in json.loads(path.read_text(encoding='utf-8')):
  n.update(audits.get(n['id'],{}))
  e=papers[n['id']]; ident=n['id']; repo=ROOT/'repositories'/'03'/ident/'snapshot.json'
  r=json.loads(repo.read_text(encoding='utf-8')) if repo.exists() else {}
  inspected=n.get('code_files',[])
  source=n.get('source',e['source']) or 'Supplied PDF; external bibliography verification recorded below.'
  bibliography=n.get('bibliography','')
  txt=f"# {n.get('short_name',e['title'])}\n\nAssignment 03. Investigation cutoff: 2026-10-22.\n\n## Bibliography and reviewed material\n\n**{n.get('title',e['full_title'])}**. {', '.join(n.get('authors',[])) or 'Author list not independently verified; see supplied PDF page 1.'} {n.get('venue','Publication status not independently verified')} ({n.get('year','unknown')}).\n\nPrimary record: {source}\n\n{bibliography}\n\nSupplied main file: [{e['filename']}](../../Paper/{e['filename'].replace(' ','%20')}). Pages actually reviewed: {n['pages']}. {n.get('versions','Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.')}\n\n## Problem, motivation, and operating assumptions\n\n{n['problem']}\n\n## Method, training, inference, and equations\n\n{n['method']}\n\n## Novelty and closest predecessors\n\n{n['novelty']}\n\n## Results and independent verdict\n\n{n['results']}\n\n## Implementation audit\n\n"
  # Keep the concrete associated-version record visible in the dossier as well as JSON.
  if n.get('associated_files_reviewed'):
   block='\n\nAssociated material:\n\n'
   for material in n['associated_files_reviewed']:
    block+='- '+str(material.get('file'))+'; reviewed pages/status: '+str(material.get('pages'))
    if material.get('sha256'):block+='; SHA256 `'+material['sha256']+'`'
    if material.get('version'):block+='; version '+material['version']
    block+='\n'
   txt=txt.replace('\n\n## Problem, motivation, and operating assumptions',block+'\n## Problem, motivation, and operating assumptions')
  txt+=n.get('code','No executable implementation was verified; a retrieved snapshot alone is not a claim of inspection.')+'\n\n'
  if r:
   txt+=f"Repository: {r.get('repo_url')}. Inspected snapshot commit: `{r.get('commit','unavailable')}`. Retrieval status: {r.get('status')}. "
   if r.get('error'):txt+=r['error']+' '
   txt+='Project code was not executed.\n\n'
   for f in inspected:txt+=f"- [{f}](https://github.com/{r['repo_url'].split('github.com/')[-1]}/blob/{r.get('commit')}/{f.split(':')[0]})\n"
  txt+='\n## Bottlenecks and follow-up chain\n\n'
  for b in n['bottlenecks']:
   txt+=f"- **{b['id']}: {b['description']}** Conditions: {b['conditions']} Cause: {b['cause']} Evidence: {b['evidence']} Type: {b['kind']}. Follow-ups: {b.get('followups','Not established in scoped search')}. Status: **{b['status']}**, confidence {b['confidence']}.\n\n"
  txt+='## Source/access record and remaining uncertainty\n\n'+n.get('limitations','No benchmark experiment was executed. Paper claims and experimental results are not independent reproductions.')+'\n\n'
  for s in n.get('followup_sources',[]):txt+='- '+str(s)+'\n'
  out=ROOT/'dossiers'/(ident+'.md');out.parent.mkdir(exist_ok=True);out.write_text(txt,encoding='utf-8')
  records.append(dict(paper_id=ident,short_name=n.get('short_name',e['title']),full_title=n.get('title',e['full_title']),authors=n.get('authors',[]),year=n.get('year'),venue_or_status=n.get('venue'),bibliography_sources=[{'url':source,'locator':'Primary record and supplied PDF p.1'}],main_pdf_pages_read=n['pages'],associated_files_reviewed=n.get('associated_files_reviewed',[]),repo_url=r.get('repo_url',n.get('repo_url')),repo_commit=r.get('commit'),code_files_inspected=inspected,code_executed=False,repository_status=n.get('repository_status',r.get('status','Not verified')),dossier_path=str(out),key_findings=n.get('key_findings',[]),bottlenecks=n['bottlenecks'],followup_sources=n.get('followup_sources',[]),access_limitations=[n.get('limitations','No benchmark experiments executed')],completion_status=n.get('status','Paper review complete; implementation audit pending')))
(ROOT/'assignments'/'03_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated dossiers and records:',len(records))
