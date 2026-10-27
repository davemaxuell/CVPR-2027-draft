from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]
PAPERS={p['id']:p for p in json.loads((ROOT/'assignments/01_papers.json').read_text('utf-8-sig'))}
NOTES=ROOT/'evidence/01/notes.json'
def add(id,**kwargs):
 notes=json.loads(NOTES.read_text('utf-8')) if NOTES.exists() else {}
 notes[id]=kwargs; NOTES.write_text(json.dumps(notes,ensure_ascii=False,indent=2),'utf-8');build()
def build():
 notes=json.loads(NOTES.read_text('utf-8')); results=[]
 for id,n in notes.items():
  p=PAPERS[id]; source=p.get('source_override') or p.get('source') or n.get('source','')
  repo_path=ROOT/'repositories/01'/id/'source.json'
  repo=json.loads(repo_path.read_text('utf-8')) if repo_path.exists() else {}
  authors=n.get('authors') or [a.strip() for a in p['local_files'][0].get('authors_metadata','').split(';') if a.strip()]
  if len(authors)==1 and ',' in authors[0]:authors=[a.strip() for a in authors[0].split(',')]
  title=n.get('title') or p['local_files'][0].get('title_metadata') or p['full_title']
  code=n.get('code',[]); read=n.get('pages',[])
  bibliography=f"# {p['title']}\n\nResearch cutoff: **2026-10-22**. Assignment 01.\n\n## 1. Bibliography, versions, and evidence scope\n\n**{title}**. {', '.join(authors) if authors else 'Author names not fully verified in this record'}. {n.get('venue',p.get('venue') or 'Publication status requires verification')}.\n\n"
  bibliography+='Primary paper: '+(f'[{title}]({source})' if source else 'local PDF; see inventory')+'. Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: '+str(read)+'.\n\n'
  bibliography+='Associated files: '+ '; '.join(f"{f['filename']} ({f['role']}; {f['pages']} pages; SHA256 {f['sha256']})" for f in p['local_files'])+'.\n\n'
  bibliography+=n.get('versions','The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.')+'\n\n'
  text=bibliography
  for heading,key in [('2. Problem and operating assumptions','problem'),('3. Mechanism, training, and inference','method'),('4. Novelty and evidence','novelty'),('5. Results and independent verdict','results')]:text+=f'## {heading}\n\n{n[key]}\n\n'
  text+='## 6. Implementation and reproducibility\n\n'
  if repo.get('commit'):text+=f"Official repository lead: [{repo['repo_url']}]({repo['repo_url']}); inspected snapshot **{repo['commit']}**. Snapshot metadata and source are under `repositories/01/{id}`.\n\n"
  text+=n.get('implementation','No verified public model implementation was located in the documented search scope; no source execution or reproduction claim is made.')+'\n\n'
  if code:text+='Inspected code locators:\n\n'+'\n'.join(f"- [{c['path']}]({c.get('repo_url',repo.get('repo_url',''))}/blob/{c.get('commit',repo.get('commit',''))}/{c['path']}) — {c['locator']}: {c['finding']}" for c in code)+'\n\n'
  text+='**Execution status:** '+('The source files listed above were inspected but not executed. ' if code else 'No model source inspection is claimed beyond the explicit scope above. ')+'No benchmark reproduction or checkpoint validation was performed.\n\n## 7. Bottlenecks and follow-up status\n\n'
  bottlenecks=[]
  for i,b in enumerate(n['bottlenecks'],1):
   if isinstance(b,str):b={'description':b,'evidence':'DEDUCTION from method/experiment scope above','kind':'methodological','status':'Partially addressed','followups':[],'confidence':'medium'}
   b={'id':f'{id}-B{i}','conditions':b.get('conditions','See description'),'cause':b.get('cause','See description'),**b};bottlenecks.append(b)
   text+=f"- **{b['id']} — {b.get('status','Uncertain')}**. {b['description']} Evidence: {b.get('evidence','See method/results above')}. Type: {b.get('kind','insufficiently tested')}. Follow-ups: {', '.join(b.get('followups',[])) or 'No resolving follow-up verified in this assignment; coordinator follow-up audit pending'}. Confidence: {b.get('confidence','medium')}.\n"
  text+='\n## 8. Access limitations and source record\n\n'+n.get('limitations','No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.')+'\n\n'
  text+='Related primary sources: '+ '; '.join(n.get('followups',[]))+'\n'
  (ROOT/'dossiers'/f'{id}.md').write_text(text,'utf-8')
  results.append(dict(paper_id=id,short_name=p['title'],full_title=title,authors=authors,year=n.get('year'),venue_or_status=n.get('venue',p.get('venue') or 'Unverified'),bibliography_sources=[{'url':source,'locator':'PDF p1 and cited method/results pages'}],main_pdf_pages_read=read,associated_files_reviewed=n.get('associated',[]),repo_url=repo.get('repo_url'),repo_commit=repo.get('commit'),code_files_inspected=code,code_executed=False,repository_status=n.get('repository_status','Partial source inspection' if code else 'No model code inspected'),dossier_path=f'dossiers/{id}.md',key_findings=n.get('findings',[{'claim':n['novelty'],'evidence_type':'PAPER EXPERIMENT / DEDUCTION as labeled','source':source,'locator':str(read)}]),bottlenecks=bottlenecks,followup_sources=n.get('followups',[]),access_limitations=[n.get('limitations','No benchmark executed')],completion_status='Dossier complete within explicitly documented evidence/access scope'))
 (ROOT/'assignments/01_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),'utf-8')
if __name__=='__main__':build()
