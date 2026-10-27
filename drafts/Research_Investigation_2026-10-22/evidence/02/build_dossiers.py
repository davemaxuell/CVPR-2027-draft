import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
papers=json.loads((ROOT/'assignments/02_papers.json').read_text(encoding='utf-8-sig'))
notes=json.loads((ROOT/'evidence/02/notes.json').read_text(encoding='utf-8'))
for extra in sorted((ROOT/'evidence/02').glob('notes_extra*.json')):
 notes.update(json.loads(extra.read_text(encoding='utf-8')))
codes=json.loads((ROOT/'evidence/02/code_notes.json').read_text(encoding='utf-8')) if (ROOT/'evidence/02/code_notes.json').exists() else {}
records=[]
for p in papers:
 if p['id'] not in notes:continue
 n=notes[p['id']];code=codes.get(p['id'],{})
 snapshot=ROOT/'repositories/02'/code.get('snapshot_id',p['id'])/'snapshot.json'
 repo=json.loads(snapshot.read_text(encoding='utf-8')) if snapshot.exists() else {}
 pages=[]
 for r in n['pages'].split(','):
  a,_,b=r.partition('-');pages+=list(range(int(a),int(b or a)+1))
 source=p.get('source') or n.get('source') or 'Supplied PDF; see local file and SHA256 in corpus metadata'
 assoc=n.get('associated',[])
 sourcetext='\n'.join('- '+f['filename']+'; '+f['role']+'; SHA256 `'+f['sha256']+'`; '+str(f['pages'])+' pages.' for f in p['local_files'])
 doc=f'''# {p['title']}

## Bibliography and reviewed material

**{n['title']}**. {n['authors']}. {n['venue']} ({n['year']}). Research cutoff: **2026-10-22**.

Primary publication/source: {source}. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: {n['pages']}. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: {', '.join(assoc) or 'No separately supplied associated file; included appendices are identified by the page list.'}

{sourcetext}

## Problem, task and assumptions

{n['problem']}

## Method, training and inference

{n['mechanism']}

## Novelty and closest predecessors

{n['novelty']}

## Results and independent verdict

{n['results']}

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: {repo.get('url',n.get('repo','No verified official implementation located in the recorded search.'))}. Snapshot commit: `{repo.get('commit','not available')}`. Retrieval: 2026-10-22.

{code.get('text','Repository search/snapshot retrieval is in progress. No model-code inspection or execution is claimed in this interim dossier.')}

Inspected files: {', '.join('`'+x+'`' for x in code.get('files',[])) or 'No implementation files inspected; access scope is stated above'}. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

{n['bottleneck']}

Relevant follow-up relationship: {n['followups']}

## Access and uncertainty record

Full text is preserved under `corpus/{p['id']}/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/{code.get('snapshot_id',p['id'])}/`. {n.get('access','Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.')}
'''
 (ROOT/'dossiers'/f"{p['id']}.md").write_text(doc,encoding='utf-8')
 findings=[dict(claim=n['mechanism'],evidence_type='AUTHOR CLAIM',source=source,locator='PDF pp.'+n['pages']),dict(claim=n['results'],evidence_type='PAPER EXPERIMENT',source=source,locator='Tables/figures identified in claim'),dict(claim=n['bottleneck'],evidence_type='DEDUCTION',source=source,locator='Method and evaluation identified in dossier')]
 if code:findings.append(dict(claim=code['text'],evidence_type='INSPECTED CODE' if code.get('files') else 'UNVERIFIED',source=repo.get('url'),locator=repo.get('commit')))
 records.append(dict(paper_id=p['id'],short_name=p['title'],full_title=n['title'],authors=n['authors'].split('; '),year=n['year'],venue_or_status=n['venue'],bibliography_sources=[dict(url=source,locator='PDF first page and publication source')],main_pdf_pages_read=pages,associated_files_reviewed=assoc,repo_url=repo.get('url',n.get('repo')),repo_commit=repo.get('commit'),code_files_inspected=code.get('files',[]),code_executed=False,repository_status=code.get('status','Retrieval/inspection in progress'),dossier_path='dossiers/'+p['id']+'.md',key_findings=findings,bottlenecks=[dict(id='02-'+p['id'],description=n['bottleneck'],conditions=n['problem'],cause='See method-specific explanation in dossier',evidence='Primary PDF sections/pages cited in dossier',kind='Methodological or insufficiently tested, as specified',followups=n['followups'],status='Partially addressed or uncertain; scoped assessment in dossier',confidence='Medium unless explicitly qualified')],followup_sources=n.get('followup_sources',[]),access_limitations=[n.get('access','No benchmark execution; source gaps explicitly recorded'),code.get('status','Code inspection pending')],completion_status='Paper review complete; code audit recorded' if code else 'Paper review complete; code audit pending'))
(ROOT/'assignments/02_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print('Dossiers and result records:',len(records),'of',len(papers))
if (ROOT/'evidence/02/finalize_schema.py').exists():
 import runpy
 runpy.run_path(str(ROOT/'evidence/02/finalize_schema.py'),run_name='__main__')
