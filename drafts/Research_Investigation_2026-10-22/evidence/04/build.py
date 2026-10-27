import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
NOTES=ROOT/'evidence/04/notes.json'
def put(pid,**fields):
    notes=json.loads(NOTES.read_text(encoding='utf-8')) if NOTES.exists() else {}
    notes.setdefault(pid,{}).update(fields)
    NOTES.write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
def render():
    notes=json.loads(NOTES.read_text(encoding='utf-8')); inventory=json.loads((ROOT/'assignments/04_papers.json').read_text(encoding='utf-8'))
    results=[]
    for p in inventory:
        pid=p['id']
        if pid not in notes:continue
        n=notes[pid]; repo_path=ROOT/'repositories/04'/pid/'source.json'; repo=json.loads(repo_path.read_text(encoding='utf-8')) if repo_path.exists() else {}
        sources=n.get('sources',[])
        if p.get('source') and not any(s.get('url')==p['source'] for s in sources):sources=[{'url':p['source'],'locator':'Supplied main paper'}]+sources
        rec={'paper_id':pid,'short_name':p['title'],'full_title':n['title'],'authors':n.get('authors',[]),'year':n.get('year'),'venue_or_status':n.get('venue'),'bibliography_sources':sources,'main_pdf_pages_read':n.get('pages',[]),'associated_files_reviewed':n.get('associated',[]),'repo_url':repo.get('repo_url',n.get('repo_url')),'repo_commit':repo.get('commit'),'code_files_inspected':n.get('code_files',[]),'code_executed':False,'repository_status':n.get('repo_status','Repository audit in progress'),'dossier_path':'dossiers/'+pid+'.md','key_findings':n.get('findings',[]),'bottlenecks':n.get('bottlenecks',[]),'followup_sources':n.get('followups',[]),'access_limitations':n.get('limits',[]),'completion_status':n.get('status','Paper reviewed; implementation audit in progress')}
        results.append(rec)
        lines=['# '+p['title'],'',n['title'],'',f"{'; '.join(n.get('authors',[])) or 'Authorship not established'}. {n.get('venue','Publication status not verified')}. Investigation cutoff: 2026-10-22.",'', '## Materials and evidence scope','',n.get('scope','The supplied PDF was read using its page-labelled full-text extraction. Figures are interpreted only to the extent stated below. No navigation benchmark was executed.'),'',f"Main PDF: [{p['filename']}](../../Paper/{p['filename'].replace(' ','%20')}). Pages reviewed: {n.get('pages',[])}.",'']
        for title,field in [('Problem and operating conditions','problem'),('Mechanism, training, and inference','method'),('Novelty and closest predecessors','novelty'),('Results and independent verdict','results'),('Implementation and reproducibility','implementation')]:lines.extend(['## '+title,'',n.get(field,'Audit in progress.'),''])
        if repo.get('commit'):lines.extend([f"Inspected release reference: [{repo['repo_url']} at {repo['commit']}]({repo['repo_url']}/tree/{repo['commit']}). File-level inspection is restricted to the paths listed below.",''])
        for f in n.get('code_files',[]):lines.extend(['- '+(f if isinstance(f,str) else str(f))])
        lines.extend(['','## Specific bottlenecks',''])
        for b in n.get('bottlenecks',[]):
            lines.extend([f"**{b['id']}: {b['description']}**",'',f"Conditions: {b['conditions']} Cause: {b['cause']} Evidence: {b['evidence']}",'',f"Type: {b['kind']}. Status: {b['status']}. Confidence: {b['confidence']}. Follow-ups: {b['followups']}",''])
        lines.extend(['## Sources and limitations',''])
        for s in sources:lines.append('- ['+s.get('locator','Primary source')+']('+s['url']+')')
        for l in n.get('limits',[]):lines.append('- '+l)
        for a in n.get('associated',[]):lines.append('- Associated material: '+(a if isinstance(a,str) else json.dumps(a,ensure_ascii=False)))
        (ROOT/'dossiers'/f'{pid}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'assignments/04_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rendered',len(results),'dossiers.')
if __name__=='__main__':render()
