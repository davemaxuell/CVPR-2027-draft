import concurrent.futures,json,subprocess,requests,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=json.loads((ROOT/'evidence/02/repos.json').read_text())
base=ROOT/'repositories/02';base.mkdir(exist_ok=True,parents=True)
def fetch(item):
 pid,repo=item;dest=base/pid;dest.mkdir(exist_ok=True)
 try:
  ref=subprocess.run(['git','ls-remote','https://github.com/'+repo+'.git','HEAD'],capture_output=True,text=True,timeout=90)
  if ref.returncode or not ref.stdout:raise RuntimeError(ref.stderr[:250])
  sha=ref.stdout.split()[0]
  r=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1',timeout=60);r.raise_for_status();tree=r.json()
  (dest/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
  paths=[x['path'] for x in tree.get('tree',[]) if x['type']=='blob' and x['path'].lower().endswith(('.py','.sh','.bash','.yaml','.yml','.json','.md')) and x.get('size',0)<250000]
  # Source snapshots are fetched, but only files named in dossier are counted as inspected.
  picked=paths if len(paths)<95 else sorted(paths,key=lambda x:(-sum(w in x.lower() for w in ['model','policy','train','loss','agent','planner','world','config','readme','eval']),len(x)))[:95]
  files=[]
  for path in picked:
   u=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
   rr=requests.get(u,timeout=40)
   if rr.status_code==200:
    out=dest/'source'/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(rr.content);files.append(path)
  rec=dict(paper_id=pid,url='https://github.com/'+repo,commit=sha,retrieval_date='2026-10-22',files_fetched=files,all_source_paths=paths,status='Retrieved; source inspection recorded separately')
 except Exception as e:rec=dict(paper_id=pid,url='https://github.com/'+repo,status='Unavailable',error=str(e))
 (dest/'snapshot.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
 print(pid,rec['status'],len(rec.get('files_fetched',[])),flush=True)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 records=list(pool.map(fetch,spec.items()))
(ROOT/'evidence/02/repository_snapshots.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
