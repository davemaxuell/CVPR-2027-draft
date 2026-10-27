import concurrent.futures,json,subprocess,requests
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
repos=json.loads((ROOT/'evidence/02/repos.json').read_text())
def run(args):return subprocess.run(args,capture_output=True,text=True,timeout=180)
def fetch(item):
 pid,repo=item;dest=ROOT/'repositories/02'/pid
 existing=dest/'snapshot.json'
 if existing.exists() and json.loads(existing.read_text()).get('commit'):return
 dest.mkdir(parents=True,exist_ok=True);clone=dest/'_git'
 try:
  if not clone.exists():
   r=run(['git','clone','--depth','1','--filter=blob:none','--no-checkout','https://github.com/'+repo+'.git',str(clone)])
   if r.returncode:raise RuntimeError(r.stderr[:500])
  sha=run(['git','-C',str(clone),'rev-parse','HEAD']).stdout.strip()
  paths=run(['git','-C',str(clone),'ls-tree','-r','--name-only','HEAD']).stdout.splitlines()
  paths=[p for p in paths if p.endswith(('.py','.yaml','.yml','.sh','.bash','.md','.json'))]
  priority=['model','world','policy','train','loss','agent','planner','readme','config','eval']
  picked=sorted(paths,key=lambda p:(-sum(w in p.lower() for w in priority),len(p)))[:24]
  files=[]
  for path in picked:
   rr=requests.get(f'https://raw.githubusercontent.com/{repo}/{sha}/{path}',timeout=40)
   if rr.status_code==200 and len(rr.content)<600000:
    out=dest/'source'/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(rr.content);files.append(path)
  rec=dict(paper_id=pid,url='https://github.com/'+repo,commit=sha,retrieval_date='2026-10-22',files_fetched=files,all_source_paths=paths,status='Retrieved through git refs/raw fallback; source inspection recorded separately')
 except Exception as e:rec=dict(paper_id=pid,url='https://github.com/'+repo,status='Access failed, not evidence of no release',error=str(e))
 existing.write_text(json.dumps(rec,indent=2),encoding='utf-8');print(pid,rec['status'],len(rec.get('files_fetched',[])),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,repos.items()))
