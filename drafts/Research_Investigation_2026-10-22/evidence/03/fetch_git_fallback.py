import json,subprocess,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def work(ident):
 folder=ROOT/'repositories'/'03'/ident; rec=json.loads((folder/'snapshot.json').read_text())
 clone=folder/'git_snapshot'
 try:
  if not (clone/'.git').exists(): subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout',rec['repo_url']+'.git',str(clone)],check=True,timeout=180,capture_output=True)
  sha=subprocess.check_output(['git','-C',str(clone),'rev-parse','HEAD'],text=True).strip(); rec['commit']=sha
  paths=subprocess.check_output(['git','-C',str(clone),'ls-tree','-r','--name-only','HEAD'],text=True).splitlines()
  paths=[p for p in paths if p.lower().endswith(('.py','.sh','.bash','.yaml','.yml','.md','.json')) and not any(v in p for v in ['third_party/','assets/','data/','habitat-lab/','habitat-sim/'])]
  (folder/'paths.txt').write_text('\n'.join(paths),encoding='utf-8')
  def score(p):return (-sum(w for x,w in [('readme',80),('model',20),('agent',25),('moe',30),('router',30),('train',20),('eval',10),('loss',20),('stream',20),('memory',20)] if x in p.lower()),len(p))
  selected=sorted(paths,key=score)[:40]
  subprocess.run(['git','-C',str(clone),'sparse-checkout','set','--no-cone','--stdin'],input='\n'.join('/'+p for p in selected),text=True,check=True,timeout=120,capture_output=True)
  subprocess.run(['git','-C',str(clone),'checkout'],check=True,timeout=180,capture_output=True)
  rec['downloaded']=[]
  for p in selected:
   src=clone/p
   if src.is_file():
    dst=folder/p;dst.parent.mkdir(exist_ok=True,parents=True);dst.write_bytes(src.read_bytes());rec['downloaded'].append(p)
  rec['status']='Commit-pinned source retrieved through sparse git checkout';rec.pop('error',None)
 except Exception as e:rec['fallback_error']=str(e)
 (folder/'snapshot.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(ident,rec['status'],len(rec.get('downloaded',[])),rec.get('fallback_error',''),flush=True)
if __name__=='__main__':
 import sys
 with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(work,sys.argv[1:] or ['rl4vla','skillnav','streamvln','susa','uninavid']))
