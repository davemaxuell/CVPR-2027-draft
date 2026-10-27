from pathlib import Path
import requests,json,subprocess,concurrent.futures,re,time
ROOT=Path(__file__).resolve().parents[2]
REPOS={
'adanav':'xinding-sys/AdaNav','analogical':'zhangyuejoslin/VLN-Analogical-Reasoning',
'aoplanner':'chen-judge/AO-Planner','auxthink':'HorizonRobotics/RoboOrchardLab',
'awarevln':'GWxuan/AwareVLN','ca_nav':'Chenkehan21/CA-Nav-code','console':'expectorlin/CONSOLE',
'discussnav':'LYX0501/DiscussNav','ecot':'MichalZawalski/embodied-CoT',
'gc_vln':'bagh2178/GC-VLN','mapgpt':'chen-judge/MapGPT','mapnav':'linglingxiansen/MapNav',
'navgpt':'GengzeZhou/NavGPT','navgpt2':'GengzeZhou/NavGPT-2',
'navgrpo':'Songlin-Dong/NavGRPO-CVPR2026','navid':'jzhzhang/NaVid-VLN-CE',
'navila':'AnjieCheng/NaVILA','polarmem':'czs-ict/PolarMem','psgnav':'Chenrf1121/psg-nav',
'r3':'IAII-CAS/navigation','rl4vla':'gen-robot/RL4VLA','skillnav':'HLR/SkillNav',
'streamvln':'InternRobotics/StreamVLN','susa':'HCI-LMC/VLN-SUSA','uninavid':'jzhzhang/Uni-NaVid'}
def fetch(kv):
 ident,repo=kv; folder=ROOT/'repositories'/'03'/ident; folder.mkdir(parents=True,exist_ok=True)
 rec={'id':ident,'repo_url':'https://github.com/'+repo,'retrieved':'2026-10-22','downloaded':[],'code_executed':False}
 try:
  sha=subprocess.check_output(['git','ls-remote',rec['repo_url']+'.git','HEAD'],text=True,timeout=45).split()[0]
  rec['commit']=sha
  response=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1',timeout=45); response.raise_for_status()
  tree=response.json(); (folder/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
  paths=[x['path'] for x in tree.get('tree',[]) if x['type']=='blob' and x.get('size',0)<400000 and x['path'].lower().endswith(('.py','.yaml','.yml','.sh','.bash','.md','.toml','.json')) and not any(z in x['path'].lower() for z in ['third_party/','3rdparty/','node_modules/','.github/','assets/','data/','habitat-lab/','habitat-sim/'])]
  (folder/'paths.txt').write_text('\n'.join(paths),encoding='utf-8')
  # Store a bounded review snapshot; downloaded does not mean inspected.
  def score(s):
   low=s.lower(); n=0
   for term,weight in [('readme',100),('aux_think',80),('progress_think',80),('model',15),('agent',18),('train',12),('loss',25),('policy',25),('eval',12),('config',4),('prompt',10),('grpo',22),('polar',22),('memory',22),('router',25),('moe/',20),('reason',15),('graph',15)]:
    if term in low:n+=weight
   return -n,len(s)
  selected=sorted(paths,key=score)[:35]
  for path in selected:
   url=f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
   rr=requests.get(url,timeout=30)
   if rr.status_code==200:
    dest=folder/path; dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(rr.content);rec['downloaded'].append(path)
  rec['status']='Source snapshot retrieved; file inspection tracked separately'
 except Exception as e:rec['status']='Access limitation';rec['error']=str(e)
 (folder/'snapshot.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
 print(ident,rec.get('commit'),len(rec['downloaded']),rec.get('error',''),flush=True)
 return rec
if __name__=='__main__':
 import sys
 ids=sys.argv[1:]
 chosen={k:v for k,v in REPOS.items() if not ids or k in ids}
 out=list(concurrent.futures.ThreadPoolExecutor(5).map(fetch,chosen.items()))
 (ROOT/'evidence'/'03'/'repository_retrieval.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
