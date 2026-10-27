from pathlib import Path
import subprocess,requests,json,concurrent.futures,re
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'repositories/01'
REPOS={
'duet':'cshizhe/VLN-DUET','scalevln':'wz0919/ScaleVLN','srdf':'wz0919/VLN-SRDF',
'gaussianmap':'Gaozzzz/3D-Gaussian-Map-VLN','uagm':'Gaozzzz/Uncertainty-Aware-VLN',
'bevbert':'MarSaKi/VLN-BEVBert','gridmm':'MrZihan/GridMM','kerm':'XiangyangLi20/KERM','ver':'DefaultRui/VLN-VER',
'goat':'CrystalSixone/VLN-GOAT','gavln':'jahhaoyang/GA-VLN','vlnbert':'YicongHong/Recurrent-VLN-BERT',
'hamt':'cshizhe/VLN-HAMT','airbert':'airbert-vln/airbert','prevalent':'weituo12321/PREVALENT',
'navillm':'LaVi-Lab/NaviLLM','navrag':'MrZihan/NavRAG','crossmodalmap':'ggeorgak11/CM2',
'weakmap':'PeihaoChen/WS-MGMap','regretful':'chihyaoma/regretful-agent','selfmonitoring':'chihyaoma/selfmonitoring-agent',
'speakerfollower':'ronghanghu/speaker_follower','tacticalrewind':'Kelym/FAST','rxr':'google-research-datasets/RxR',
'r2r':'peteanderson80/Matterport3DSimulator','reverie':'YuankaiQi/REVERIE','ivln':'jacobkrantz/IVLN'
}
def fetch(item):
 id,repo=item; dest=OUT/id;dest.mkdir(parents=True,exist_ok=True)
 record={'paper_id':id,'repo_url':'https://github.com/'+repo,'retrieved':'2026-10-22','executed':False}
 try:
  cp=subprocess.run(['git','ls-remote',record['repo_url'],'HEAD'],capture_output=True,text=True,timeout=45)
  sha=cp.stdout.split()[0];record['commit']=sha
  resp=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1',timeout=40)
  resp.raise_for_status();tree=resp.json()
  (dest/'tree.json').write_text(json.dumps(tree,ensure_ascii=False,indent=2),'utf-8')
  paths=[x['path'] for x in tree.get('tree',[]) if x['type']=='blob' and x['path'].endswith(('.py','.yaml','.yml','.sh','.bash','.json','.md'))]
  record['paths']=paths
  readme=next((p for p in paths if p.lower()=='readme.md'),None)
  if readme:
   res=requests.get(f'https://raw.githubusercontent.com/{repo}/{sha}/{readme}',timeout=30);res.raise_for_status();(dest/'README.md').write_text(res.text,'utf-8')
  record['status']='tree and README fetched; model code pending inspection'
 except Exception as e:record['error']=str(e);record['status']='access failure'
 (dest/'source.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),'utf-8')
 return {'id':id,'commit':record.get('commit'),'paths':len(record.get('paths',[])),'error':record.get('error')}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  for result in ex.map(fetch,REPOS.items()):print(json.dumps(result),flush=True)
