from pathlib import Path
import subprocess, requests, json, concurrent.futures,sys,datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'repositories/04'
REPOS={'vln_trans':'HLR/VLN-trans','gela':'CSir1996/VLN-GELA','navnuances':'zehao-wang/navnuances','activevig':'HanqingWangAI/Active_VLN','fstta':'Feliciaxyao/ICML2024-FSTTA','babywalk':'Sha-Lab/babywalk','gsavln':'honghd16/GSA-VLN','same':'GengzeZhou/SAME','omniagent':'harryhsing/OmniAgent','uot':'zhiyuanhubj/UoT','vlnmme':'billzhao1030/VLN-MME','walkreadless':'wdqin/VLN-NAP'}
def fetch(item):
    pid,repo=item;dest=OUT/pid;dest.mkdir(parents=True,exist_ok=True)
    record={'paper_id':pid,'repo_url':'https://github.com/'+repo,'retrieved':'2026-10-22','executed':False}
    try:
        res=subprocess.run(['git','ls-remote',record['repo_url'],'HEAD'],capture_output=True,text=True,timeout=60,env={**__import__('os').environ,'GIT_TERMINAL_PROMPT':'0'})
        if not res.stdout.strip(): raise RuntimeError(res.stderr[:500])
        sha=res.stdout.split()[0];record['commit']=sha
        tree=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1',timeout=45)
        if tree.status_code==200:
            tree=tree.json();paths=[x['path'] for x in tree.get('tree',[]) if x['type']=='blob']
        else:
            bare=dest/'git_metadata'
            if not bare.exists():
                subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout',record['repo_url'],str(bare)],capture_output=True,text=True,timeout=180,check=True)
            paths=subprocess.run(['git','-C',str(bare),'ls-tree','-r','--name-only','HEAD'],capture_output=True,text=True,check=True).stdout.splitlines()
            sha=subprocess.run(['git','-C',str(bare),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip();record['commit']=sha
        record['paths']=paths
        for p in paths:
            if '/' not in p and p.lower().startswith('readme'):
                r=requests.get(f'https://raw.githubusercontent.com/{repo}/{sha}/{p}',timeout=40)
                if r.status_code==200:(dest/p).write_text(r.text,encoding='utf-8')
        record['status']='Repository tree and README retrieved. Source inspection recorded separately.'
    except Exception as e:record['error']=str(e);record['status']='Access limited; not evidence of unavailable code.'
    (dest/'source.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return {k:record.get(k) for k in ['paper_id','commit','status','error']}
if __name__=='__main__':
    jobs=list(REPOS.items()) if len(sys.argv)==1 else [(p,REPOS[p]) for p in sys.argv[1:]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for rec in ex.map(fetch,jobs):print(json.dumps(rec),flush=True)
