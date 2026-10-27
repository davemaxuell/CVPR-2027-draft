from pathlib import Path
import requests,json,concurrent.futures,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'repositories/04'
def fetch(job):
    pid,path,meta=job;dest=OUT/pid/path
    if dest.exists():return {'paper_id':pid,'path':path,'cached':True}
    url=meta['repo_url'].replace('github.com','raw.githubusercontent.com')+'/'+meta['commit']+'/'+path
    try:
        r=requests.get(url,timeout=35);r.raise_for_status()
        if len(r.content)>1500000:return {'paper_id':pid,'path':path,'skipped':'source size exceeds1.5MB'}
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(r.text,encoding='utf-8')
        return {'paper_id':pid,'path':path,'url':url,'bytes':len(r.content)}
    except Exception as e:return {'paper_id':pid,'path':path,'error':str(e)}
if __name__=='__main__':
    jobs=[]
    for pid in sys.argv[1:]:
        meta=json.loads((OUT/pid/'source.json').read_text(encoding='utf-8'))
        for path in meta.get('paths',[]):
            if not path.endswith(('.py','.yaml','.yml','.sh','.bash','.md')):continue
            if any(part in path.split('/') for part in ['CLIP','transformers','third_party','third-party','.github','build','external','site-packages']):continue
            jobs.append((pid,path,meta))
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:results=list(ex.map(fetch,jobs))
    (ROOT/'evidence/04'/('downloads_'+'_'.join(sys.argv[1:])+'.json')).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(results),'errors':[r for r in results if 'error'in r]}),flush=True)
