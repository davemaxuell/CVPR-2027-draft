import json,sys,urllib.request,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[2]
def fetch(arg):
 repo,_,path=arg.partition('/')
 base=R/'repositories/02'/repo
 meta=json.loads((base/'snapshot.json').read_text(encoding='utf-8'))
 target=base/'source'/path
 if target.exists(): return arg+' cached'
 url=meta['url'].replace('https://github.com/','https://raw.githubusercontent.com/').rstrip('/')+'/'+meta['commit']+'/'+path
 try:
  data=urllib.request.urlopen(url,timeout=45).read();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  return arg+' fetched'
 except Exception as e:return arg+' ERROR '+str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for result in pool.map(fetch,sys.argv[1:]):print(result)
