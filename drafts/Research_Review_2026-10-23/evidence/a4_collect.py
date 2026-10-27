from pathlib import Path
import urllib.request, json, hashlib, concurrent.futures
from html.parser import HTMLParser

OUT=Path(__file__).parent/'a4_sources'
OUT.mkdir(exist_ok=True)
SOURCES={
 'dreamwalker':'https://arxiv.org/html/2308.07498v1',
 'hnr':'https://arxiv.org/html/2404.01943v1',
 'dino_wm':'https://arxiv.org/html/2411.04983v2',
 'nwm':'https://arxiv.org/html/2412.03572v2',
 'streamvln':'https://arxiv.org/html/2507.05240v1',
 'vistav2':'https://arxiv.org/html/2512.00041v1',
 'seekvln':'https://arxiv.org/html/2609.37353v1',
 'futurenav':'https://arxiv.org/html/2606.30367v1',
 'cast':'https://arxiv.org/html/2508.13446v2',
 'worldmap':'https://arxiv.org/html/2604.07957v1',
 'dreamnav':'https://arxiv.org/html/2509.11197',
 'navmorph':'https://arxiv.org/html/2506.23468',
 'cvaml':'https://arxiv.org/html/2505.22772',
}
class TextParser(HTMLParser):
 def __init__(self):
  super().__init__(); self.out=[]; self.skip=0
 def handle_starttag(self,t,a):
  if t in ['script','style']: self.skip+=1
  if t in ['p','h1','h2','h3','h4','h5','h6','li','div','section','tr']: self.out.append('\n')
 def handle_endtag(self,t):
  if t in ['script','style']: self.skip-=1
  if t in ['p','h1','h2','h3','h4','h5','h6','li','div','section','tr']: self.out.append('\n')
 def handle_data(self,d):
  if not self.skip:self.out.append(d)
def get(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'ResearchReview/1.0'})
  with urllib.request.urlopen(req,timeout=45) as r: data=r.read(); final=r.url
  (OUT/(name+'.html')).write_bytes(data)
  p=TextParser();p.feed(data.decode('utf8'))
  text='\n'.join(x.strip() for x in ''.join(p.out).splitlines() if x.strip())
  (OUT/(name+'.txt')).write_text(text,encoding='utf8')
  return {'id':name,'url':url,'resolved_url':final,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'saved'}
 except Exception as e:return {'id':name,'url':url,'status':'error','error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex: rows=list(ex.map(get,SOURCES.items()))
 (OUT/'download_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(rows,ensure_ascii=False,indent=2))
