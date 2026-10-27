import pathlib,requests,fitz,hashlib,json,concurrent.futures
dst=pathlib.Path(__file__).resolve().parent/'sources'
items={'mtrack':'https://arxiv.org/pdf/2202.07028','landmark_rxr':'https://papers.nips.cc/paper/2021/file/0602940f23884f782058efac46f64b0f-Paper.pdf','marky':'https://arxiv.org/pdf/2111.12872'}
def run(kv):
 k,url=kv
 try:
  data=requests.get(url,timeout=60).content
  if not data.startswith(b'%PDF'): raise ValueError('Response not PDF')
  doc=fitz.open(stream=data,filetype='pdf');(dst/f'{k}.pdf').write_bytes(data)
  (dst/f'{k}.txt').write_text('\n'.join(f'=== PDF PAGE {i+1} ===\n'+p.get_text(sort=True) for i,p in enumerate(doc)),encoding='utf-8')
  m=dict(id=k,url=url,pages=len(doc),sha256=hashlib.sha256(data).hexdigest(),retrieved='2026-10-22',status='Downloaded; review pending')
  (dst/f'{k}.json').write_text(json.dumps(m,indent=2),encoding='utf-8');return m
 except Exception as e:return dict(id=k,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for result in ex.map(run,items.items()): print(json.dumps(result))
