import requests,fitz,pathlib,json,hashlib,concurrent.futures
root=pathlib.Path(__file__).resolve().parents[2]; dst=root/'evidence'/'05'/'sources';dst.mkdir(exist_ok=True)
items={'seekvln':'https://arxiv.org/pdf/2609.37353v1','cag':'https://arxiv.org/pdf/2602.17659v2','behavior':'https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Behavioral_Analysis_of_Vision-and-Language_Navigation_Agents_CVPR_2023_paper.pdf','stay_path':'https://aclanthology.org/P19-1181.pdf','ccc':'https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Counterfactual_Cycle-Consistent_Learning_for_Instruction_Following_and_Generation_in_Vision-Language_CVPR_2022_paper.pdf'}
def run(kv):
 k,url=kv
 try:
  r=requests.get(url,timeout=90);r.raise_for_status();data=r.content
  if not data.startswith(b'%PDF'):raise ValueError('Not PDF')
  (dst/f'{k}.pdf').write_bytes(data);doc=fitz.open(stream=data,filetype='pdf');(dst/f'{k}.txt').write_text('\n'.join(f'=== PDF PAGE {i+1} ===\n'+p.get_text(sort=True) for i,p in enumerate(doc)),encoding='utf-8')
  m={'id':k,'url':url,'pages':len(doc),'sha256':hashlib.sha256(data).hexdigest(),'retrieved':'2026-10-22','status':'Downloaded; review pending'};(dst/f'{k}.json').write_text(json.dumps(m,indent=2),encoding='utf-8');return m
 except Exception as e:return {'id':k,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
 for x in e.map(run,items.items()): print(json.dumps(x))
