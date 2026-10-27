import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
papers=json.loads((ROOT/'assignments/02_papers.json').read_text(encoding='utf-8-sig'))
out=[]
for p in papers:
 text=Path(p['local_files'][0]['text_path']).read_text(encoding='utf-8-sig')
 first=re.split(r'=== PDF PAGE 2 ===',text)[0]
 urls=sorted(set(re.findall(r'https?://[^\s<>]+',text)))
 entry=dict(id=p['id'],first=first[:2300],urls=urls,source=p.get('source'),metadata=p['local_files'][0].get('authors_metadata'))
 out.append(entry)
(ROOT/'evidence/02/discovery.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for p in out: print(p['id'],p['urls'])
