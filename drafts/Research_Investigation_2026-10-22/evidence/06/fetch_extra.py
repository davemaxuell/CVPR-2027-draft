from pathlib import Path
import concurrent.futures, hashlib, json, requests, fitz
ROOT=Path(__file__).resolve().parents[2]
SOURCES={
 'reinsvln':'https://arxiv.org/pdf/2610.26545v1',
 'dartvln':'https://arxiv.org/pdf/2607.01043v4',
 'cmmrvln':'https://arxiv.org/pdf/2603.07997',
 'imaginenavpp':'https://arxiv.org/pdf/2512.17435',
 'schrodinger_nav':'https://arxiv.org/pdf/2512.21201',
}
def fetch(pair):
 name,url=pair; dest=ROOT/'evidence'/'06'/'papers'; dest.mkdir(exist_ok=True)
 pdf=dest/(name+'.pdf'); txt=dest/(name+'.txt')
 try:
  if not pdf.exists():
   r=requests.get(url,timeout=90); r.raise_for_status(); pdf.write_bytes(r.content)
  doc=fitz.open(pdf)
  txt.write_text('\n'.join(f'=== PDF PAGE {i+1} ===\n'+page.get_text(sort=True) for i,page in enumerate(doc)),encoding='utf-8')
  return dict(id=name,url=url,pdf=str(pdf.relative_to(ROOT)),text=str(txt.relative_to(ROOT)),pages=len(doc),sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),retrieved='2026-10-22',status='Downloaded; not yet reviewed')
 except Exception as exc:return dict(id=name,url=url,status='Unavailable',error=str(exc))
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(fetch,SOURCES.items()))
 (ROOT/'evidence'/'06'/'extra_downloads.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); print(json.dumps(rows))
