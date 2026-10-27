"""Retrieve public primary PDFs as data and extract page-labelled text; never run project code."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request
import fitz

ROOT = Path(__file__).resolve().parent
SOURCES = {
 'law': 'https://aclanthology.org/2021.emnlp-main.328.pdf',
 'himeco': 'https://arxiv.org/pdf/2610.20312v1',
 'doctor': 'https://arxiv.org/pdf/2608.28264v1',
 'capability_vln': 'https://arxiv.org/pdf/2604.25161v1',
 'cvs': 'https://arxiv.org/pdf/2603.09715v2',
 'psr': 'https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf',
 'dispo': 'https://arxiv.org/pdf/2403.06328',
 'distributional_sm': 'https://arxiv.org/pdf/2402.08530',
 'reflect': 'https://arxiv.org/pdf/2306.15724',
 'cartography': 'https://aclanthology.org/2020.emnlp-main.746.pdf',
}

def fetch(item):
    name, url = item
    try:
        dest = ROOT / 'sources' / (name + '.pdf')
        if not dest.exists():
            req = urllib.request.Request(url, headers={'User-Agent':'Research evidence review contact via public source'})
            with urllib.request.urlopen(req, timeout=50) as r: data = r.read()
            if not data.startswith(b'%PDF'): raise ValueError('Response is not a PDF')
            dest.write_bytes(data)
        data = dest.read_bytes()
        doc = fitz.open(stream=data, filetype='pdf')
        (dest.with_suffix('.txt')).write_text('\n\n'.join(f'===== PDF PAGE {i+1} =====\n'+p.get_text(sort=True) for i,p in enumerate(doc)),encoding='utf-8')
        return {'id':name,'url':url,'pdf':str(dest.relative_to(ROOT)),'pages':len(doc),'sha256':hashlib.sha256(data).hexdigest(),'access':'retrieved; review scope recorded separately','retrieved':'2026-10-22'}
    except Exception as e:
        return {'id':name,'url':url,'error':str(e)}

(ROOT/'sources').mkdir(parents=True,exist_ok=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(fetch,SOURCES.items()))
(ROOT/'retrieval_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
