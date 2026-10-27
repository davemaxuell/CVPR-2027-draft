"""Coordinator-independent pinned source acquisition; never imports fetched code."""
from pathlib import Path
import hashlib, json, urllib.request
ROOT = Path(__file__).resolve().parents[2]
def acquire():
    commit='181d878ff207845a31e542a19a6592925c1e9bab'
    paths=['utils/rag_manager.py','utils/goal_detection.py','server.py','configs/dataset_registry.py']
    out=[]
    for path in paths:
        url=f'https://raw.githubusercontent.com/Chenrf1121/psg-nav/{commit}/{path}'
        dest=ROOT/'repositories/10/psgnav'/path
        try:
            data=urllib.request.urlopen(url,timeout=40).read()
            dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
            out.append(dict(path=path,url=url,commit=commit,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),status='Retrieved, inspection separate'))
        except Exception as ex: out.append(dict(path=path,url=url,status=str(ex)))
    (ROOT/'evidence/10/acquisition.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__': acquire()
