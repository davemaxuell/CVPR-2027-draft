from pathlib import Path
import requests,fitz,zipfile,io,json,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'evidence/04/supplements';OUT.mkdir(parents=True,exist_ok=True)
URLS={'gela':'https://openaccess.thecvf.com/content/ICCV2023/supplemental/Cui_Grounded_Entity-Landmark_Adaptive_ICCV_2023_supplemental.pdf','atena':'https://proceedings.nips.cc/paper_files/paper/2025/file/3f510f82323c1293ae3e343893dd77b1-Supplemental-Conference.zip'}
records=[]
for pid,url in URLS.items():
    r=requests.get(url,timeout=60);r.raise_for_status();data=r.content
    if data.startswith(b'%PDF'):files=[(pid+'.pdf',data)]
    else:
        z=zipfile.ZipFile(io.BytesIO(data));files=[(Path(n).name,z.read(n)) for n in z.namelist() if n.lower().endswith('.pdf') and not n.startswith('__MACOSX/')];print(pid,'archive paths',z.namelist())
    for filename,data in files:
        dest=OUT/(pid+'_'+filename);dest.write_bytes(data);doc=fitz.open(stream=data,filetype='pdf')
        text='\n\n'.join(f'=== PDF PAGE {i+1} ===\n'+p.get_text() for i,p in enumerate(doc))
        dest.with_suffix('.txt').write_text(text,encoding='utf-8');records.append({'paper_id':pid,'url':url,'file':str(dest.relative_to(ROOT)),'pages':len(doc),'sha256':hashlib.sha256(data).hexdigest(),'status':'Downloaded; reading recorded separately.'})
(OUT/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(records)
