import pathlib,requests,zipfile,io,json,fitz,hashlib
root=pathlib.Path(__file__).resolve().parents[2];out=root/'evidence/04/supplements'
jobs={'continualvln':'https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_84/supplementary.zip','ieve':'https://openaccess.thecvf.com/content/CVPR2024/supplemental/Lei_Instance-aware_Exploration-Verification-Exploitation_for_CVPR_2024_supplemental.zip'}
records=[]
for pid,url in jobs.items():
 rec=dict(paper_id=pid,url=url)
 try:
  r=requests.get(url,timeout=60,stream=True);r.raise_for_status();chunks=[];size=0
  for c in r.iter_content(1048576):
   size+=len(c)
   if size>100*1048576:raise RuntimeError('Supplement ZIP exceeds100MB bounded retrieval limit; PDF extraction unavailable from this archive request.')
   chunks.append(c)
  data=b''.join(chunks);z=zipfile.ZipFile(io.BytesIO(data));rec['archive_paths']=z.namelist();rec['files']=[]
  for name in z.namelist():
   if name.lower().endswith('.pdf') and not name.startswith('__MACOSX/'):
    b=z.read(name);doc=fitz.open(stream=b,filetype='pdf');dst=out/(pid+'_'+pathlib.Path(name).name);dst.write_bytes(b)
    dst.with_suffix('.txt').write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n'+p.get_text() for i,p in enumerate(doc)),encoding='utf8')
    rec['files'].append(dict(file=str(dst.relative_to(root)),pages=len(doc),sha256=hashlib.sha256(b).hexdigest()))
 except Exception as e:rec['error']=str(e)
 records.append(rec)
(out/'zip_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf8');print(json.dumps(records,indent=2))
