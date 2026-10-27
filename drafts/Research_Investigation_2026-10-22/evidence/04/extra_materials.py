import pathlib,json,requests,fitz,hashlib,concurrent.futures,re,urllib.parse
root=pathlib.Path(__file__).resolve().parents[2]; out=root/'evidence/04/supplements';out.mkdir(exist_ok=True)
jobs={
'same':'https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zhou_SAME_Learning_Generic_ICCV_2025_supplemental.pdf',
'metaexplore':'https://openaccess.thecvf.com/content/CVPR2023/supplemental/Hwang_Meta-Explore_Exploratory_Hierarchical_CVPR_2023_supplemental.pdf',
'landmark_rxr':'https://proceedings.neurips.cc/paper/2021/file/0602940f23884f782058efac46f64b0f-Supplemental.pdf',
'embodiedqa':'https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/0052-supp.pdf',
'aiuta':'https://openaccess.thecvf.com/content/ICCV2025/html/Taioli_Collaborative_Instance_Object_Navigation_Leveraging_Uncertainty-Awareness_to_Minimize_Human-Agent_Dialogues_ICCV_2025_paper.html',
'ieve':'https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.html',
'embodiedgap':'https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Rethinking_the_Embodied_Gap_in_Vision-and-Language_Navigation_A_Holistic_Study_ICCV_2025_paper.html'}
def get(item):
 pid,url=item;rec=dict(paper_id=pid,url=url)
 try:
  r=requests.get(url,timeout=50);r.raise_for_status()
  if not r.content.startswith(b'%PDF'):
   urls=re.findall(r'href=[\"\']([^\"\']+)[\"\'][^>]*>\s*\[?supp',r.text,re.I)
   if not urls:rec['status']='Official proceedings page has no supplementary link';return rec
   url=urllib.parse.urljoin(url,urls[0]);rec['url']=url;r=requests.get(url,timeout=50);r.raise_for_status()
  data=r.content; doc=fitz.open(stream=data,filetype='pdf');dst=out/(pid+'_supp.pdf');dst.write_bytes(data)
  dst.with_suffix('.txt').write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n'+p.get_text() for i,p in enumerate(doc)),encoding='utf8')
  rec.update(file=str(dst.relative_to(root)),pages=len(doc),sha256=hashlib.sha256(data).hexdigest(),status='Retrieved; reading recorded separately')
 except Exception as e:rec['error']=str(e)
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:recs=list(ex.map(get,jobs.items()))
(out/'extra_manifest.json').write_text(json.dumps(recs,indent=2),encoding='utf8');print(json.dumps(recs,indent=2))
meta=json.loads((root/'repositories/04/knowno/source.json').read_text())
for p in meta['paths']:
 if p.endswith('.ipynb'):
  dst=root/'repositories/04/knowno'/p;dst.parent.mkdir(exist_ok=True,parents=True)
  r=requests.get(meta['repo_url'].replace('github.com','raw.githubusercontent.com')+'/'+meta['commit']+'/'+p,timeout=45);r.raise_for_status();dst.write_bytes(r.content)
  cells=json.loads(r.content)['cells']; dst.with_suffix('.txt').write_text('\n\n'.join(f'=== CELL {i} ({c["cell_type"]}) ===\n'+''.join(c['source']) for i,c in enumerate(cells)),encoding='utf8')
