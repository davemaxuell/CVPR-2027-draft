"""Read-only metadata and local prerequisites; no model/data downloads."""
from pathlib import Path
import json, hashlib, urllib.request, importlib.util
ROOT=Path(__file__).resolve().parents[2]
out={'date':'2026-10-22','scope':'Read-only local annotation hashes and remote listing; not checkpoint loading or simulator execution.'}
inv=json.loads((ROOT/'experiments/local_data_inventory.json').read_text())
out['annotations']=[];scenes=[]
for item in inv['available_annotations']:
    p=Path(item['path']);data=p.read_bytes(); rows=json.loads(data);ss={r['scan'] for r in rows};scenes.append(ss)
    out['annotations'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'hash_matches_coordinator':hashlib.sha256(data).hexdigest()==item['sha256'],'records':len(rows),'scenes':len(ss),'instructions':sum(len(r['instructions']) for r in rows),'has_instr_encodings':all('instr_encodings' in r for r in rows)})
out['scene_overlap']=sorted(scenes[0]&scenes[1]);out['connectivity_files']=len(list((ROOT/'experiments/geometry/connectivity').glob('*.json')))
out['package_presence']={n:importlib.util.find_spec(n) is not None for n in ['habitat','habitat_sim','MatterSim']}
url='https://huggingface.co/api/datasets/woyut/NavQ_ICCV25-files/tree/main/CKPT_release?recursive=true'
try:
    data=urllib.request.urlopen(url,timeout=20).read();rows=json.loads(data)
    out['navq_remote_checkpoints']={'url':url,'status':'Listing retrieved; contents not downloaded/verified','entries':[{'path':r.get('path'),'size':r.get('size'),'oid':r.get('oid')} for r in rows]}
except Exception as ex:out['navq_remote_checkpoints']={'url':url,'status':str(ex)}
(ROOT/'evidence/10/asset_check_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
