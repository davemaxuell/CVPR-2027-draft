from pathlib import Path
from urllib.request import urlopen, Request
from datetime import datetime, timezone
import hashlib, json
root=Path('G:/내 드라이브/CVPR/261022')
out={'time_utc':datetime.now(timezone.utc).isoformat(),'primary_sources':[],'backbone_metadata':[],'errors':[]}
def fetch(url):
    with urlopen(Request(url,headers={'User-Agent':'research-feasibility-audit'}),timeout=15) as r:return r.read()
items=[('dinowm','gaoyuezhou/dino_wm','0a9492fa12044b852ae9e001cc74604b79c8bb0c','models/visual_world_model.py'),('dinowm','gaoyuezhou/dino_wm','0a9492fa12044b852ae9e001cc74604b79c8bb0c','train.py'),('nwm','facebookresearch/nwm','3f6cd8e70d6f2d1e2b9684acff510710135f0f41','models.py'),('nwm','facebookresearch/nwm','3f6cd8e70d6f2d1e2b9684acff510710135f0f41','isolated_nwm_infer.py')]
for folder,repo,rev,path in items:
    local=root/'Research_Investigation_2026-10-22/repositories/02'/folder/'source'/path
    url=f'https://raw.githubusercontent.com/{repo}/{rev}/{path}'
    record={'repo':repo,'revision':rev,'path':path,'url':url,'local_sha256':hashlib.sha256(local.read_bytes()).hexdigest()}
    try:
        data=fetch(url);record['remote_sha256']=hashlib.sha256(data).hexdigest();record['local_matches_remote']=data==local.read_bytes()
    except Exception as e: record['error']=repr(e)
    out['primary_sources'].append(record)
trials=json.loads((root/'Research Approach/Backbone_Trials.json').read_text())['trials']
for t in trials:
    model,rev=t['model_id'],t['revision']; record={'id':model,'revision':rev}
    for kind,url in [('metadata',f'https://huggingface.co/api/models/{model}/revision/{rev}'),('config',f'https://huggingface.co/{model}/resolve/{rev}/config.json')]:
        try:
            data=json.loads(fetch(url))
            if kind=='metadata': record[kind]={k:data.get(k) for k in ['sha','gated','safetensors','library_name']}
            else: record[kind]=data
        except Exception as e: record[kind]={'error':repr(e)}
    out['backbone_metadata'].append(record)
try:
    url='https://huggingface.co/api/models/facebook/nwm';data=json.loads(fetch(url))
    out['nwm_current']={'url':url,**{k:data.get(k) for k in ['sha','gated','lastModified']},'checkpoint_files':[i['rfilename'] for i in data.get('siblings',[]) if i['rfilename'].endswith('.pth.tar')]}
except Exception as e: out['errors'].append({'url':url,'error':repr(e)})
(root/'Research_Review_2026-10-23/evidence/06/source_pins.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'primary_sources':out['primary_sources'],'backbone_parameters':[{k:r.get(k) for k in ['id','revision','metadata']} for r in out['backbone_metadata']],'nwm_current':out.get('nwm_current'),'errors':out['errors']},indent=2))
