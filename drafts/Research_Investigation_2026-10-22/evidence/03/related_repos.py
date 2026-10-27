from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
for ident,url in [('auxthink_training','https://github.com/markinruc/VILA'),('auxthink_evaluation','https://github.com/markinruc/VLN_CE'),('navila_locomotion','https://github.com/yang-zj1026/legged-loco')]:
 p=ROOT/'repositories'/'03'/ident;p.mkdir(parents=True,exist_ok=True)
 if not (p/'snapshot.json').exists(): (p/'snapshot.json').write_text(json.dumps({'repo_url':url,'status':'Retrieval pending','retrieved':'2026-10-22'}),encoding='utf-8')
source=ROOT/'repositories'/'03'/'auxthink'/'snapshot.json'
dest=ROOT/'repositories'/'03'/'progressthink';dest.mkdir(exist_ok=True)
r=json.loads(source.read_text());r['shared_snapshot']='../auxthink';(dest/'snapshot.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
