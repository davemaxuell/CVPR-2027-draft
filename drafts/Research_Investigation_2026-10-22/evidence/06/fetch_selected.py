from pathlib import Path
import requests,json,hashlib,concurrent.futures
R=Path(__file__).resolve().parents[2]
repos={'pijepa':('AmirhoseinCh/PiJEPA','f38a265cbce4fac5aa5fd9b9f7b18c723bccb645',['README.md','scripts/eval_octo_mppi_cast.py','scripts/eval_octo_wm_plan.py','scripts/train_cast_wm.py','scripts/configs/train_cast_wm_config.py','octo/data/dataset_cast.py','octo/data/utils/cast_transforms.py','scripts/configs/finetune_cast_dino_config.py']), 'dartvln':('Japluto/DART-VLN','a8889cf27723f1ffaac96ee7a802657a0b8e371a',['README.md','map_nav_src/models/graph_utils.py','map_nav_src/models/vilmodel.py','map_nav_src/r2r/agent.py','map_nav_src/r2r/parser.py','map_nav_src/reverie/agent_obj.py','map_nav_src/reverie/parser.py','map_nav_src/scripts/EVAL_COMMANDS.md','map_nav_src/scripts/run_r2r.sh']), 'imaginenavpp':('200203Z/ImagineNav_plus','3803a563af97a1a0a2a6a176573a4a25c60c62a2',['README.md'])}
def f(z):
 name,repo,sha,p=z;url=f'https://raw.githubusercontent.com/{repo}/{sha}/{p}';r=requests.get(url,timeout=50);target=R/'repositories/06'/f'{name}_snapshot'/p
 if r.ok:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(r.content)
 return dict(repository=repo,commit=sha,path=p,url=url,local=str(target.relative_to(R)),status=r.status_code,sha256=hashlib.sha256(r.content).hexdigest() if r.ok else None)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as e:out=list(e.map(f,[(n,r,s,p) for n,(r,s,ps) in repos.items() for p in ps]))
(R/'evidence/06/more_code_manifest.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print([(x['repository'],x['path'],x['status']) for x in out])
