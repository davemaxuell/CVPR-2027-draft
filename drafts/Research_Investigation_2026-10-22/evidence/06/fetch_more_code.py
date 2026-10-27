from pathlib import Path
import requests,json,concurrent.futures,hashlib
R=Path(r'G:\내 드라이브\CVPR\261022\Research_Investigation_2026-10-22')
repos={'pijepa':('AmirhoseinCh/PiJEPA','f38a265cbce4fac5aa5fd9b9f7b18c723bccb645'),'dartvln':('Japluto/DART-VLN','a8889cf27723f1ffaac96ee7a802657a0b8e371a'),'imaginenavpp':('200203Z/ImagineNav_plus','3803a563af97a1a0a2a6a176573a4a25c60c62a2')}
for name,(repo,sha) in repos.items():
 folder=R/'repositories/06'/f'{name}_snapshot';folder.mkdir(parents=True,exist_ok=True)
 response=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1',timeout=45)
 if response.ok:
  tree=response.json();(folder/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
  paths=[x['path'] for x in tree.get('tree',[]) if x['type']=='blob']
  print(name, '\n'.join(p for p in paths if p.endswith(('.py','.md','.yaml','.sh')) and ('octo/data/oxe/' not in p))[:14000])
 else:print(name,response.status_code)
