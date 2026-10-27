import sys,re,json,requests
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ident=sys.argv[1]; root=ROOT/'repositories'/'03'/ident
if sys.argv[2]=='paths':
 pat=re.compile(sys.argv[3],re.I)
 print('\n'.join(p for p in (root/'paths.txt').read_text().splitlines() if pat.search(p)))
elif sys.argv[2]=='fetch':
 r=json.loads((root/'snapshot.json').read_text());repo=r['repo_url'].split('github.com/')[-1]
 for p in sys.argv[3:]:
  q=requests.get(f'https://raw.githubusercontent.com/{repo}/{r["commit"]}/{p}',timeout=45)
  if q.status_code==200:
   dest=root/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(q.content)
   if p not in r['downloaded']:r['downloaded'].append(p)
  print(p,q.status_code)
 (root/'snapshot.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
else:
 for spec in sys.argv[2:]:
  p,_,sel=spec.partition('::');f=root/p
  if not f.exists():print('MISSING',p);continue
  lines=f.read_text(encoding='utf-8',errors='replace').splitlines();print('\nFILE',ident,p,'LINES',len(lines))
  chosen=set()
  if sel.startswith('@'):
   pat=re.compile(sel[1:],re.I)
   for i,l in enumerate(lines):
    if pat.search(l):chosen.update(range(max(0,i-3),min(len(lines),i+9)))
  elif sel:
   for s in sel.split(','):
    a,_,b=s.partition('-');chosen.update(range(int(a)-1,int(b or a)))
  else:chosen.update(range(len(lines)))
  for i in sorted(chosen):
   if i<len(lines):print(str(i+1)+': '+lines[i])
