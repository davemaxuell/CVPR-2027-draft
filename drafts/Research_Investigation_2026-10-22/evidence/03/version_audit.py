from pathlib import Path
import json,difflib,re
ROOT=Path(__file__).resolve().parents[2]
for ident in ['awarevln','evonav','profocus','progressthink','susa']:
 m=json.loads((ROOT/'corpus'/ident/'metadata.json').read_text(encoding='utf-8'))
 print('\n',ident)
 for f in m['local_files']:
  print(f['filename'],f['pages'],f['sha256'])
 main=Path(m['local_files'][0]['text_path']).read_text(encoding='utf-8')
 for f in m['local_files'][1:]:
  if f['sha256']==m['local_files'][0]['sha256']:print('EXACT DUPLICATE');continue
  alt=Path(f['text_path']).read_text(encoding='utf-8')
  # Print changed text blocks for human version inspection, including numbers.
  aa=[re.sub(r'\s+',' ',x).strip() for x in main.splitlines() if x.strip()]
  bb=[re.sub(r'\s+',' ',x).strip() for x in alt.splitlines() if x.strip()]
  sm=difflib.SequenceMatcher(None,aa,bb,autojunk=False)
  changed=[]
  for tag,a,b,c,d in sm.get_opcodes():
   if tag!='equal':changed.append((tag,aa[a:b],bb[c:d]))
  outfile=ROOT/'evidence'/'03'/(ident+'_version_diff.json')
  outfile.write_text(json.dumps(changed,ensure_ascii=False,indent=2),encoding='utf-8')
  print('Changed blocks',len(changed),'lines',len(aa),len(bb),'saved',outfile.name)
  for tag,x,y in changed[:4]:print(tag,'OLD',' '.join(x)[:800],'NEW',' '.join(y)[:800])
