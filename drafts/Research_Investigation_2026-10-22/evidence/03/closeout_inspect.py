from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[2]
notes={n['id']:n for p in (ROOT/'evidence'/'03').glob('notes_*.json') for n in json.loads(p.read_text(encoding='utf-8'))}
if len(sys.argv)>1:
 for ident in sys.argv[1:]:
  print('\nID',ident)
  if ident in notes: print(json.dumps(notes[ident],ensure_ascii=False,indent=2))
  meta=ROOT/'corpus'/ident/'metadata.json'
  if meta.exists():
   m=json.loads(meta.read_text(encoding='utf-8')); print('CORPUS',json.dumps({k:v for k,v in m.items() if k in ['extractions','files','text_files']},ensure_ascii=False))
else:
 for k,n in notes.items():
  print(k, '|',n.get('pages'), '|',n.get('status'), '|',n.get('versions'), '|',n.get('associated_files_reviewed'))
  print('STATUS',[(b['id'],b['status']) for b in n['bottlenecks']])
 for ident in ['auxthink_training','auxthink_evaluation','navila_locomotion']:
  s=json.loads((ROOT/'repositories'/'03'/ident/'snapshot.json').read_text()); print(ident,s['commit'])
