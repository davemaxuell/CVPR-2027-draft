import sys,re,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]; log=R/'evidence/02/code_read_log.json'
records=json.loads(log.read_text()) if log.exists() else []
for arg in sys.argv[1:]:
 path,pat=arg.split('::',1); repo,_,file=path.partition('/')
 p=R/'repositories/02'/repo/'source'/file
 if not p.exists():print('MISSING',path);continue
 lines=p.read_text(encoding='utf-8',errors='replace').splitlines(); selected=set()
 for i,line in enumerate(lines):
  if re.search(pat,line,re.I):selected.update(range(max(0,i-2),min(len(lines),i+4)))
 print('\n###',repo,file,'selected lines',len(selected))
 print('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(selected)))
 records.append({'repo':repo,'file':file,'selected_lines':[i+1 for i in sorted(selected)],'pattern':pat})
log.write_text(json.dumps(records,indent=2))
