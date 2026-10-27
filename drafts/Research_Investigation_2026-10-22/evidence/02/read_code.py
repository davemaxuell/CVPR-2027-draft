import sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
log=R/'evidence/02/code_read_log.json'
records=json.loads(log.read_text()) if log.exists() else []
for arg in sys.argv[1:]:
 path,sep,ranges=arg.partition(':'); repo,_,file=path.partition('/')
 p=R/'repositories/02'/repo/'source'/file
 lines=p.read_text(encoding='utf-8',errors='replace').splitlines()
 print('\n###',repo,file)
 intervals=ranges.split(',') if sep else ['1-'+str(len(lines))]
 for interval in intervals:
  a,_,b=interval.partition('-'); a=int(a); b=int(b or a)
  print('\n'.join(f'{i+1}: {lines[i]}' for i in range(a-1,min(b,len(lines)))))
 records.append({'repo':repo,'file':file,'ranges':intervals})
 log.write_text(json.dumps(records,indent=2))
log.write_text(json.dumps(records,indent=2))
