from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
pid,path=sys.argv[1:3];start=int(sys.argv[3]) if len(sys.argv)>3 else 1;end=int(sys.argv[4]) if len(sys.argv)>4 else 100000
p=ROOT/'repositories/04'/pid/path
lines=p.read_text(encoding='utf-8').splitlines()
print(f'{pid}: {path} ({len(lines)} lines)')
for i,line in enumerate(lines,start=1):
    if start<=i<=end:print(f'{i}: {line}')
ledger=ROOT/'evidence/04/source_read_ranges.json'
data=json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else []
data.append({'paper_id':pid,'path':path,'start':start,'end':min(end,len(lines)),'executed':False})
ledger.write_text(json.dumps(data,indent=2),encoding='utf-8')
