import json,sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
papers=json.loads((ROOT/'assignments/02_papers.json').read_text(encoding='utf-8-sig'))
lookup={p['id']:p for p in papers}
for spec in sys.argv[1:]:
    pid,_,ranges=spec.partition(':')
    p=lookup[pid]
    text=Path(p['local_files'][0]['text_path']).read_text(encoding='utf-8-sig')
    pages=re.split(r'=== PDF PAGE (\d+) ===',text)
    want=set()
    for r in (ranges or '1').split(','):
        s,_,e=r.partition('-');want.update(range(int(s),int(e or s)+1))
    print('\n######## '+pid+' ########')
    for i in range(1,len(pages),2):
        if int(pages[i]) in want:print('\nPAGE '+pages[i]+':\n'+pages[i+1].strip())
