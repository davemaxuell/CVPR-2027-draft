import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for spec in sys.argv[1:]:
    ident,_,span=spec.partition(':')
    ident,_,variant=ident.partition('/')
    path=ROOT/'corpus'/ident/(variant+'.txt' if variant else 'main.txt')
    alt=ROOT/'evidence'/'03'/(ident+'_alternate.txt')
    if alt.exists() and not variant: path=alt
    text=path.read_text(encoding='utf-8')
    pages=re.split(r'=== PDF PAGE (\d+) ===',text)
    wanted=set()
    for part in (span or '1-999').split(','):
        a,_,b=part.partition('-'); wanted.update(range(int(a),int(b or a)+1))
    print('\n'+ident.upper()+'\n')
    for i in range(1,len(pages),2):
        if int(pages[i]) in wanted: print('=== PDF PAGE '+pages[i]+' ===\n'+pages[i+1])
