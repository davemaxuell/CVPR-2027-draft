import pathlib,re,sys,json
root=pathlib.Path(__file__).resolve().parents[2]
p=root/sys.argv[1];start,end=map(int,sys.argv[2:4])
for b in re.split(r'(?==== PDF PAGE )',p.read_text(encoding='utf8')):
 m=re.match(r'=== PDF PAGE (\d+) ===',b)
 if m and start<=int(m[1])<=end:print(b)
with (root/'evidence/04/extra_reads.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(file=str(p.relative_to(root)),start=start,end=end))+'\n')
