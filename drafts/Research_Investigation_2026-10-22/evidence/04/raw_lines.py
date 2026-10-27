import pathlib,sys,json
root=pathlib.Path(__file__).resolve().parents[2]
pid,path,start,end=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
data=(root/'repositories/04'/pid/path).read_bytes().decode('utf8').split('\n')
print(pid,path,'LF line numbers')
for i in range(start,min(end,len(data))+1):print(f'{i}: {data[i-1].rstrip(chr(13))}')
log=root/'evidence/04/raw_source_reads.jsonl'
with log.open('a',encoding='utf8') as f:f.write(json.dumps(dict(paper_id=pid,path=path,start=start,end=end))+'\n')
