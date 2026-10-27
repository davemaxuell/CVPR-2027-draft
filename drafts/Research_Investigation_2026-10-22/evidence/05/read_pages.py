import re,sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
id=sys.argv[1]; path=root/'followups'/'sources'/f'{id}.txt'
if not path.exists(): path=root/'evidence'/'05'/'sources'/f'{id}.txt'
if not path.exists(): path=root/'evidence'/'06'/'papers'/f'{id}.txt'
if not path.exists(): path=root/'corpus'/id/'main.txt'
text=path.read_text(encoding='utf-8')
parts=re.split(r'=== PDF PAGE (\d+) ===',text)
lo=int(sys.argv[2]) if len(sys.argv)>2 else 1
hi=int(sys.argv[3]) if len(sys.argv)>3 else 999
for i in range(1,len(parts),2):
 if lo<=int(parts[i])<=hi:
  s=re.sub(r'[ \t]+',' ',parts[i+1]); s=re.sub(r'\n\s*\n','\n',s)
  print(f'=== PAGE {parts[i]} ===\n{s}')

