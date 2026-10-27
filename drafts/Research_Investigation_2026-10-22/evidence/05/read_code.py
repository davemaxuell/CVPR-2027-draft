import pathlib,sys
root=pathlib.Path(__file__).resolve().parents[2]
p=root/sys.argv[1];lines=p.read_text(encoding='utf-8',errors='replace').splitlines();ranges=sys.argv[2:] or ['1:999999']
for rr in ranges:
 a,b=map(int,rr.split(':'));print(f'=== {sys.argv[1]} {a}:{b} ===')
 for i in range(a-1,min(b,len(lines))):print(f'{i+1}: {lines[i]}')
