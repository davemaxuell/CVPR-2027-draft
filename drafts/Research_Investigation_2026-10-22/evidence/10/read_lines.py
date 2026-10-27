from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
for spec in sys.argv[1:]:
    path,start,end=spec.rsplit(':',2); start=int(start);end=int(end)
    print('\nFILE '+path)
    text=(root/path).read_text(encoding='utf-8-sig').splitlines()
    print('\n'.join(f'{i+1}: {text[i]}' for i in range(start-1,min(end,len(text)))))
