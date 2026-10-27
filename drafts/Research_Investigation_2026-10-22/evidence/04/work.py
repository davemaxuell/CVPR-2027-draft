import json, pathlib, re, sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
def pages(pid, start=1, end=999):
    text=(ROOT/'corpus'/pid/'main.txt').read_text(encoding='utf-8')
    for block in re.split(r'(?==== PDF PAGE )',text):
        m=re.match(r'=== PDF PAGE (\d+) ===',block)
        if m and start<=int(m.group(1))<=end: print(block)
if __name__=='__main__':
    pages(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 1,int(sys.argv[3]) if len(sys.argv)>3 else 999)
