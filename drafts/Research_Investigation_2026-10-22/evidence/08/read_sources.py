"""Read-only scoped extraction for the independent logic reviewer."""
from pathlib import Path
import argparse, hashlib, json, re, textwrap

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('path')
parser.add_argument('--pages')
parser.add_argument('--lines')
args = parser.parse_args()
p = ROOT / args.path
text = p.read_text(encoding='utf-8-sig')
record = {'path': args.path, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
          'pages': args.pages, 'lines': args.lines, 'reviewer': 'assignment_08'}
with (ROOT/'evidence/08/source_reads.jsonl').open('a', encoding='utf8') as f:
    f.write(json.dumps(record, ensure_ascii=False)+'\n')
if args.pages:
    requested = [int(x) for x in args.pages.split(',')]
    for page in requested:
        pat = re.search(r'=== PDF PAGE '+str(page)+r' ===\s*(.*?)(?==== PDF PAGE |\Z)', text, re.S)
        if not pat:
            raise ValueError(f'Page {page} marker absent')
        print(f'PAGE {page}\n'+textwrap.fill(pat.group(1), 150))
elif args.lines:
    lines = text.splitlines()
    for piece in args.lines.split(','):
        ends=piece.split('-'); low=int(ends[0]); high=int(ends[-1])
        for i in range(low-1,min(high,len(lines))): print(f'{i+1}: {lines[i]}')
else:
    print(text)
