from pathlib import Path
import json,re,fitz
ROOT=Path(__file__).resolve().parents[2]
for f in sorted((ROOT/'evidence'/'03').glob('notes_*.json')):
 for n in json.loads(f.read_text(encoding='utf-8')):
  bad=set(re.findall(r'\b[A-Za-z]{18,}\b',json.dumps(n)))
  if bad: print(n['id'],sorted(bad))
p=fitz.open(ROOT.parent/'Paper'/'SkillNav.pdf')[17]
p.get_pixmap(matrix=fitz.Matrix(1.3,1.3),clip=fitz.Rect(0,0,p.rect.width,310)).save(ROOT/'evidence'/'03'/'skillnav_table10.png')
