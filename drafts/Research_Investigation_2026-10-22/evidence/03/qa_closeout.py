from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]
records=json.loads((ROOT/'assignments'/'03_results.json').read_text(encoding='utf-8'))
issues=[]
for r in records:
 ident=r['paper_id']; repo=ROOT/'repositories'/'03'/ident
 snap=json.loads((repo/'snapshot.json').read_text()) if (repo/'snapshot.json').exists() else {}
 shared=snap.get('shared_snapshot')
 for file in r['code_files_inspected']:
  f=repo/file.split(':')[0]
  if not f.exists():issues.append([ident,'missing cited code',str(f)])
 text=Path(r['dossier_path']).read_text(encoding='utf-8')
 for word in re.findall(r'[A-Za-z]{25,}',text):
  if word.lower() not in ['roboticsproceedings']:issues.append([ident,'long word',word])
 for match in re.finditer(r'.{0,70}(?:pending|not yet|needs.*inspection|require.*audit).{0,100}',text,re.I):
  issues.append([ident,'stale/explicit-scope wording',match.group(0)])
print(json.dumps(issues,ensure_ascii=False,indent=2))
print('Records',len(records),'code repos',sum(bool(r['code_files_inspected']) for r in records))
for ident in ['navfom','mapgpt','mapnav','psgnav']:
 text=(ROOT/'corpus'/ident/'main.txt').read_text(encoding='utf-8');pages=re.split(r'=== PDF PAGE (\d+) ===',text)
 print('\nPAGE INDEX',ident)
 for i in range(1,len(pages),2):
  if int(pages[i])>=9:print(pages[i],re.sub(r'\s+',' ',pages[i+1])[:160])
