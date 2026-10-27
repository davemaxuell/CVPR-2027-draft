from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'evidence'/'03'
files=list(E.glob('notes_*.json'))+[E/'code_audits.json']
prefixes='synthesize|of|in|to|at|by|from|with|over|under|between|within|through|only|per|each|all|and|or|a|an|one|two|three|four|five|eight|about|roughly|approximately|above|below|before|after|than|versus|plus|minus|batch|capacity|lr|weight|weights|temperature|threshold|coefficient|clipping|probability|margin|Lines|lines|line|Section|Sections|Table|Tables|Figure|Figures|Appendix|Algorithm|Stage|stages|Stage|Eqs|Eq|p|pp|SR|SPL|NE|OSR|nDTW|reports|reaches|gives|uses|has|have|had|retains|removes|adds|takes|for|size|width|length|epoch|epochs|iteration|iterations|steps|turns|tokens|frames|views|maximum|minimum|max|mean|default|requires|returns|approximately|takes|from|exceeds|sampled|observes|precomputed|examines|receives|selects|mean|waypoint|layer|linear|rank|pixel|forward|turn|waypoint|learning|FP|BF|corresponding|near|rises|rising|roughly|trial|trials|dimensions|dimensional|intercepts|recovers|leaves|contains|total|dataset|samples|experiments|generates|produces|retraining|takes'
rx=re.compile(r'\b('+prefixes+r')(?=\d)',re.I)
modelnums=re.compile(r'\b(SkillNav|ScaleVLN|NaVILA|NaVid|Uni-NaVid|SUSA|DUET|SRDF|CONSOLE|Progress-Think)(?=\d)')
protected=re.compile(r'https?://\S+|`[^`]*`|\b(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|yaml|yml|json|sh|bash|md|txt)(?::[0-9,–-]+)?')
def textfix(s):
 saved=[]
 def protect(m):saved.append(m.group());return f'§PROTECTED§{len(saved)-1}§'
 s=protected.sub(protect,s)
 s=rx.sub(r'\1 ',s);s=modelnums.sub(r'\1 ',s)
 # Common units in explanatory prose; leave compact model names and code identifiers intact.
 s=re.sub(r'(?<=\d)(GB|GPU-hours|Hz|FPS|hours|minutes|seconds|episodes|examples|steps|tokens|frames|cm|ms)\b',r' \1',s)
 s=re.sub(r'\b(October|November|January|February|March|April|May|June|July|August|September|December)(?=\d)',r'\1 ',s)
 for i,v in enumerate(saved):s=s.replace(f'§PROTECTED§{i}§',v)
 return s
def walk(x,key=''):
 if isinstance(x,str):
  if key in {'id','source','repo_url','dossier_path','file','sha256','version','locator'} or x.startswith(('https://','http://')):return x
  return textfix(x)
 if isinstance(x,list):
  if key=='code_files':return x
  return [walk(v,key) for v in x]
 if isinstance(x,dict):return {k:walk(v,k) for k,v in x.items()}
 return x
for p in files:p.write_text(json.dumps(walk(json.loads(p.read_text(encoding='utf-8'))),ensure_ascii=False,indent=2),encoding='utf-8')
for p in [ROOT/'assignments'/'03_summary.md',E/'source_ledger.md']:
 if p.exists():p.write_text(textfix(p.read_text(encoding='utf-8')),encoding='utf-8')
print('Numeric spacing normalized in prose; paths, IDs and code-file fields protected')
