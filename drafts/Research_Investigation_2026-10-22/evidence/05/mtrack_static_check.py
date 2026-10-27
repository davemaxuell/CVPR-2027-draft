"""Parse source text only. Does not import or execute M-TRACK code."""
from pathlib import Path
import ast,json,subprocess
R=Path(__file__).resolve().parents[2]; P=R/'repositories/05/M-Track'
result={'commit':subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),
 'executed':'2026-10-22','mode':'Python AST parsing and file existence checks; no project import/execution','syntax':{}}
for rel in ['mtrack_src/env.py','mtrack_src/agent.py','mtrack_src/milestone_checker.py','mtrack_src/milestone.py']:
 try: ast.parse((P/rel).read_text(encoding='utf-8')); result['syntax'][rel]={'parse':'pass'}
 except SyntaxError as e: result['syntax'][rel]={'parse':'fail','line':e.lineno,'message':e.msg,'text':e.text.strip() if e.text else None}
result['documented_launcher_target_exists']=(P/'r2r_src/train.py').exists()
result['runtime_checker_reference_files']=[str(p.relative_to(P)) for p in P.rglob('*.py') if 'milestone_checker' in p.read_text(encoding='utf-8',errors='replace')]
(R/'evidence/05/mtrack_static_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
