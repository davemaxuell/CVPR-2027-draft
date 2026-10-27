"""Compile/inspect source text without importing or executing fetched code."""
from pathlib import Path
import ast, hashlib, json
R=Path(__file__).resolve().parents[2]
repo=R/'repositories/04/m3e'
files=['models/nav_model.py','models/moe_lora.py','models/marco_router.py',
       'tools/momentum/experts_stats.py','tools/momentum/momentum_update.py',
       'train.py','tasks/agents/mp3d_agent.py','tasks/datasets/r2r.py']
records=[]
for relative in files:
    path=repo/relative
    source=path.read_text(encoding='utf-8')
    record={'file':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        compile(source,str(path),'exec')  # create code object only; never execute it
        record['compile']='pass'
    except SyntaxError as e:
        record.update(compile='fail',line=e.lineno,message=e.msg)
    records.append(record)
moe=ast.parse((repo/'models/moe_lora.py').read_text(encoding='utf-8'))
definitions=[x.name for x in moe.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))]
output={'date':'2026-10-22','commit':'35595353e6838b474ea835b9670419fa56e8f615',
        'scope':'Python compile and AST inspection only; no fetched module imported/executed',
        'records':records,'moe_lora_top_level_definitions':definitions,
        'constructor_requested_names_present':{name:name in definitions for name in ['inject_moe','freeze_except_moe']},
        'code_executed':False,'benchmark_executed':False}
(R/'evidence/05/m3e_static_results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
