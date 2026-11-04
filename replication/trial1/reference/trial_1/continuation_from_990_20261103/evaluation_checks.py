"""Read-only integrity checks, reused from the verified evaluator."""
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parents[4]/"queue_after_stageA"))
from common import read_json,sha256
SPLITS={"val_seen":778,"val_unseen":1839}

def verify_split(root,split,checkpoint):
    expected=SPLITS[split];folder=root/'phase1'/split
    report=read_json(folder/'complete.json',{})
    if report.get('status')!='complete' or report.get('episodes')!=expected:raise RuntimeError('Official evaluation is incomplete: '+split)
    cfg=read_json(folder/'config.json',{})
    if cfg.get('checkpoint_sha256')!=checkpoint['checkpoint_sha256'] or cfg.get('weights')!=checkpoint['checkpoint']:raise RuntimeError('Evaluation checkpoint identity mismatch.')
    selected=cfg['selected_episodes'];files={p.stem:p for p in (folder/'episodes').glob('*.json')}
    if len(set(selected))!=expected or set(selected)!=set(files):raise RuntimeError('Incomplete/duplicate evaluation episode set.')
    sums={};hashes={}
    for eid in selected:
        row=read_json(files[eid]);hashes[eid]=sha256(files[eid])
        if row['episode_id']!=eid or row['dataset']!='r2r' or row['split']!=split:raise RuntimeError('Evaluation record identity mismatch.')
        if len(row['executed_actions'])!=len(row['action_steps']) or len(row['executed_actions'])!=row['metrics']['steps_taken']:raise RuntimeError('Evaluation action accounting mismatch.')
        for name,value in row['metrics'].items():
            if not math.isfinite(value):raise RuntimeError('Nonfinite evaluation metric.')
            sums[name]=sums.get(name,0.)+value
    for name,total in sums.items():
        if not math.isclose(report['metrics'][name],total/expected,rel_tol=1e-10,abs_tol=1e-10):raise RuntimeError('Evaluation aggregation mismatch.')
    stable=[]
    for p in (folder/'workers').glob('*.identity.json'):
        row=read_json(p)
        if row['weights']!=checkpoint['checkpoint'] or any(row['weight_sha256'].get(n)!=v for n,v in checkpoint['checkpoint_sha256'].items()):raise RuntimeError('Evaluation worker loaded another policy.')
        if row.get('saved_places_slot') or row.get('stop_logit_bias')!=0:raise RuntimeError('Unexpected custom slot/STOP change.')
        stable.append({k:v for k,v in row.items() if k not in {'device','seconds','nbytes'}})
    if not stable or any(row!=stable[0] for row in stable):raise RuntimeError('Evaluation worker identities differ.')
    return dict(report=report,episode_sha256=hashes,config_sha256=sha256(folder/'config.json'))

