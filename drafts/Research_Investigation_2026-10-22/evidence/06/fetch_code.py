"""Fetch immutable public source text only; never import or execute it."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, json, requests
ROOT=Path(__file__).resolve().parents[2]
REPOS={
 'ua_nwm': ('DurYi/UA-NWM','399960ce5fea5a3aa98084beaf44d92c2def9774',[
  'README.md','configs/ua_nwm.yaml','uanwm/models/hep.py','uanwm/models/world_model.py',
  'uanwm/losses/post_training_losses.py','uanwm/losses/prediction_losses.py',
  'uanwm/train_hep.py','uanwm/staged_train.py','uanwm/datasets/trajectory_dataset.py',
  'uanwm/eval/scoring.py','uanwm/eval/planning_eval.py','uanwm/eval/ranking_eval.py',
  'uanwm/eval/validate_from_raw.py','deploy/src/uav_openpi/ua_nwm_policy.py',
  'deploy/src/uav_openpi/goal_termination.py','deploy/src/uav_openpi/onboard_runtime.py']),
 'janusvln': ('MIV-XJTU/JanusVLN','8cfe3129f7af4ca276bd5e38f1fae7e5367a09ec',[
  'README.md','config/vln_r2r.yaml','config/vln_dagger.yaml','scripts/train.sh',
  'scripts/evaluation.sh','src/evaluation.py','src/dagger.py',
  'src/qwen_vl/model/modeling_qwen2_5_vl.py','src/qwen_vl/model/loss.py',
  'src/qwen_vl/model/vggt/models/vggt.py','src/qwen_vl/model/vggt/models/aggregator.py',
  'src/qwen_vl/data/data_qwen.py','src/qwen_vl/train/train_qwen.py','create_data/create_data.py'])
}
def fetch(args):
 name,repo,commit,path=args
 url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
 target=ROOT/'repositories'/'06'/f'{name}_snapshot'/path
 try:
  response=requests.get(url,timeout=60); response.raise_for_status()
  target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(response.content)
  return dict(repository=repo,commit=commit,path=path,url=url,local=str(target.relative_to(ROOT)),sha256=hashlib.sha256(response.content).hexdigest(),status='Downloaded; not yet counted as inspected')
 except Exception as exc:return dict(repository=repo,commit=commit,path=path,url=url,status='Unavailable',error=str(exc))
if __name__=='__main__':
 tasks=[(name,repo,sha,p) for name,(repo,sha,paths) in REPOS.items() for p in paths]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: records=list(pool.map(fetch,tasks))
 (ROOT/'evidence'/'06'/'code_download_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'files':len(records),'available':sum(r['status'].startswith('Downloaded') for r in records)}))
