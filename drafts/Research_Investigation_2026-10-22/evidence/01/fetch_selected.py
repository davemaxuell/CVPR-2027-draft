from pathlib import Path
import requests,json,concurrent.futures
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'repositories/01'
SELECT={
'duet':['map_nav_src/r2r/agent.py','map_nav_src/models/vilmodel.py','map_nav_src/models/graph_utils.py','map_nav_src/r2r/env.py','map_nav_src/r2r/eval_utils.py','pretrain_src/model/pretrain_cmt.py','map_nav_src/r2r/main_nav.py'],
'scalevln':['VLN-DUET-RVR/map_nav_src/reverie/agent_obj.py','VLN-DUET-RVR/map_nav_src/models/vilmodel.py','VLN-DUET-RVR/map_nav_src/reverie/env_hm3d.py','VLN-DUET-RVR/pretrain_src/train_hm3d_reverie.py'],
'bevbert':['bevbert_ce/pretrain/pretrain_src/model/bev_utils.py','bevbert_ce/pretrain/pretrain_src/model/vilmodel.py','bevbert_ce/pretrain/pretrain_src/model/pretrain_cmt.py'],
'gridmm':['VLN_CE/vlnce_baselines/models/Policy_ViewSelection_GridMap.py','VLN_CE/vlnce_baselines/models/gridmap/vilmodel.py'],
'kerm':['knowledge_nav_src/models/vilmodel.py','preprocess/retrieve_facts.py','preprocess/get_knowledge_base.py'],
'goat':['map_nav_src/models/vilmodel.py','map_nav_src/do_utils/do_intervention.py'],
'gavln':['gavln/model/model_gavln.py','gavln/gavln_eval.py','gavln/dataset/vln_action_dataset.py'],
'vlnbert':['r2r_src/vlnbert/vlnbert_PREVALENT.py','r2r_src/agent.py','r2r_src/train.py','r2r_src/eval.py'],
'hamt':['finetune_src/models/vilmodel_cmt.py','finetune_src/r2r/agent_cmt.py','pretrain_src/model/pretrain_cmt.py'],
'airbert':['airbert.py','utils/dataset/bnb_dataset.py','train_bnb.py'],
'navillm':['models/nav_model.py','models/image_embedding.py','tasks/agents/r2r.py','tasks/agents/mp3d_agent.py'],
'navrag':['VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py','VLN-DUET-NAVRAG/map_nav_src/NavRAG/env.py','VLN-DUET-NAVRAG/pretrain_src/train_rag.py'],
'crossmodalmap':['models/vln_goal_predictor.py','models/semantic_grid.py','tester.py','trainer_vln.py'],
'weakmap':['vlnce_baselines/models/mg_map_policy.py','vlnce_baselines/common/rgb_mapping.py','vlnce_baselines/dagger_trainer.py'],
'regretful':['tasks/R2R-pano/models/policy_model.py','tasks/R2R-pano/agents/pano_agent.py'],
'selfmonitoring':['tasks/R2R-pano/models/policy_model.py','tasks/R2R-pano/agents/pano_agent.py'],
'speakerfollower':['tasks/R2R/rational_follower.py','tasks/R2R/data_augmentation_from_speaker.py','tasks/R2R/follower.py','tasks/R2R/speaker.py'],
'tacticalrewind':['follower.py','run_search.py','scorer.py'],
'r2r':['tasks/R2R/agent.py','tasks/R2R/model.py','tasks/R2R/eval.py'],
}
def fetch(job):
 id,path=job;meta=json.loads((OUT/id/'source.json').read_text('utf-8'));url=meta['repo_url'].replace('github.com','raw.githubusercontent.com')+'/'+meta['commit']+'/'+path
 try:
  res=requests.get(url,timeout=35);res.raise_for_status(); dest=OUT/id/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(res.text,'utf-8')
  return {'id':id,'path':path,'lines':len(res.text.splitlines()),'url':url}
 except Exception as e:return {'id':id,'path':path,'error':str(e)}
if __name__=='__main__':
 jobs=[(id,p) for id,paths in SELECT.items() for p in paths]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: records=list(ex.map(fetch,jobs))
 (ROOT/'evidence/01/code_downloads.json').write_text(json.dumps(records,indent=2),'utf-8')
 print(json.dumps({'downloaded':sum('error' not in r for r in records),'errors':[r for r in records if 'error' in r]}),flush=True)
