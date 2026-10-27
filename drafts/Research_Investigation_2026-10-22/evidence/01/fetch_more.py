from pathlib import Path
import json,requests,concurrent.futures
R=Path(__file__).resolve().parents[2]
sel={
'hsgm':['README.md','src/agent/agent.py','src/mapper.py','src/mapping_utils/path_planning.py','src/run_experiments.py','config/vlnce_test.yaml'],
'ivln':['README.md','ivlnce_baselines/models/map_cma_policy.py','ivlnce_baselines/common/mapping_module/mapper.py','ivlnce_baselines/trainers/iterative_dagger_trainer.py','ivlnce_baselines/trainers/tour_dataset.py','habitat_extensions/measures.py','ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/1_ftune_dagger.yaml'],
'roomtour3d':['README.md','tasks/agents/tour3d_agent.py','tasks/datasets/tour3d_dataset.py','tasks/datasets/room3d_envs.py','models/nav_model.py','configs/multi.yaml','train.py'],
'prevalent_r2r':['README.md','r2r_src/r2rmodel.py','r2r_src/agent.py','r2r_src/r2rpretrain_class.py','r2r_src/eval.py','tasks/R2R/nav_hugging_pretrain.py'],
'reverie':['tasks/REVERIE/eval_release.py','tasks/REVERIE/modelFast.py','tasks/REVERIE/follower.py','tasks/REVERIE/trainFast.py'],
'goat':['map_nav_src/models/vilmodel_GOAT.py','map_nav_src/do_utils/extract_room_type.py'],
'navrag':['instruction_generator/convert_to_dataset.py','VLN-DUET-NAVRAG/map_nav_src/NavRAG/eval_utils.py','VLN-DUET-NAVRAG/map_nav_src/scripts/rag_h14_envedit_mix.sh'],
'crossmodalmap':['datasets/dataloader.py','datasets/util/map_utils.py','train_options.py','configs/habitat_config.yaml'],
'weakmap':['vlnce_baselines/config/CMA_AUG_DA_TUNE.yaml','habitat_extensions/measures.py'],
'hamt':['finetune_src/r2r/env.py','finetune_src/r2r/data_utils.py','finetune_src/scripts/run_r2r.sh'],
'airbert':['utils/dataset/pano_dataset.py'],
}
jobs=[]
for id,paths in sel.items():
 meta=json.loads((R/'repositories/01'/id/'source.json').read_text())
 for p in paths:jobs.append((id,p,meta))
def job(j):
 id,p,m=j;u=m['repo_url'].replace('github.com','raw.githubusercontent.com')+'/'+m['commit']+'/'+p
 try:
  res=requests.get(u,timeout=45);res.raise_for_status();f=R/'repositories/01'/id/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(res.text,'utf-8');return {'id':id,'path':p,'url':u,'lines':len(res.text.splitlines())}
 except Exception as e:return {'id':id,'path':p,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:out=list(ex.map(job,jobs))
(R/'evidence/01/code_downloads_more.json').write_text(json.dumps(out,indent=2),'utf-8')
print(json.dumps({'downloaded':sum('lines'in x for x in out),'errors':[x for x in out if 'error'in x]},indent=2))
