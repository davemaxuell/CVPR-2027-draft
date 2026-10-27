from pathlib import Path
import hashlib,json
r=Path('G:/내 드라이브/CVPR/261022/Research_Review_2026-10-23')
inventory=json.loads((r/'material_inventory.json').read_text(encoding='utf-8-sig'))
check=[{'path':x['path'],'sha256_now':hashlib.sha256(Path(x['path']).read_bytes()).hexdigest(),'matches_inventory':hashlib.sha256(Path(x['path']).read_bytes()).hexdigest().lower()==x['sha256'].lower()} for x in inventory]
meta=json.loads((r/'evidence/06/source_pins.json').read_text())
resources={'basis':'Arithmetic estimates, not measured GPU memory. GiB=2**30; MiB=2**20. Optimizer estimate assumes FP32 parameters+gradient+two Adam moments = 16 bytes/parameter; excludes activations/runtime/workspace.', 'models':[]}
for m in meta['backbone_metadata']:
    n=m['metadata']['safetensors']['total']; resources['models'].append({'name':m['id'],'parameters_metadata':n,'fp16_weights_gib':2*n/2**30,'fp32_weights_gib':4*n/2**30,'full_fp32_adam_state_gib':16*n/2**30})
resources['scorer']={'nparams':2890115,'fp32_weights_mib':2890115*4/2**20,'fp32_adam_state_mib':2890115*16/2**20,'persistent_memory_fp16_bytes':8*256*2}
resources['feature_cache']={'current64_fp16_bytes':64*384*2,'current49_fp16_bytes':49*384*2,'raw256_fp16_bytes':256*384*2,'raw196_fp16_bytes':196*384*2,'current_plus_3_actual_plus_12_predicted_64_fp16_bytes_per_root':16*64*384*2,'same_for_64_roots_mib':64*16*64*384*2/2**20,'instruction256_width960_fp16_bytes_per_episode':256*960*2}
resources['native_grid_revision']={'original_candidate_tokens':65,'proposed_candidate_tokens':50,'candidate_self_attention_matrix_ratio':50**2/65**2,'visual_cache_ratio':49/64,'native_wm_attention_256_vs_196_ratio':256**2/196**2,'native_wm_default_history_tokens':3*196,'resized_wm_history_tokens':3*256}
cache=Path('C:/Users/user/.cache/huggingface/hub')
assets={'scope':'Expected HF cache only; does not prove assets absent elsewhere. Workspace rg search *.glb, *.navmesh, *r2r*.json.gz, *rxr*.json.gz, *train_guide.json.gz, *val_unseen_guide.json.gz found no matches.', 'expected_cached_models':{m['id']:(cache/('models--'+m['id'].replace('/','--'))).exists() for m in meta['backbone_metadata']}}
(r/'evidence/06/original_integrity.json').write_text(json.dumps(check,indent=2)+'\n',encoding='utf-8')
(r/'evidence/06/resource_arithmetic.json').write_text(json.dumps(resources,indent=2)+'\n',encoding='utf-8')
(r/'evidence/06/asset_check.json').write_text(json.dumps(assets,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'original_files':len(check),'all_unchanged':all(x['matches_inventory'] for x in check),'resources':resources,'assets':assets},indent=2))
