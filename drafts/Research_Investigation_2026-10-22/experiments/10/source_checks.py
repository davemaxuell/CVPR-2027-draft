"""Isolated reviewed AST functions; no downloaded project module is imported.

Fake experts test dispatch, synthetic RGB tests memory logic. These are NOT model,
simulator, calibration, prevalence, or benchmark experiments. Disk methods are stubs.
"""
from pathlib import Path
import ast, copy, hashlib, json, io, contextlib, types
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/10'; OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
 'skill':'repositories/03/skillnav/skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py',
 'rag':'repositories/10/psgnav/utils/rag_manager.py',
 'psg':'repositories/03/psgnav/PSG_Nav.py',
 'detect':'repositories/10/psgnav/utils/object_detection.py',
 'duet':'repositories/01/duet/map_nav_src/r2r/agent.py'}
ledger=[]
class CPUOnly(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda' and not node.args and not node.keywords:
            return node.func.value
        return node
def getfn(key,name,cpu=False):
    data=(ROOT/SOURCES[key]).read_bytes(); tree=ast.parse(data.decode('utf-8-sig'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node=copy.deepcopy(node)
    ledger.append({'path':SOURCES[key],'sha256':hashlib.sha256(data).hexdigest(),'function':name,
                   'lines':[node.lineno,node.end_lineno],'changes':'Remove zero-argument .cuda() calls only' if cpu else 'None; dependencies explicitly stubbed below'})
    if cpu: node=CPUOnly().visit(node)
    node.decorator_list=[]
    mod=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    (OUT/f'extracted_{key}_{name}.py').write_text(ast.unparse(mod),encoding='utf-8')
    ns=dict(globals()); exec(compile(mod,SOURCES[key],'exec'),ns)
    return ns[name]

class Logger:
    def info(self,*a,**kw):pass
    def warning(self,*a,**kw):pass
def pad_tensors(xs): return torch.nn.utils.rnn.pad_sequence(xs,batch_first=True)
class Expert:
    def __init__(self,i):self.i=i
    def __call__(self,mode,data):
        if mode=='language':return torch.full((1,2,1),float(self.i))
        if mode=='panorama':return torch.full((1,2,1),float(self.i)),torch.ones((1,2))
        if mode=='navigation':return torch.full((data['txt_embeds'].shape[0],),self.i,dtype=torch.long)
        raise ValueError(mode)
skillfn=getfn('skill','_get_input_embeds',cpu=True)
source=ast.parse((ROOT/SOURCES['skill']).read_text(encoding='utf-8'))
navstmt=next(n for n in ast.walk(source) if isinstance(n,ast.Assign) and n.lineno==838)
navfn=ast.FunctionDef(name='navigation_call',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self'),ast.arg(arg='nav_inputs')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[copy.deepcopy(navstmt),ast.Return(value=ast.Name(id='nav_outs',ctx=ast.Load()))],decorator_list=[])
navmod=ast.fix_missing_locations(ast.Module(body=[navfn],type_ignores=[])); ns={};exec(compile(navmod,'isolated_navigation_call','exec'),ns)
def skillrun(ids):
    model=types.SimpleNamespace(vln_berts=[Expert(i) for i in range(5)],logger=Logger(),args=types.SimpleNamespace(seed=0))
    model._panorama_feature_variable=lambda obs:{}
    obs=[{'instr_id':f'i{i}','viewpoint':'v'} for i in range(len(ids))]
    maps=[types.SimpleNamespace(node_step_ids={}) for _ in ids]
    txt,*_=skillfn(model,obs,{'txt_ids':torch.ones((len(ids),2))},{},maps,[False]*len(ids),ids.copy(),0)
    return ns['navigation_call'](model,{'txt_embeds':txt}).tolist()
ids=[0,1,2,3,4,0,1,2];perm=[7,6,5,4,3,2,1,0]
single=[skillrun([i])[0] for i in ids];batch=skillrun(ids)
permuted=skillrun([ids[i] for i in perm]);inverse=[permuted[perm.index(i)] for i in range(8)]
skill_result={'selected_experts':ids,'batch1_navigation_experts':single,'batch8_navigation_experts':batch,
 'batch8_reversed_then_unpermuted':inverse,'batch1_vs8_mismatches':sum(a!=b for a,b in zip(single,batch)),
 'permutation_mismatches':sum(a!=b for a,b in zip(batch,inverse)),
 'interpretation':'Actual extracted dispatch with fake distinct expert functions. Shows route-dependent navigation-head selection is batch/order dependent in this code. No learned logits or paper result tested.'}

class RAGDocument:pass
RAGDocument.__init__=getfn('rag','__init__')
rag=types.SimpleNamespace(false_positive_documents={},documents={},diversity_threshold=.97,
 max_false_positives_per_category=20,false_positive_threshold=.95,similarity_threshold=.1,caption_penalty=1.)
for name in ['add_false_positive','verify_detection','_extract_feature','_extract_color_histogram','_cosine_similarity','_captions_match','_compute_max_similarity']:
    setattr(rag,name,types.MethodType(getfn('rag',name),rag))
rag._save_false_positive=lambda doc:None
rag._delete_false_positive=lambda doc:None
crop=np.zeros((100,100,3),dtype=np.uint8);crop[:50,:,0]=255;crop[50:,:,2]=255
agent=types.SimpleNamespace(metrics={},simulator=types.SimpleNamespace(_env=types.SimpleNamespace(episode_over=True)),
 total_steps=500,using_random_goal=False,random_nav_start_step=None,rag_collect_enabled=True,rag_manager=rag,
 count_episodes=1,current_dataset='synthetic',current_scene_id='scene_A',obj_goal='chair',
 rag_sliding_window=[dict(crop=crop,caption='chair',confidence=.9)],random_nav_periods=[],args=types.SimpleNamespace(visualize=False))
update=getfn('psg','update_metrics')
with contextlib.redirect_stdout(io.StringIO()):
    before=rag.verify_detection('chair',crop,'chair',.9)
    update(agent,dict(distance_to_goal=2.,spl=0.,softspl=.5,success=0.))
    after=rag.verify_detection('chair',crop,'chair',.9)
    # Explicitly provide a positive record too. FP priority must still reject.
    rag.documents['chair']=[RAGDocument('chair',crop,'chair',.9,feature=rag._extract_feature(crop))]
    positive_conflict=rag.verify_detection('chair',crop,'chair',.9)
    # Same per-channel histogram but swapped image regions.
    rearranged=rag.verify_detection('chair',crop[::-1].copy(),'chair',.9)
    # Read-side caller: real rejection clears found_goal and marks spatial region.
    agent.goal_gps_map=np.zeros((50,50));agent.goal_map=np.zeros((50,50));agent.goal_gps_timestamp_map=np.zeros((50,50))
    agent.goal_merge_threshold=.8;agent.map_resolution=10;agent.map_size=50
    agent.found_goal=True;agent.found_goal_times=3;agent.found_possible_goal=False
    agent.rag_latest_goal_detection=dict(crop=crop,caption='chair',confidence=.9,map_x=25,map_y=25)
    agent.reset_goal_detection=types.MethodType(getfn('psg','reset_goal_detection'),agent)
    detector=types.SimpleNamespace(agent=agent)
    getfn('detect','_verify_detection_with_rag')(detector)
rag_result={'input_contract':'Synthetic crop stipulated to represent correct category; terminal failure stipulated to arise from control. Functions do not observe these hidden stipulations.',
 'before_failure_verified':before[0],'false_positive_records_after_failure':len(rag.false_positive_documents['chair']),
 'after_failure_verified':after[0],'after_failure_confidence':after[1],
 'with_positive_record_verified':positive_conflict[0],'same_histogram_rearranged_crop_verified':rearranged[0],
 'goal_retained_after_real_read_caller':agent.found_goal,'negative_map_cells':int((agent.goal_gps_map==-1).sum()),
 'interpretation':'Source-function conditional counterexample to outcome-only semantic writes, including actual rejection side effects. Not an observed real correct detection or empirical harmful-write frequency; no disk banks/simulator/model were used.'}

teacher=getfn('duet','_teacher_action_r4r',cpu=True)
model=types.SimpleNamespace(args=types.SimpleNamespace(ignoreid=-100,expert_policy='spl'))
ob={'viewpoint':'goal','gt_path':['start','A','goal'],'scan':'synthetic'}
labels=teacher(model,[ob,ob],[[None,'A'],[None,'A']],[False,False],traj=[{'path':[['start','goal']]},{'path':[['start','A','goal']]}]).tolist()
duet_result={'teacher_labels_different_history_at_endpoint':labels,'interpretation':'The native teacher STOP condition is exact endpoint, independent of whether the illustrative A obligation was visited. The new semantic task must separately validate that obligation; this is not a measured standard-R2R teacher error.'}

result={'date':'2026-10-22','python_and_dependencies':{'torch':torch.__version__,'numpy':np.__version__,'cv2':cv2.__version__},
 'execution_scope':'Reviewed source AST functions with explicit stubs and synthetic inputs; no project imports, models, assets, simulator, or benchmark.',
 'skillnav':skill_result,'psgnav':rag_result,'duet':duet_result,'source_functions':ledger}
(OUT/'source_check_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['skillnav','psgnav','duet']},indent=2))
