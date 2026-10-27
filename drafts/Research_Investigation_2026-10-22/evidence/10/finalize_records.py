from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2]
entries=[]
def add(repo,commit,local,upstream,ranges,scope):
    p=ROOT/local
    entries.append(dict(repo_url='https://github.com/'+repo,commit=commit,local_path=local,upstream_path=upstream,
      url=f'https://github.com/{repo}/blob/{commit}/{upstream}',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
      inspected_ranges=ranges,scope=scope,project_module_imported=False))
duet='93e8b233164bc079a6db48b8a0a78d123ec8de41'
for path,ranges in [('map_nav_src/r2r/agent.py',['110–163','207–310','340–482']),('map_nav_src/models/vilmodel.py',['786–828']),('map_nav_src/r2r/main_nav.py',['1–85','134–172']),('map_nav_src/scripts/run_r2r.sh',['1–40']),('README.md',['1–50'])]:
    add('cshizhe/VLN-DUET',duet,'repositories/01/duet/'+path,path,ranges,'Teacher, termination, candidate, data scheduling and prerequisites; isolated teacher function executed separately')
add('cshizhe/VLN-DUET',duet,'repositories/10/duet/data_utils.py','map_nav_src/r2r/data_utils.py',['1–46'],'Raw/encoded data contract and instruction flattening')
rsi='5e8fa2c74cf8e7dc305b33088a963b200e20805d'
for path,ranges in [('evo/evo_rollout.py',['1490–1600']),('evo/evo_utils.py',['1013–1105']),('README.md',['1–120'])]:
    add('OpenGVLab/RSI-Nav',rsi,'repositories/05/RSI-Nav/'+path,path,ranges,'Acceptance, source-route samples, replay and dependencies; inspected only')
navq='9465fe43c9c1583bd7ba390b6be30996ea4f5a5c'
for path,ranges in [('code/Q_pretrain/dataset.py',['136–244']),('code/Q_pretrain/GPT_VL.py',['209–243']),('code/map_nav_src/models/vilmodel.py',['638–674','724–760','786–828']),('code/map_nav_src/models/vlnbert_init.py',['58–90']),('code/map_nav_src/reverie/agent_obj.py',['111–122','409–442','470–526']),('code/map_nav_src/scripts/run_reverie.sh',['17–20','66–81']),('README.md',['1–60'])]:
    add('woyut/NavQ_ICCV25',navq,'repositories/02/navq/source/'+path,path,ranges,'Predictor targets, gradients, checkpoint/feature integration and prerequisites; inspected only')
skill='b56de12a6af07ce09c9252938b1fb69e55d33103'
for path,ranges in [('skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py',['595–684','720–865']),('skillnav/backbones/scalevln/maps_nav_src/scripts/test_r2r_b16_moe-top1.sh',['1–98'])]:
    add('HLR/SkillNav',skill,'repositories/03/skillnav/'+path,path,ranges,'Actual dispatch and launcher; isolated dispatch function/call executed with fake experts')
psg='181d878ff207845a31e542a19a6592925c1e9bab'
for path,ranges in [('PSG_Nav.py',['238–252','295–323','1470–1589','1659–1678']),('README.md',['1–130'])]:
    add('Chenrf1121/psg-nav',psg,'repositories/03/psgnav/'+path,path,ranges,'Terminal metrics, writes, resets, default options and prerequisites; isolated selected methods executed')
for path,ranges in [('utils/rag_manager.py',['1–110','185–453','739–813']),('utils/object_detection.py',['260–325','427–465','536–564','618–672']),('server.py',['1–70'])]:
    add('Chenrf1121/psg-nav',psg,'repositories/10/psgnav/'+path,path,ranges,'Memory write/read/feature and rejection caller; selected methods executed with synthetic data and I/O stubs; server only inspected')
add('GengzeZhou/SAME','b74c57bbb189a0ed8c908e90b53d02dfbe417528','repositories/04/same/src/models/vilmodel.py','src/models/vilmodel.py',['967–988'],'Router batch-axis inspection only; not executed')
(ROOT/'evidence/10/source_manifest.json').write_text(json.dumps({'date':'2026-10-22','entries':entries,'paper_inspection':[{'paper_id':'psgnav','locator':'Section 3.4, Eqs.9–11; Section 4.1, PDF pp.6–7','scope':'Direct extracted full method/protocol passage, not figures or entire paper'}],'remote_listing':'evidence/10/asset_check_results.json','execution_record':'experiments/10/source_check_results.json'},indent=2),encoding='utf-8')
findings=[
 {'proposal_id':'P1','decision':'revise','feasibility':'Proceed with teacher diagnostic after assets/labels; defer model training','critical_findings':[
  {'id':'10-P1-01','evidence_type':'INSPECTED CODE','claim':'Replacing only the DUET teacher leaves endpoint-based training termination, no-frontier/budget termination and cached endpoint return outside the new semantics.','source':'map_nav_src/r2r/agent.py:453–478','commit':duet},
  {'id':'10-P1-02','evidence_type':'INSPECTED CODE','claim':'Visited-node global logits are masked and macro actions jump the simulator to the destination; new candidate/event contracts must be shared by all controls.','source':'agent.py:110–123,287–303; models/vilmodel.py:786–813','commit':duet},
  {'id':'10-P1-03','evidence_type':'EXECUTED TEST','claim':'Extracted native teacher returns STOP for both synthetic histories at the same endpoint; no real instruction violation measured.','source':'experiments/10/source_check_results.json#duet'}],
  'required_revisions':['Add termination, cached-return and macro-observation rules to implementation map.','Separate semantic viability from efficiency preference.','Keep candidate/return/event access identical in all semantic-task controls; native DUET reported separately.'],
  'cheapest_decisive_test':'Blinded natural frozen-policy checkpoint annotation plus separate constructed history pairs; compare teachers before training.',
  'go_gate':'Interpretable scene-distributed natural disagreement survives adjudication and simple reversible/milestone baselines.',
  'stop_gate':'Negligible natural discrepancy, unreliable semantics or simpler teacher eliminates gap.',
  'resource_dependencies':['Pinned compatible checkpoint','Encoded annotations','Visual features/scene observations','MatterSim or validated replay','Independent annotation'],
  'benchmark_executed':False},
 {'proposal_id':'P2','decision':'revise','feasibility':'Proceed with fixed-pool annotation audit; no need for large RSI training stack','critical_findings':[
  {'id':'10-P2-01','evidence_type':'INSPECTED CODE','claim':'DUET alternates ordinary and augmentation updates; pool size alone does not fix optimizer exposure.','source':'map_nav_src/r2r/main_nav.py:143–155','commit':duet},
  {'id':'10-P2-02','evidence_type':'INSPECTED CODE','claim':'RSI acceptance uses two SR/SPL rules, while accepted samples use source frames/actions and later mix spatial/STOP QA.','source':'evo/evo_rollout.py:1517–1599; evo/evo_utils.py:1035–1101','commit':rsi}],
  'required_revisions':['Screen/label validity after probability sampling; enriched four-cell samples do not estimate prevalence.','Cross-fit by scene and derived route family; preserve immutable source and verification trajectories.','Match human minutes including witness work and optimizer/replay/token budgets.'],
  'cheapest_decisive_test':'At equal acceptance rates, compare false acceptance/rejection against blinded validity labels and simple compatibility/quality scorers.',
  'go_gate':'Independent validity evidence improves selection beyond ordinary scorers under equal labels/effort, followed by matched navigation study.',
  'stop_gate':'Hard examples are annotation errors, witness-free scorer matches, or downstream gain disappears at matched budgets.',
  'resource_dependencies':['Fixed generated pool','Route images/features','Frozen navigator difficulty scores','Blinded semantic/witness annotations'],
  'benchmark_executed':False},
 {'proposal_id':'P3','decision':'revise','feasibility':'Defer training; technically implementable but high data/task-port risk','critical_findings':[
  {'id':'10-P3-01','evidence_type':'INSPECTED CODE','claim':'NavQ predictor is loaded separately then evaluated under no_grad with detached outputs; navigation loss does not train a newly attached predictor head in this path.','source':'map_nav_src/models/vlnbert_init.py:72–81; models/vilmodel.py:724–752','commit':navq},
  {'id':'10-P3-02','evidence_type':'INSPECTED CODE','claim':'Release predicts 512-dimensional features and integrates frontier predictions separately from visited-node observed past features in REVERIE.','source':'Q_pretrain/dataset.py:189–244; models/vilmodel.py:640–643; reverie/agent_obj.py:419–437','commit':navq}],
  'required_revisions':['Specify event-labelled continuation sampler/distribution and new dataset.','Explicitly pretrain the joint head or control changed end-to-end gradients.','Validate route-following port; match instruction queries, labels and capacity for sequence/future-QA baselines.'],
  'cheapest_decisive_test':'Oracle-representation decision headroom then learned event prediction from deployable observations under frozen candidate/continuation policy.',
  'go_gate':'Both representation headroom and predictable signal survive simple sequence/QA/history controls.',
  'stop_gate':'No oracle headroom, observational unidentifiability, capacity confound, or simpler predictors match.',
  'resource_dependencies':['Visual/caption features','Q checkpoint','Ordered-event dataset','Matched continuation labels','Validated navigation task port'],
  'benchmark_executed':False},
 {'proposal_id':'P4','decision':'revise','feasibility':'Release-level diagnostic supported; paper-level method and novelty claims not yet supported','critical_findings':[
  {'id':'10-P4-01','evidence_type':'INSPECTED CODE','claim':'Released memory uses color histograms and category-level hard negative veto, unlike paper contextual additive EEC; negative capacity defaults to20 while CLI10 controls positive bank.','source':'utils/rag_manager.py:61–99,277–302,354–392; PSG_Nav.py:246–251','commit':psg},
  {'id':'10-P4-02','evidence_type':'EXECUTED TEST','claim':'Actual terminal-write and read functions can insert/reject a stipulated-correct synthetic crop after control failure, even with a positive entry; caller clears goal and marks synthetic map region.','source':'experiments/10/source_check_results.json#psgnav'},
  {'id':'10-P4-03','evidence_type':'INSPECTED CODE','claim':'Native bank/window does not supply the proposed causal witness/track/controller trace; stored captions use goal category and negative retrieval crosses scene subfolders within dataset.','source':'utils/object_detection.py:289–324,543–564; utils/rag_manager.py:739–809','commit':psg}],
  'required_revisions':['Hold features/retrieval/veto fixed across write-policy arms before comparing richer representations.','Collect new observable traces and independently adjudicate natural harmful-write prevalence.','Beat no-write/quarantine/simple controller rule/witness-only and narrow against HiMeco/EmbodiSkill.','Report paper-faithful EEC reconstruction separately from source release; keep category vs instance tasks distinct.'],
  'cheapest_decisive_test':'Log-only natural traces plus controlled source replay; compare false writes and later correct-category suppression under identical observations.',
  'go_gate':'Natural harm is meaningful and deployable attribution improves over quarantine at matched recall/memory/feedback cost.',
  'stop_gate':'Rare harm, quarantine suffices, unavailable causal evidence, or stale-positive tradeoff removes net benefit.',
  'resource_dependencies':['Natural repeated-task traces','Category correctness/visibility labels','Controller/track observations','Compatible simulator and model servers','Explicit bank reset/order/feedback protocol'],
  'benchmark_executed':False}]
package={'assignment':'10','status':'Complete','cutoff':'2026-10-22','findings':findings,
 'additional_baseline_findings':[{'id':'10-B01','paper_id':'skillnav','evidence_type':'EXECUTED TEST','claim':'Actual AST dispatch with fake experts differs in6/8 items between batch-one and batch-eight and8/8 after reversing batch; no learned policy tested.','source':'experiments/10/source_check_results.json#skillnav'}, {'id':'10-B02','paper_id':'same','evidence_type':'INSPECTED CODE','claim':'Independent inspection corroborates batch-axis softmax in TaskMoeRouting.','source':'src/models/vilmodel.py:979–984','commit':'b74c57bbb189a0ed8c908e90b53d02dfbe417528'}],
 'executed_tests':'experiments/10/source_check_results.json','source_manifest':'evidence/10/source_manifest.json','asset_check':'evidence/10/asset_check_results.json',
 'priority':['P1 diagnostic','P2 audit','P4 release-level trace audit','P3 representation/learnability gate'],
 'limitations':['No model or simulator benchmark executed','No empirical gain or natural harmful-write/teacher-error prevalence established','Remote checkpoint listing is not checkpoint verification','Novelty judgments depend on independent assignment09']}
(ROOT/'evaluations/10_findings.json').write_text(json.dumps(package,indent=2),encoding='utf-8')
print(json.dumps({'source_files':len(entries),'proposals':len(findings),'all_exist':all((ROOT/e['local_path']).exists() for e in entries)},indent=2))
