from pathlib import Path
import csv
import hashlib
import json
import re

R = Path(__file__).resolve().parents[2]
E = R / 'evidence/06'
records = json.loads((E / 'sources.json').read_text(encoding='utf-8'))
records = [x for x in records if x['paper_id'] not in {'vla_calibration','reliablenav','safety_gating'}]

def add_source(pid, title, authors, venue, url, pages, review, locators, repo=None, commit=None, files=(), limitations=(), version='Retrieved 22 October 2026', full=True):
    p = E / 'papers' / (pid + '.pdf')
    records.append(dict(source_id='FU06-'+pid, paper_id=pid, title=title, authors=authors,
        year=2026, venue_or_status=venue, primary_url=url, version=version, cutoff='2026-10-22',
        retrieved='2026-10-22', local_pdf=str(p.relative_to(R)) if p.exists() else None,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,
        full_text_reviewed=full, reviewed_pdf_pages=pages,
        review_scope='Only the explicitly listed text sections and source files were reviewed; no benchmark execution.',
        precise_locators=locators, review=review, repo_url=repo, repo_commit=commit,
        code_files_inspected=list(files), code_executed=False, limitations=list(limitations)))

add_source('vla_calibration','Confidence Calibration in Vision-Language-Action Models',
 ['Thomas P. Zollo','Richard Zemel'],'TMLR, October 2026, as stated in final manuscript',
 'https://arxiv.org/abs/2507.17383v3','1-14,18-21',
 'AUTHOR METHOD: Mean selected-token probabilities estimate eventual episode success; instruction-paraphrase ensembles and action-wise Platt scaling recalibrate the score. PAPER EXPERIMENT: Fine-tuned token-based VLAs in LIBERO support calibration improvements, not physical safety. Temporal alignment uses realized episode length retrospectively; the proposed progress-aware intervention is qualitative and requires an online progress estimator. Rephrased actions are discarded: original instructions control execution. INSPECTED CODE: each dimension is fitted separately against the same terminal label; the experiment explicitly averages outputs despite a product constructor default. DEDUCTION: neither paraphrase-based confidence nor generic progress-conditioned monitoring is new. Separate risk discrimination, calibration and intervention utility under policy/scene shift.',
 '§§3.3-3.6; §4.2 pp.8-11; §4.4; §§6-7; Appendix B-C.1',
 'https://github.com/thomaspzollo/vla_calibration','ee1ad08266c803b90bd601cfa6618bdb87a2af79',
 ['vla_calibration/calibration.py:1-265','notebooks/recalibration.ipynb:cell 2','README.md'],
 ['Supplement pp.22-34 not reviewed; no notebook execution or raw rollout inspection; OpenReview verification page inaccessible.'],
 'v3, 16 October 2026')
add_source('reliablenav','ReliableNav: Uncertainty-Aware Zero-Shot Vision-Language Navigation in Continuous Environments',
 ['Zhanbo Shi','Lin Zhang','Shengjie Zhao','Yicong Zhou'],'IEEE TMM Early Access, 21 August 2026',
 'https://doi.org/10.1109/TMM.2026.3726310','Full paper inaccessible; publisher abstract and targeted official code only',
 'AUTHOR ABSTRACT: Atomic action reasoning and a reliability-check map address semantic and sampling uncertainty. INSPECTED CODE: clustered same-class landmark instances receive numbered image masks and trigger another subgoal query; geometry rejects nonnavigable choices. This is explicit disambiguation/replanning, not a verified probability-calibration mechanism. The released configuration defaults to replayed action/LLM logs for 100 R2R-CE episodes. DEDUCTION: generic uncertainty-triggered re-prompting is established prior art. Quantitative paper claims were not independently audited because full text was inaccessible.',
 'Publisher abstract; official README Usage; released map_handler.py and reliable_agent.py locators',
 'https://github.com/ZhanboShiAI/ReliableNav','5234e9f96292b319463df5ab2d10ab6340cced1e',
 ['src/reliable_agent.py:90-203','src/mapping/map_handler.py:243-291,547-683','src/config/r2r_baselines/test.yaml:1-41','readme.md'],
 ['Publisher full text blocked; project paper button says coming soon; full planner and LLM parser not audited; no API or benchmark calls.'],full=False)
add_source('safety_gating','Uncertainty-Calibrated Safety Gating for Vision-Language-Action Manipulation Under Domain Shift: Reliability Gains and Intervention-Efficiency Trade-Offs',
 ['Atef M. Ghaleb','Ali S. Allahloh','Sobhi Mejjaouli','Mohammed A. H. Ali','Adel Al-Shayea'],
 'Sensors 26(10),3140; 15 May 2026','https://doi.org/10.3390/s26103140',
 'HTML sections 3.2,4.1-4.3,5.3,6.1-6.4,9 and Tables2-7; partial text review',
 'AUTHOR METHOD: A calibrated failure-risk score selects normal control, pause/reobserve or a MoveIt fallback. PAPER EXPERIMENT: Two simulated manipulation tasks under structured visual shifts show an intervention-efficiency tradeoff; the more aggressive uncalibrated baseline has stronger raw success/collision results. Threshold sweeps are retrospective, not rerun policies; most interventions use fallback, and residual contacts occur during pause/fallback. The authors explicitly deny a formal safety guarantee. DEDUCTION: a calibrated three-way gate alone is not a new mechanism; equal intervention budgets and closed-loop testing are essential. No released executable implementation or underlying logs were inspected.',
 '§§4.1-4.3,5.3,6.1-6.4; Tables2-7',limitations=['Partial HTML review; access intermittent; data offered on request; no code or raw-data audit.'],full=False)

# Normalize accidental missing spaces in old draft prose without altering identifiers/URLs/code.
for x in records:
    for key in ['review','version','venue_or_status','precise_locators']:
        s=x[key]
        for a,b in {'a20m':'a 20 m','claiming99%':'claiming 99%','confidence1−δ':'confidence 1−δ','and32':'and 32','first8/recent48':'first 8/recent 48','after48frames':'after 48 frames','All933':'All 933','The20-pair':'The 20-pair','improves19/812':'improves 19/812','breaks78/71':'breaks 78/71','recovering173/126':'recovering 173/126','and20 physical':'and 20 physical','HM3D58.5/26.6':'HM3D 58.5/26.6','oracle-real-future62.5/32.8':'oracle-real-future 62.5/32.8','is30 trials':'is 30 trials','retrieved22October2026':'retrieved 22 October 2026','accepted15March2026':'accepted 15 March 2026'}.items(): s=s.replace(a,b)
        s=re.sub(r'\b(Table|Tables|Eq|Eqs|Figure|Algorithm)(\d)',r'\1 \2',s)
        s=re.sub(r'\b(Table|Tables|Appendix|Algorithm)([IVX]+)\b',r'\1 \2',s)
        s=re.sub(r'\b(ICLR|CVPR|ICML|IROS|CoRL|SMC)(20\d\d)',r'\1 \2',s)
        s=re.sub(r'(\d)(January|February|March|April|May|June|July|August|September|October|November|December)',r'\1 \2',s)
        s=re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December)(20\d\d)',r'\1 \2',s)
        s=re.sub(r',(?=[A-Za-z0-9])',', ',s)
        for a,b in {'is95.98':'is 95.98','versus78.34':'versus 78.34','and42.06':'and 42.06','pages834':'pages 834','Humanoids2026':'Humanoids 2026','py670-703':'py:670-703','cluster95%':'cluster 95%','steps2/13':'steps 2/13','of11':'of 11','version2025':'version 2025','p13 captions':'p. 13 captions','references16-17':'references 16-17','2.34pp':'2.34 pp','CI[-.99, 5.68]':'CI [-0.99, 5.68]'}.items():s=s.replace(a,b)
        x[key]=s
(E/'sources.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
cards=['# Assignment 06 follow-up source cards','',
 'Cutoff: 22 October 2026. These compact source summaries distinguish author reports, inspected code and deductions. Full-text review means the listed substantive sections were read, not that unlisted appendices or images were inspected. Limited-scope records explicitly identify inaccessible or partial material. No fetched project code or navigation benchmark was executed. Bibliography, PDF hashes, precise scopes and pinned code are in [sources.json](sources.json).','']
for x in records:
    cards += [f"## {x['source_id']}: {x['title']}",'',
       '; '.join(x['authors'])+'. '+x['venue_or_status']+'. '+f"[{x['version']}]({x['primary_url']}). Reviewed: {x['reviewed_pdf_pages']}.",'',x['review'],'',
       'Locators: '+x['precise_locators']+'.','', 'Limitations: '+'; '.join(x['limitations']),'']
(E/'source_cards.md').write_text('\n'.join(cards),encoding='utf-8')

by_id={x['source_id']:x for x in records}
rows=[]
def row(bid, originals, failure, followups, mechanism, locator, comparable, status, gap, confidence):
    repos=[]
    for fid in followups:
        s=by_id.get(fid)
        if s and s['repo_commit']: repos.append({'source_id':fid,'commit':s['repo_commit'],'files':s['code_files_inspected']})
    rows.append(dict(bottleneck_id=bid,original_papers=originals,precise_failure=failure,
      followup_paper=followups,mechanism=mechanism,evidence_locator=locator,repo_commit_and_files=repos,
      comparable_setting=comparable,status=status,remaining_gap=gap,confidence=confidence))

row('06-W01',['SC2-WM','NavMorph','HNR','UNeMo'],
 'Internal change, forecast error and probability of a bad navigation decision are different quantities.',
 ['FU06-ua_nwm','FU06-vla_calibration'],'Structured residual scoring; empirical success recalibration.',
 'dossiers/sc2wm.md:02-sc2wm-calibration; FU06-ua_nwm §3/supplement G; FU06-vla_calibration §§3-4',
 'Aerial image goals and manipulation do not match language-route navigation.', 'partial',
 'Test action ranking and risk after scene/policy shift under identical candidates and observable feedback.', 'High distinction; medium gap breadth')
row('06-W02',['Pathdreamer','DREAMWALKER','MonoDream','NWM'],
 'Longer prediction horizons can add expensive or decision-irrelevant error.',
 ['FU06-adaptive_rollout','FU06-neubay','FU06-pijepa'],'Adaptive training horizons; policy-guided planning.',
 'dossiers/monodream.md; Adaptive Rollout Appendices C,F; NEUBAY §4; PiJEPA Tables1-3',
 'Offline locomotion/RL prediction and recorded navigation goals differ from online VLN.', 'shifted',
 'Separate state-specific uncertainty value from a matched deterministic horizon schedule and candidate generator.', 'High for control requirement')
row('06-W03',['NWM','Look Before You Leap','PlaNet','Dreamer'],
 'Good recorded-goal trajectory agreement does not establish closed-loop route completion.',
 ['FU06-pijepa','FU06-worldmap'],'World-model candidate optimization; generated-map waypoint supervision.',
 'PiJEPA eval_octo_mppi_cast.py:670-703; WorldMAP TablesII-III',
 'Recorded future image goals or projected trajectory metrics; not a shared VLN-CE execution test.', 'uncertain',
 'Evaluate the same learned controller with true resets, identical goal access and collision/latency accounting.', 'High protocol distinction')
row('06-W04',['SGM','Pathdreamer','WMNav'],
 'Semantic priors can influence unseen-space decisions without calibrated support.',
 ['FU06-schrodinger_nav','FU06-imaginenavpp'],'Separate provenance buffers; discounted imagined semantics; selected history.',
 'dossiers/sgm.md; dossiers/wmnav.md:02-wmnav-recovery; Schrodinger AppendixC.3/Table3; ImagineNav++ §IV.J',
 'ObjectNav, with different low-level planners and observation access.', 'partial',
 'Measure false-prior correction and visibility-conditioned invalidation while retaining useful unknown-space hypotheses.', 'Medium; no universal failure claim')
row('06-W05',['WMNav'],
 'Ordinary minimum-map updates cannot reverse a false low value; identified goal cells are an exception.',
 [],'Existing goal-mask override.', 'WMNav commit57b74cee7a00b669733e8ca917f34a8aadfd6fd0; WMNav_agent.py:1263-1287',
 'Same released update rule.', 'resolved-in-setting',
 'Resolved only for identified goal cells; ordinary cells and physical performance remain separate.', 'High source-level scope')
row('06-W06',['SC2-WM'],
 'Ambiguity over whether any real next observation supplies correction.',
 [],'Delayed actual-panorama residual adaptation.', 'dossiers/sc2wm.md; commitf2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b ss_trainer_ETP.py:2246-2308; supplementpp15-16',
 'The inspected asynchronous release branch.', 'resolved-in-setting',
 'Do not propose observed residual correction as absent; evaluate alignment, update scope and actual decision risk.', 'High source-level scope')
row('06-U01',['KnowNo','Explore until Confident','AIUTA'],
 'Sequence/set coverage is often overread as conditional success or physical safety.',
 ['FU06-cpnav'],'Episode-level bottleneck-score admissible action sets.',
 'KnowNo AppendixA3; Explore §V-B; CPNAV Eqs11-15; evidence/06/math_checks_results.json:M1,M5',
 'Different scenario distributions, human help and local-control contracts.', 'uncertain',
 'Reconcile calibration unit and confidence direction; state search-budget, grounding and policy-shift assumptions.', 'High for logical boundary; empirical calibration unknown')
row('06-U02',['UAGM','CA-Nav','FSTTA','ATENA'],
 'Confidence may be low for recoverable states and high after irreversible failure.',
 ['FU06-vla_calibration','FU06-safety_gating'],'Temporal calibration analysis; policy/pause/fallback supervision.',
 'VLA Calibration §4.2; Safety Gating §§4,6.2-6.4; dossiers/atena.md',
 'Adjacent manipulation evidence; not a matched navigation benchmark.', 'partial',
 'Estimate intervention benefit using observable progress and equal-cost alternatives, with held-out policy/scene tests.', 'Medium; calibration alone already established')
row('06-M01',['IVLN','GSA-VLN','HAMT','GridMM'],
 'Adding persistent history does not ensure task-relevant retrieval.',
 ['FU06-memoir','FU06-dartvln','FU06-imaginenavpp'],'Imagination-guided retrieval; read-side decay; keyframe selection.',
 'Memoir TableV/§V.E; DART Eqs1-5; ImagineNav++ TableV; dossiers/ivln.md',
 'Different episode-order, training and controller contracts; only within-paper ablations are matched.', 'partial',
 'Hold memory bytes, vision calls and feedback fixed; test instruction-critical relations and harmful retrieval explicitly.', 'High within-paper support; medium generalization')
row('06-M02',['NavFoM','StreamVLN','NaVid','JanusVLN'],
 'Bounded visual context may discard instruction-critical evidence, but oldest-frame fallback exists.',
 ['FU06-janusvln','FU06-reinsvln'],'Geometric cache plus sampled RGB; fixed evidence revisits.',
 'dossiers/navfom.md; Janus evaluation.py:179-198,369; ReInsVLN TablesI-II/IV',
 'Pretrained model/data and diagnostic checkpoint access differ.', 'partial',
 'Disentangle initial grounding, more inference and actual recovery of missing evidence at matched checkpoints.', 'High attribution concern')
row('06-M03',['PolarMem','HAM-VLN','Context-Nav','SGM'],
 'A failed route or unseen object does not establish a persistent negative fact.',
 ['FU06-cmmrvln','FU06-memoir','FU06-schrodinger_nav'],'Outcome memory; future retrieval; source-separated mapping.',
 'dossiers/polarmem.md:POLAR-1; CMMR §III.C; Memoir §V.E; Schrodinger AppendixC.3',
 'Visual RAG, persistent VLN and ObjectNav have different evidence semantics.', 'unresolved-in-reviewed-scope',
 'Test view/pose/time validity and failure-cause attribution before retracting a memory; compare tri-state and source-tag baselines.', 'Medium; bounded literature, no universal absence claim')
row('06-M04',['GridMM','DUET'],
 'Anti-loop rules can penalize necessary recovery and decay need not remove obsolete evidence.',
 ['FU06-dartvln'],'Finite next-hop penalties and relative memory priors.',
 'DART agent.py:452-501, vilmodel.py:817-851; M4; DUET cached-stop/backtrack dossier',
 'Same-checkpoint discrete evaluation is relevant; continuous cost differs.', 'partial',
 'Construct necessary-return and stale-singleton cases; compare loop reduction against path-fidelity/goal loss.', 'High algebra/source; benchmark effect untested')
row('06-A01',['GSA-VLN','IVLN','NavMorph'],
 'Scene-persistent memory claims can be confused with label-free online weight adaptation.',
 ['FU06-memoir','FU06-cmmrvln'],'Cross-episode history retrieval.',
 'dossiers/gsavln.md:training/reset protocol; dossiers/ivln.md:oracle phases; Memoir TableV; CMMR §III',
 'Episode ordering, passive oracle transit and feedback access are not interchangeable.', 'partial',
 'Use a protocol vector and reset/order controls before claiming general autonomous lifelong adaptation.', 'High distinction')
row('06-A02',['FSTTA','FeedTTA','ATENA','SC2-WM'],
 'Confidence/feedback proxies can update parameters toward an incorrect policy.',
 ['FU06-vla_calibration'],'Empirical calibration offers a diagnostic, not an update guarantee.',
 'dossiers/fstta.md,feedtta.md,atena.md; evidence/04/isolated_check_results.json; VLA Calibration §6',
 'Entropy-only, external feedback, self-label and observed-residual updates must be compared separately.', 'uncertain',
 'Test feedback precision and selection shift with frozen-policy, no-update and identical-budget controls.', 'High need for controls; no new TTA method tested')
row('06-A03',['SAME','M3E','Continual VLN'],
 'Joint expert routing and sequential supervised retention address different failure settings.',
 [],'State routing versus supervised expert consolidation.',
 'dossiers/same.md:Tables4-7; dossiers/m3e.md:Tables2,4 and Eqs9-14',
 'SAME joint multi-task data; M3E labelled 80% new-domain train split.', 'partial',
 'Evaluate old-domain retention and false topology separately; workload is not causal parameter importance.', 'High for protocol; 04 code audit defines release limits')
row('06-P01',['Rethinking the Embodied Gap','NaVILA','Uni-NaVid','MapNav'],
 'Simulator navigation success need not survive morphology, tracking error and camera motion.',
 ['FU06-humanoidvln'],'Physics/embodiment benchmark.',
 'HumanoidVLN TableIII/§VI.D; corresponding original dossiers',
 'Morphology, controller and action interface co-vary; curated scenes.', 'partial',
 'Cross controller and perception perturbations while holding the high-level policy fixed.', 'High benchmark gap; architecture cause unresolved')
row('06-P02',['NWM','SGM','WMNav','SC2-WM'],
 'Planning evidence can depend on oracle geometry, offboard compute or different goal thresholds.',
 ['FU06-schrodinger_nav','FU06-pijepa','FU06-ua_nwm'],'Different closed/open-loop planning pipelines.',
 'Schrodinger AppendixE.1; PiJEPA eval source; UA-NWM Tables1/3 and hardware appendix; originals code/config dossiers',
 'No cross-paper leaderboard ranking accepted without common information/control contract.', 'shifted',
 'Report true continuous path, collisions, sensing, stops and total wall time under one controller.', 'High protocol concern')
row('06-P03',['NavNuances','GELA','VLN-Trans','CA-Nav'],
 'Reliability language can hide repeated LLM calls and geometric disambiguation assumptions.',
 ['FU06-reliablenav','FU06-reinsvln'],'Instance masking and replanning; history reminder control.',
 'ReliableNav map_handler.py:547-683; ReInsVLN §III; assignment05 review for instruction chains',
 'ReliableNav full paper unavailable; code demonstrates mechanism only.', 'uncertain',
 'Match query budgets, landmark inventory, localization and reminder content.', 'High code finding; low complete-paper coverage')
row('06-W07',['SGM'],
 'The released hard detached IoU term cannot supply its advertised loss gradient.',
 [],'Other BCE/MSE terms still train the map decoder.',
 'dossiers/sgm.md:02-sgm-iou_gradient; models_mae_cross.py:254-295',
 'Only the inspected release branch.', 'unresolved-in-reviewed-scope',
 'A differentiable surrogate needs a matched training ablation before a benchmark benefit is claimed.', 'High gradient structure; empirical effect unknown')
row('06-M05',['Memoir','GSA-VLN'],
 'Persistent retrieval succeeds less often when task clauses require distinct but visually similar places.',
 ['FU06-memoir','FU06-reinsvln'],'Future-guided behavioral matching; diagnostic evidence revisit.',
 'Memoir §V.E and TableV; graph_utils.py:419-531; ReInsVLN TablesI-II',
 'Memory-persistent and episodic VLM diagnostics differ; no direct numerical comparison.', 'partial',
 'Use swapped relational/ordinal requirements at fixed views and match within-scene feedback history.', 'Medium; promising measurable slice, novelty unestablished')

original_code = {
 '06-W05': ('57b74cee7a00b669733e8ca917f34a8aadfd6fd0',['src/WMNav_agent.py:1263-1287'],'dossiers/wmnav.md'),
 '06-W06': ('f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b',['vlnce_baselines/ss_trainer_ETP.py:2246-2308'],'dossiers/sc2wm.md'),
 '06-W07': ('579982dae2fe040d2fa26e82778352d950e31a19',['SGM_train/models_mae_cross.py:254-295'],'dossiers/sgm.md')}
for x in rows:
    x['evidence_locator']=x['evidence_locator'].replace('commit57','commit 57').replace('commitf2','commit f2').replace('supplementpp','supplement pp. ')
    if x['bottleneck_id'] in original_code:
        sha,files,record=original_code[x['bottleneck_id']]
        x['repo_commit_and_files']=[{'commit':sha,'files':files,'inspection_owner':'assignment02','record':record}]
(R/'followups/06_matrix.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
with (R/'followups/06_matrix.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
    for x in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in x.items()})

queries=[
 ('World-model scoring','vision language navigation world model uncertainty calibration decision risk 2026 October'),
 ('Memory correction','navigation memory retraction negative evidence hallucination confidence 2026'),
 ('Exact follow-up sources','WorldMAP PiJEPA JanusVLN HumanoidVLN UA-NWM CPNAV Adaptive Rollout Truncation official paper GitHub'),
 ('Persistent retrieval','"Memoir" "Dream to Recall" TPAMI 2026'),
 ('Memory/diagnostic releases','ReInsVLN DART-VLN CMMR-VLN ImagineNav++ Schrodinger Navigator official code'),
 ('Adjacent model-based RL','NEUBAY Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism'),
 ('Calibration follow-up','"Confidence Calibration in Vision-Language-Action Models" Zollo'),
 ('Navigation reliability','"ReliableNav" "Reliability Check Map"; "ReliableNav" pdf github'),
 ('Adjacent safety policy','"Uncertainty-Calibrated Safety Gating"; "Uncertainty-Calibrated Safety Gating" "Table 6"')]
log={'cutoff':'2026-10-22','search_date':'2026-10-22','scope':'Primary source searches, source citation trails and official implementation inspection; targeted, not exhaustive.',
 'query_record_note':'Earlier exact-title searches are reconstructed from saved retrieval scripts/source records; final calibration queries are literal issued queries.',
 'queries':[{'purpose':a,'query':b} for a,b in queries],
 'reviewed_sources':[{'id':x['source_id'],'url':x['primary_url'],'scope':x['reviewed_pdf_pages']} for x in records],
 'unreviewed_leads':[{'title':'VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions','url':'https://aclanthology.org/2026.acl-long.1152/','state':'Discovery only; passed to assignment05'},
 {'title':'Causal learning with uncertainty-aware transformer for vision-and-language navigation','url':'https://doi.org/10.1016/j.neucom.2025.132196','state':'Primary preview only; no mechanism or novelty claim based on it'}],
 'excluded_or_corrected':['PiJEPA and Adaptive Rollout are workshop papers, not verified main-conference publications.','Confidence Calibration final manuscript states TMLR; older under-review ICLR pages are not acceptance evidence.','Secondary blogs and aggregators were discovery aids only.','Inaccessible repositories do not establish nonexistence.']}
(E/'search_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'sources':len(records),'matrix_rows':len(rows),'source_review_words':{x['paper_id']:len(x['review'].split()) for x in records}}))
