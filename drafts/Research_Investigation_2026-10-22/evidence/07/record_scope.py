from pathlib import Path
import json, hashlib, datetime

root=Path(__file__).resolve().parents[2]
out=root/'evidence/07'
sources=[
 {'source_id':'07-planning_sufficiency','paper_id':'planning_sufficiency','title':'What Must a World Model Distinguish for Planning?','authors':['Rongzhe Wei','Hans Hao-Hsun Hsu','Peizhi Niu','Yifan Li','Pan Li'],'year':2026,'venue_or_status':'arXiv preprint','primary_url':'https://arxiv.org/abs/2609.33030v1','version':'v1, 26 September 2026','reviewed_pdf_pages':[1,3,4,5,6,7,8],'precise_locators':'Sections 3-6; definitions and candidate-search/selection/interface experiments','limitations':['Remaining results, appendices, proofs and code not fully audited.']},
 {'source_id':'07-compactnav','paper_id':'compactnav','title':'Towards Minimal Sufficient Representation for Vision-Language Navigation','authors':['Yihao Wu','Chenyi Xu','Liqi Yan','Chenhuan Cai','Geyong Min','Bin Lin','Fangli Guan','Jianhui Zhang','Pan Li'],'year':2026,'venue_or_status':'arXiv preprint; reviewed PDF calls model CompactNav','primary_url':'https://arxiv.org/abs/2607.23181v1','version':'v1, 25 July 2026','reviewed_pdf_pages':[1,3,4,5,6],'precise_locators':'Section 3.1-3.4; equations 1-11','limitations':['Abstract-page search uses Dual-Brain/BrainNav but downloaded v1 PDF and HTML use CompactNav.','Appendix proofs,full experiments and code not audited.']},
 {'source_id':'07-semantic_world_models','paper_id':'semantic_world_models','title':'Semantic World Models','authors':['Jacob Berg','Chuning Zhu','Yanda Bao','Ishan Durugkar','Abhishek Gupta'],'year':2025,'venue_or_status':'Reviewed arXiv preprint; conference status not independently verified','primary_url':'https://arxiv.org/abs/2510.19818v1','version':'v1, 22 October 2025','reviewed_pdf_pages':[1,2,3,4,5],'precise_locators':'Section 3.1-3.3,training future QA and planning likelihoods','limitations':['Quantitative experiments and implementation not audited.']},
 {'source_id':'07-ground_compose_reinforce','paper_id':'ground_compose_reinforce','title':'Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data','authors':['Andrew Li','Toryn Klassen','Andrew Wang','Parand Alamdari','Sheila McIlraith'],'year':2025,'venue_or_status':'NeurIPS 2025 according to primary arXiv record; reviewed v2','primary_url':'https://arxiv.org/abs/2507.10741v2','version':'v2, 26 October 2025','reviewed_pdf_pages':[1,3,4,5,6],'precise_locators':'Sections 3-6; Definitions 1-2, Algorithm 1','limitations':['Experiments,extensive appendices and implementation not audited by assignment 07.']},
 {'source_id':'07-negative_information','paper_id':'negative_information','title':'Negative Information and Proprioception in Monte Carlo Self-Localization for a 4-Legged Robots','authors':['Jan Hoffmann','Michael Spranger','Daniel Göhring','Matthias Jüngel'],'year':2005,'venue_or_status':'Author-hosted primary paper; year inferred from source path; exact workshop record unverified','primary_url':'https://faculty.kfupm.edu.sa/COE/mayez/ps-coe499/core/Vision-Localization/2005-08-ijcaj05ws-HoffmannSprangerGoehringJuengel-NegativeInformation.pdf','version':'Author-hosted PDF','reviewed_pdf_pages':[1,5,6],'precise_locators':'Section 4, Algorithm 1,equations 7-10','limitations':['Exact publication record,full experiments and historical implementation not audited.']},
 {'source_id':'07-dataset_cartography','paper_id':'dataset_cartography','title':'Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics','authors':['Swabha Swayamdipta','Roy Schwartz','Nicholas Lourie','Yizhong Wang','Hannaneh Hajishirzi','Noah A. Smith','Yejin Choi'],'year':2020,'venue_or_status':'EMNLP 2020','primary_url':'https://aclanthology.org/2020.emnlp-main.746/','version':'Proceedings metadata/abstract','reviewed_pdf_pages':[],'precise_locators':'Primary abstract and metadata only','limitations':['Full paper and implementation not reviewed; novelty lead.']},
 {'source_id':'07-learning_to_reweight','paper_id':'learning_to_reweight','title':'Learning to Reweight Examples for Robust Deep Learning','authors':['Mengye Ren','Wenyuan Zeng','Bin Yang','Raquel Urtasun'],'year':2018,'venue_or_status':'ICML 2018','primary_url':'https://proceedings.mlr.press/v80/ren18a.html','version':'Primary record and author method page','reviewed_pdf_pages':[],'precise_locators':'Primary abstract;author webpage displayed method/experiment portions','limitations':['Full appendix and code not reviewed;comparison lead.']}
]
for s in sources:
 s['cutoff']='2026-10-22';s['review_scope']='Scoped preparatory overlap check, not a full independent novelty audit';s['code_executed']=False;s['code_files_inspected']=[];s['repo_commit']=None
 f=out/(s['paper_id']+'.pdf')
 if f.exists():s['local_pdf']=str(f.relative_to(root));s['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
(out/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
m=json.loads((out/'additional_source_manifest.json').read_text(encoding='utf-8'))
lookup={s['source_id']:s for s in sources}
for row in m:
 s=lookup[row['id']];row['review_status']='Scoped sections read; not full-paper audit';row['reviewed_pdf_pages']=s['reviewed_pdf_pages'];row['limitations']=s['limitations']
(out/'additional_source_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')

files=[
 ('repositories/01/duet/map_nav_src/r2r/agent.py','93e8b233164bc079a6db48b8a0a78d123ec8de41','239-285,400-435','Teacher and loss path'),
 ('repositories/07/duet/data_utils.py','93e8b233164bc079a6db48b8a0a78d123ec8de41','1-46','Whole annotation loader'),
 ('repositories/02/navq/source/code/Q_pretrain/dataset.py','9465fe43c9c1583bd7ba390b6be30996ea4f5a5c','155-250','Candidate and expected future target construction'),
 ('repositories/02/navq/source/code/Q_pretrain/GPT_VL.py','9465fe43c9c1583bd7ba390b6be30996ea4f5a5c','190-247','Padded inputs, prediction head and MSE'),
 ('repositories/03/psgnav/PSG_Nav.py','181d878ff207845a31e542a19a6592925c1e9bab','1480-1562','Outcome-dependent positive/negative memory write'),
 ('repositories/05/RSI-Nav/evo/evo_rollout.py','5e8fa2c74cf8e7dc305b33088a963b200e20805d','1490-1540,1570-1600','Filter outcomes and source-path supervision'),
 ('repositories/05/RSI-Nav/evo/evo_utils.py','5e8fa2c74cf8e7dc305b33088a963b200e20805d','1013-1105','Accepted data queue and additional QA mixing')]
scope={'cutoff':'2026-10-22','role':'Synthesis of team evidence with targeted independent primary inspections; not rereading every PDF/repository',
 'supplied_entries_in_coverage':132,
 'team_evidence_inputs':['assignments/01_results.json','assignments/02_results.json','assignments/03_results.json','assignments/04_results.json','followups/05_review.md','followups/05_matrix.json','followups/06_review.md','followups/06_matrix.json','evidence/claims.json'],
 'selected_full_dossiers_read':['navq','mtrack','fgr2r','correctnav','sc2wm','polarmem','psgnav'],
 'selected_supplied_primary_pages':{'correctnav':[4],'srdf':[5],'navq':[5,6],'mtrack':[8],'hamt_extended_pdf':[9,16],'gsavln':[16,'17: first part of A.2'],'metaexplore_supplement':[4]},
 'additional_dossier_sections_read':{'hamt':['Mechanism and supplementary task protocol','Novelty','Results','Implementation scope','Bottlenecks'],'gsavln':['Mechanism','Results','Construction controls'],'metaexplore':['Full final dossier'],'eoe':['Final coordinator supplement review and scope','Main mechanism, code and transfer limits']},
 'source_code_inspections':[],
 'executed_tests':[{'script':'evidence/07/mechanism_checks.py','result':'evidence/07/mechanism_check_results.json','scope':'Three original finite graph/arithmetic examples, no project imports or benchmarks'}],
 'completion':'Final assignment 04 records and all final assignment 05 evidence incorporated after coordinator gate; assignment 07 synthesis complete.',
 'pending':'Independent evaluations 08–10; no novelty certification or proposed-system empirical validation'}
for file,commit,lines,purpose in files:
 f=root/file
 scope['source_code_inspections'].append({'path':file,'commit':commit,'lines':lines,'purpose':purpose,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'executed':False})
(out/'inspection_scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
queries=[
 'vision language navigation instruction progress reward machine temporal logic recovery history dependent obligations',
 'world model successor features language conditioned decision sufficient representation navigation semantic averaging',
 'navigation memory negative evidence object visibility false absence belief revision',
 'site.arxiv.org navigation reward machines language',
 'site.roboticsproceedings.org negative evidence object search',
 'site.arxiv.org navigation belief revision visibility',
 'site.aclanthology.org Dataset Cartography Mapping and Diagnosing Datasets with Training Dynamics 2020',
 'site.proceedings.mlr.press Learning to Reweight Examples for Robust Deep Learning 2018',
 'site.arxiv.org data selection valid difficult noisy samples vision language navigation']
search={'cutoff':'2026-10-22','queries':queries,'primary_records_consulted':[s['primary_url'] for s in sources],
 'scope':'Additional proposal-construction overlap search. Non-primary results used only as discovery leads. No absence-of-prior-art claim.',
 'unreviewed_leads':['Mind the Tail: Data Selection Through Sparse Concepts,arXiv 2610.21007','Fully Learnable Neural Reward Machines,arXiv 2509.19017','Leveraging Temporal Structure in Task Specifications,ICRA 2022'],
 'full_paper_reviews_claimed':False}
(out/'search_log.json').write_text(json.dumps(search,ensure_ascii=False,indent=2),encoding='utf-8')
print('Recorded7 sources,7 pinned code inspections and3 finite checks.')
