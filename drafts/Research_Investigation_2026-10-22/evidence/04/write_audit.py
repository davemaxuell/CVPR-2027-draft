import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'assignments/04_results.json').read_text(encoding='utf8'))
phase={
'activevig':'Exploration rollout, training reward/update and late-action reconstruction; no exhaustive physical replay audit.',
'aiuta':'VQA uncertainty and LLM dialogue prompts/branches; inherited navigation stack not exhaustively audited.',
'atena':'DUET outcome head, mixture loss, episodic adaptation, inference feedback and launcher configuration; hidden-test GPT-4o path not verified.',
'babywalk':'Instruction/memory model, curriculum data path and supervised rollout/training; no simulator execution.',
'continualvln':'Official supplementary algorithms, training losses and schedule; no executable source located.',
'cot':'Exact prompting recipe, output examples and resource statement; historical model/API execution unavailable.',
'embodiedgap':'Maintained author-linked RDP model/loss, action-chunk inference and configuration; historical paper commit equivalence unverified.',
'embodiedqa':'Navigation/QA model paths, reward/training and archived data protocol; House3D assets not tested.',
'environment_review':'Not applicable; internal secondary report.',
'explore_confident':'Map/exploration, VLM answer/relevance and final aggregation; calibration/intersection/stopping source not located.',
'feedtta':'Paper formulas, oracle prompt and independent algebra only; official implementation not located.',
'fgr2r':'Linguistic chunker and dataset alignment/release documentation; navigator/loss/train/eval source absent from inspected tree.',
'fstta':'Fast/slow adaptation, covariance/eigenvalue path, losses and launch configuration; documented release defects retained.',
'gela':'Grounding objectives, pretraining modules/configurations and HAMT fine-tuning path.',
'gsavln':'Persistent graph memory, per-scene reset, episode reset and inference/data loading; benchmark generation is paper evidence.',
'ieve':'Exploration training entry point and verification/exploitation thresholds/control branches; supplementary boundary differs from active code.',
'iqa':'Hierarchical planning/memory, controller supervision and training/evaluation entry points; simulator not initialized.',
'knowno':'Released calibration/evaluation notebooks read as text; sampled coverage/assumed successful help is not robot runtime validation.',
'landmark_rxr':'Repository tree/README and dataset links only; no model/loss/train/eval source at pinned commit.',
'mtrack':'Independent 05 model/CRF, reachability loss, rollout/training, checker and evaluator audit; incomplete runtime and AST defect.',
'm3e':'Independent 05 macro/micro model, workload/consolidation, CE training, data/evaluation/configuration and compile-only check.',
'metaexplore':'Full main and relevant supplementary method/results; no official executable model source located.',
'mind_error':'Error labels/sensors, detection/localization head, BCE/CE objectives and frozen-navigation training phase.',
'walkreadless':'BGP/BTP/VPP model/inference paths and configs; released paper-equivalent hyperparameters unconfirmed.',
'navnuances':'Independent 05 audit of six evaluators, eval entry point and geometry utilities; no benchmark executed.',
'omniagent':'TAURA reward/advantage weighting and trainer integration; bounded video-tool entry-point inspection, not entire training framework.',
'same':'Expert model/router, training task configuration and data setup; batch-axis softmax discrepancy.',
'slow4fast':'Documented agent entry point and separate reflection/retrieval/embedding modules; explicit placeholders and missing integration.',
'uln':'Independent 05 dual text encoders, uncertainty labels/training and physical branch scoring/inference; HAMT incomplete.',
'uot':'Question-tree generation, uncertainty and expected-reward recursion; no training component claimed.',
'vlnmme':'Diagnostic dataset transformations/prompts, scoring and inference configuration; no closed-loop navigation run.',
'vln_trans':'Latent translator, auxiliary objectives, navigator and training configurations; raw SyFiS generation builder not located.',
'vsi_bench':'Benchmark answer parsing, numerical scoring and aggregation plus release/version documentation; no trained benchmark model.',
}
lines=['# Assignment 04 implementation and access audit','',
'Cutoff: 22 October 2026. The table describes source actually inspected, not every file downloaded. All fetched project code remained unexecuted. Paper-specific files and line ranges are in each linked dossier and `04_results.json`. Commit pins refer to inspected snapshots, not necessarily the private code or historical commit that generated a publication.','',
'The 05 contributions below are original-source assistance within the existing ten-assignment ledger. They are attributed to that assignment; no extra agent assignment was created.','',
'| Paper | Inspected commit / access | Component coverage and limitations | Status |','|---|---|---|---|']
for x in r:
 pin=f"[{x['repo_commit'][:10]}]({x['repo_url']}/tree/{x['repo_commit']})" if x['repo_commit'] else 'No pinned model implementation; see access record'
 lines.append(f"| [{x['short_name']}](../{x['dossier_path']}) | {pin} | {phase[x['paper_id']]} | Complete within stated access scope |")
lines.extend(['','## File-level provenance','',
'Official snapshots are in `repositories/04/<paper_id>`, except M-Track source in `repositories/05/M-Track`. Its 04 metadata points to that snapshot. Raw LF line ranges are recorded by `evidence/04/raw_source_reads.jsonl`. Older reads in `source_read_ranges.json` used universal-newline text; some downloaded Windows files have doubled carriage returns, so prose citations added in closeout use raw LF/GitHub line numbering. Do not interpret downloaded-file manifests as inspection manifests.','',
'Independent audit records: [M-Track](../evidence/05/code_audit.md), [NavNuances](../evidence/05/navnuances_code_audit.md), [ULN](../evidence/05/uln_code_audit.md), [M³E](../evidence/05/m3e_original_audit.md). Compile/AST checks are documented separately in `evidence/05/mtrack_static_results.json` and `m3e_static_results.json`; no project module was imported.','',
'## Supplements and versions','',
'The associated-material arrays in each result identify exact reviewed pages. Local supplements and SHA256/download URLs are in `evidence/04/supplements/manifest.json`, `extra_manifest.json`, `zip_manifest.json` and `closure_manifest.json`. Failed initial archive parsing for IEVE was resolved by extracting its four-page PDF; its accompanying video was not reviewed. Apple metadata files in ZIP archives are not supplementary papers.','',
'Reviewed separate or recovered appendices: GELA (5 pages), ATENA (7), SAME (3), Meta-Explore (methods/results pp. 1–9 of 10), Landmark-RxR (9), EmbodiedQA (10), AIUTA (10), Embodied Gap (7), Continual VLN (6), IEVE (4), VSI-Bench (16), and Explore until Confident arXiv v2 pp. 15–17. Embedded appendices are listed in the individual dossiers. CoT prompt/error/resource appendices were read in the supplied full manuscript; the official 29-page supplementary PDF covers the same appendix topics, while the mentioned output ZIP was not located through the checked official link.','',
'The official slow4fast-VLN CVPR page links its main paper and arXiv record but no supplement; the conventional supplementary PDF URL returned 404. No separate supplement was located through checked primary links/tree for Active Visual Information Gathering, IQA or Mind the Error. Those gaps remain explicit instead of being described as reviewed materials.','',
'## Interpretation limits','',
'Source defects establish limits on direct reproduction at an inspected commit. They do not prove that reported experiments used defective code. Static loss inspection does not establish convergence, numerical stability or actual benchmark accuracy. Figures reviewed through captions/text were not independently relabeled; the slow4fast-VLN Table 5 discrepancy was checked against a rendered PDF page. No GPU training, external model inference, simulator installation, benchmark run or physical robot trial was performed.','',
'The final schema/coverage validation is [validation.json](../evidence/04/validation.json). The implementation investigation is complete within these explicit limits; there is no unreported running experiment.'])
(ROOT/'assignments/04_repository_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
search={'cutoff':'2026-10-22','scope':'Primary-source discovery and access checks. This closeout log records the exact final search queries and confirmed primary links; earlier discovery is represented by pinned source records and dossier bibliography.',
'queries':[
 {'query':'Explore Until Confident Efficient Exploration Embodied Question Answering appendix supplement 2024','result':'Recovered complete arXiv v2 PDF and reviewed appendices A–C.','primary_urls':['https://arxiv.org/pdf/2403.15941v2','https://explore-eqa.github.io/']},
 {'query':'IQA Visual Question Answering Interactive Environments supplemental 2018','result':'Official CVPR page and complete arXiv v3 expose no separate appendix link.','primary_urls':['https://openaccess.thecvf.com/content_cvpr_2018/html/Gordon_IQA_Visual_Question_CVPR_2018_paper.html','https://arxiv.org/html/1712.03316v3']},
 {'query':'Towards Open Environments Instructions General Vision Language Navigation Slow4Fast VLN supplementary 2026','result':'Official CVPR page confirmed; no supplemental link, guessed conventional PDF returned 404.','primary_urls':['https://openaccess.thecvf.com/content/CVPR2026/html/Li_Towards_Open_Environments_and_Instructions_General_Vision-Language_Navigation_via_Fast-Slow_CVPR_2026_paper.html']},
 {'query':'Active Visual Information Gathering Vision Language Navigation ECCV 2020 appendix','result':'ECVA main-paper page confirmed; pinned official tree contained no supplement PDF.','primary_urls':['https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4046_ECCV_2020_paper.php']},
 {'query':'site.proceedings.neurips.cc "Chain-of-Thought Prompting Elicits" 2022 supplemental','result':'Official supplemental link is a 29-page PDF; the claimed output ZIP was not located through this link.','primary_urls':['https://proceedings.nips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Supplemental-Conference.pdf']},
 {'query':'"Native Active Perception as Reasoning" ICML 2026 proceedings','result':'Official ICML/PMLR publication, authors and pages verified.','primary_urls':['https://proceedings.mlr.press/v306/xing26g.html']},
 {'query':'"Mind the Error" "supplementary" navigation','result':'No distinct supplement located in the returned primary author paper or pinned official tree.','primary_urls':['https://github.com/intelligolabs/R2RIE-CE']}
]}
(ROOT/'evidence/04/search_log.json').write_text(json.dumps(search,ensure_ascii=False,indent=2),encoding='utf8')
print('Saved final 04 audit and closeout search log.')
