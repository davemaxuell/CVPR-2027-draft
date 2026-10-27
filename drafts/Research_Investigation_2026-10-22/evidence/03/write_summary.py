from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]; E=ROOT/'evidence'/'03'
records=json.loads((ROOT/'assignments'/'03_results.json').read_text(encoding='utf-8'))
assert len(records)==33 and len({r['paper_id'] for r in records})==33
allowed={'Resolved within a clearly specified setting','Partially addressed','Shifted into another cost or limitation','Still unresolved in the reviewed literature','Uncertain because evidence is insufficient'}
issues=[]
for r in records:
 assert r['full_title'] and r['authors'] and r['venue_or_status'],r['paper_id']
 assert r['code_executed'] is False
 for b in r['bottlenecks']:assert b['status'] in allowed,(r['paper_id'],b['status'])
 assert Path(r['dossier_path']).exists()
 for spec in r['code_files_inspected']:
  f=ROOT/'repositories'/'03'/r['paper_id']/spec.split(':')[0]
  if not f.exists():issues.append({'paper':r['paper_id'],'missing_cited_file':str(f)})
 if re.search(r'promptparagraphoutputtruncated|LocalprovidedPDF|Compositionalgeneralization',Path(r['dossier_path']).read_text(encoding='utf-8')):issues.append({'paper':r['paper_id'],'readability':'Unrepaired merged prose'})
assert not issues,issues

summary='''# Assignment03: VLM navigation, reasoning, memory and physical execution

Cutoff: **22 October2026**. Status: **33/33 supplied entries have substantive dossiers and scoped source audits**. Primary paper pages, associated versions, repository SHAs, inspected files and access limits are recorded in [03_results.json](03_results.json). This means completion of the documented review scope, not full reproduction or inspection of every dependency and every appendix page.

## Findings for synthesis and proposal evaluation

The broad mechanisms already have strong prior art. AdaNav learns uncertainty-triggered reasoning; AwareVLN trains key-event reasoning; R³ switches from a fast trained expert to a frozen deliberative agent. Aux-Think trains on reasoning while omitting it from action-time inference. Progress-Think explicitly learns semantic instruction progress; IDEAL-VLN uses a compact semantic mediator and correction data. Generic selective reasoning, auxiliary CoT, progress tracking or failure-retrieval proposals therefore need much narrower novelty claims.

Instruction constraints and verified memory are related but distinct. CA-Nav advances a constraint queue using local checks and timeouts. GC-VLN solves geometric graph constraints and backtracks. Context-Nav checks whether one common viewpoint satisfies multiple relations. HAM-VLN stores advisory failed-branch memories. PolarMem records positive/negative/uncertain concepts. PSG-Nav combines categorical scene beliefs, multiple possible worlds and outcome-labeled retrieval. None of these observations alone proves that a failed visit establishes absence under partial visibility.

Data and information access dominate many comparisons. SkillNav's temporal-only DUET control nearly matches its full R2R gain. NaVid, NaVILA, Uni-NaVid, StreamVLN, CorrectNav and NavFoM differ substantially in training exposure, action interfaces and sensing. MapNav's mapper consumes depth and pose. NaVILA's whole robot consumes LiDAR/proprioception even though its high-level VLA sees RGB. EvoNav receives terminal ground-truth-derived metrics and PSG-Nav retains outcome-labeled experience across episodes; these should not be compared to fresh-agent one-shot systems without matched feedback/reset controls.

Useful evidence-backed opportunities concern: (1) whether a proposed reasoning or expert intervention actually changes a wrong action under a fixed compute budget; (2) retaining or retracting negative evidence based on observability and cause of failure; (3) preserving instruction-relevant history under fixed memory and sensor budgets; and (4) route-faithful recovery under matched exogenous perturbations. These remain candidate gaps within the reviewed scope, not universally unsolved claims. Later assignments must test the nearest current work, including the coordinator's CLOSER/VerNav leads.

## Release and reporting issues: baseline integrity, not scientific contributions

**SkillNav batch dispatch.** At commit `b56de12a6af07ce09c9252938b1fb69e55d33103`, `_get_input_embeds` in `skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py:647–664` assigns `self.vln_bert` separately while embedding each batch item. After that loop, the member refers to the last selected expert. `rollout` at line838 applies that single navigation head to the entire batch. The released test script uses batch8. The cheapest decisive baseline audit is equivalence between independent batch-one inference and a mixed-expert batch-eight call with identical examples/seeds. This is a source call-path deduction and an isolated toy illustration; no SkillNav checkpoint or simulator was run, and it does not establish which version produced published results. Fixing the release issue is not a new navigation mechanism.

SkillNav Appendix G/Table13 reports Temporal DUET82.59 SR/75.40 SPL versus full SkillNav82.59/76.50. The union-data/temporal-only control is mandatory. Appendix Table10 also prints Region DUET SR36.67 and SPL43.02; visual PDF inspection confirms the row. Standard SPL cannot exceed SR on identical episodes. The cause is unestablished, so do not use that row until clarified.

MapNav's `caculate_metrics.py:9–14`, commit `4a3acc70270688630b4ca8f6df39f26585898e0d`, turns failures within3.2m into success and adjusted SPL1.0 regardless of path length. It prints raw and adjusted metrics. This is not evidence that the paper used the adjusted numbers; enforce the standard evaluator for comparisons.

NavGRPO's invoked `agent_grpo` reward branches at724–744 do not implement all reward components printed in its paper. PolarMem's release uses a fixed Otsu margin and weighted database ranking instead of the paper's ensemble/dispersion margin and lexicographic ordering. Context-Nav eventually soft-passes repeated Unknown attribute decisions and some unsupported spatial tests. PSG-Nav's failed-episode feedback marks recent detections as false positives even though failure can arise from control; its incidence is not measured here. Each finding has a pinned source trail in its dossier and requires empirical checking before attributing benchmark consequences.

## Corrections that must survive final synthesis

- NavFoM's BATS equation is an expected-budget rule, but Appendix A.2 specifies oldest-frame truncation at extreme horizons. The remaining question concerns evidence preservation and exact draw-level enforcement, not absence of a fallback. Table11: RTX4090 FP16 is19.8GB/218ms; RTX4090 4bit is10.7GB/248ms; Jetson Thor16bit is19.1GB/566ms.
- EvoNav supplement C.3 explicitly denies giving/storing ground-truth trajectories in the agent. It uses the agent's rollout plus terminal SR/SPL. Critique outcome feedback and ordering, not oracle-path input at inference.
- AwareVLN's supplement has contradictory RxR training prose versus Table5's caption. Preserve that uncertainty. Its no-stopping-error ablation has higher R2R SPL than the full model despite lower SR.
- AO-Planner's47% R2R SR belongs to the trained distillation variant; the training-free variant reports25.5%.
- NavGRPO is marked CVPR2026 Findings. CONSOLE is TPAMI2024. ECoT is CoRL2024 with proceedings published in2025. CA-Nav's journal-template masthead is not a publication record. HAM-VLN and PolarMem are retained with verified preprint status rather than inferred conference acceptance.
- RL4VLA is manipulation evidence. Its extended appendix has visual and some semantic regressions despite stronger execution robustness; do not generalize its optimizer ranking to VLN.

## Executed checks and their limits

Only the standalone Python checks in [isolated_logic_checks.py](../evidence/03/isolated_logic_checks.py) were executed; results are [isolated_logic_results.json](../evidence/03/isolated_logic_results.json). They import no project models or simulators.

1. A two-expert dispatch toy returns `[10,20]` independently but `[20,20]` when the last expert is reused across the batch. It illustrates the inspected call-path mechanism, not actual SkillNav outputs.
2. A hypothetical failed3.1m MapNav episode has standard SPL0; the helper's adjusted SPL is1. A path-weighted relaxed-success score would instead depend on path length (the example gives0.25).
3. Four fusion weights produced by softmax(sigmoid(logits)) have pairwise ratio below e; the maximum possible weight is e/(e+3), approximately0.4754. This is an algebraic property of the inspected SUSA fusion, not a measured performance limit.
4. A PolarMem weighted-score counterexample shows that five positive matches and one conflict can outrank two compatible matches. This tests ranking semantics, not retrieval accuracy.

The SkillNav Table10 PDF crop was rendered and visually inspected. No benchmark, training run, API inference, downloaded checkpoint, physical robot or simulator experiment was executed. Root's8GB laptop and missing Habitat/MatterSim installations are insufficient to reproduce the large-model training recipes without substantial changes.

## Inspection coverage and remaining access limits

All33 paper entries have method/training/inference, novelty, result/protocol and bottleneck analysis. Twenty-five entries have actual executable-source inspection; R³ has a pinned README-only official release. Progress-Think and Aux-Think share one RoboOrchardLab snapshot. Aux-Think additionally has pinned linked VILA/VLN_CE inspections; NaVILA has pinned linked locomotion inspection. Phase coverage is explicitly limited where the release supplies only wrappers or inference, particularly Aux-Think, Progress-Think, MapNav, and several data-generation paths.

No executable official release was verified for AdaNav, CorrectNav, EvoNav, HAM-VLN, IDEAL-VLN, NavFoM or ProFocus. This is a scoped access finding, not a proof that code cannot exist. R³'s correct repository is IAII-CAS/navigation_R3, but its inspected snapshot contains only a short README.

NaVILA and Uni-NaVid local PDFs had no extractable text. Their accessible author arXiv versions were reviewed and saved in evidence/03; no byte-equivalence claim is made. Distinct AwareVLN, ProFocus and Progress-Think alternatives were checked on relevant pages. EvoNav's separate supplement was read. SUSA's alternate and EvoNav's alternate are exact SHA256 duplicates and were not reread. Detailed scope is recorded in each dossier.

## Assignment ledger

| Paper | Review status | Dossier |
|---|---|---|
'''
for r in records:
 summary+=f"| {r['short_name']} | Complete within recorded evidence/access scope | [{r['paper_id']}](../dossiers/{r['paper_id']}.md) |\n"
(ROOT/'assignments'/'03_summary.md').write_text(summary,encoding='utf-8')
ledger='# Assignment03 source and inspection ledger\n\nCutoff:22 October2026. Code inspection is separate from execution. All project-code execution flags are false.\n\n'
for r in records:
 ledger+='## '+r['short_name']+'\n\n'
 ledger+=f"Title: {r['full_title']}\n\nAuthors: {', '.join(r['authors'])}\n\nPublication/status: {r['venue_or_status']}; year {r['year']}.\n\n"
 ledger+='Paper sources: '+', '.join(s['url'] for s in r['bibliography_sources'])+'\n\n'
 ledger+='Reviewed main pages: '+str(r['main_pdf_pages_read'])+'\n\n'
 ledger+='Repository: '+str(r['repo_url'])+'; commit `'+str(r['repo_commit'])+'`; status '+str(r['repository_status'])+'.\n\n'
 if r['code_files_inspected']:
  ledger+='Inspected files/ranges:\n\n'+''.join('- '+f+'\n' for f in r['code_files_inspected'])+'\n'
 ledger+='Limits: '+' '.join(r['access_limitations'])+'\n\n'
(E/'source_ledger.md').write_text(ledger,encoding='utf-8')
validation={'records':len(records),'unique_ids':33,'bibliography_fields_populated':True,'all_cited_local_code_files_present':True,'single_allowed_bottleneck_statuses':True,'known_merged_prose_patterns_absent':True,'project_code_executed':False,'standalone_logic_tests':'Executed; see isolated_logic_results.json','dossiers':len(records),'cutoff':'2026-10-22'}
(E/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(validation))
