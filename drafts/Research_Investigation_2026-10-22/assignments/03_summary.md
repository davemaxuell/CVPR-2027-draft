# Assignment03: VLM navigation, reasoning, memory and physical execution

Cutoff: **22 October 2026**. Status: **33/33 supplied entries have substantive dossiers and scoped source audits**. Primary paper pages, associated versions, repository SHAs, inspected files and access limits are recorded in [03_results.json](03_results.json). This means completion of the documented review scope, not full reproduction or inspection of every dependency and every appendix page.

## Findings for synthesis and proposal evaluation

The broad mechanisms already have strong prior art. AdaNav learns uncertainty-triggered reasoning; AwareVLN trains key-event reasoning; R³ switches from a fast trained expert to a frozen deliberative agent. Aux-Think trains on reasoning while omitting it from action-time inference. Progress-Think explicitly learns semantic instruction progress; IDEAL-VLN uses a compact semantic mediator and correction data. Generic selective reasoning, auxiliary CoT, progress tracking or failure-retrieval proposals therefore need much narrower novelty claims.

Instruction constraints and verified memory are related but distinct. CA-Nav advances a constraint queue using local checks and timeouts. GC-VLN solves geometric graph constraints and backtracks. Context-Nav checks whether one common viewpoint satisfies multiple relations. HAM-VLN stores advisory failed-branch memories. PolarMem records positive/negative/uncertain concepts. PSG-Nav combines categorical scene beliefs, multiple possible worlds and outcome-labeled retrieval. None of these observations alone proves that a failed visit establishes absence under partial visibility.

Data and information access dominate many comparisons. SkillNav's temporal-only DUET control nearly matches its full R2R gain. NaVid, NaVILA, Uni-NaVid, StreamVLN, CorrectNav and NavFoM differ substantially in training exposure, action interfaces and sensing. MapNav's mapper consumes depth and pose. NaVILA's whole robot consumes LiDAR/proprioception even though its high-level VLA sees RGB. EvoNav receives terminal ground-truth-derived metrics and PSG-Nav retains outcome-labeled experience across episodes; these should not be compared to fresh-agent one-shot systems without matched feedback/reset controls.

Useful evidence-backed opportunities concern: (1) whether a proposed reasoning or expert intervention actually changes a wrong action under a fixed compute budget; (2) retaining or retracting negative evidence based on observability and cause of failure; (3) preserving instruction-relevant history under fixed memory and sensor budgets; and (4) route-faithful recovery under matched exogenous perturbations. These remain candidate gaps within the reviewed scope, not universally unsolved claims. Later assignments must test the nearest current work, including the coordinator's CLOSER/VerNav leads.

## Release and reporting issues: baseline integrity, not scientific contributions

**SkillNav batch dispatch.** At commit `b56de12a6af07ce09c9252938b1fb69e55d33103`, `_get_input_embeds` in `skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py:647–664` assigns `self.vln_bert` separately while embedding each batch item. After that loop, the member refers to the last selected expert. `rollout` at line 838 applies that single navigation head to the entire batch. The released test script uses batch 8. The cheapest decisive baseline audit is equivalence between independent batch-one inference and a mixed-expert batch-eight call with identical examples/seeds. This is a source call-path deduction and an isolated toy illustration; no SkillNav checkpoint or simulator was run, and it does not establish which version produced published results. Fixing the release issue is not a new navigation mechanism.

SkillNav Appendix G/Table 13 reports Temporal DUET 82.59 SR/75.40 SPL versus full SkillNav 82.59/76.50. The union-data/temporal-only control is mandatory. Appendix Table 10 also prints Region DUET SR 36.67 and SPL 43.02; visual PDF inspection confirms the row. Standard SPL cannot exceed SR on identical episodes. The cause is unestablished, so do not use that row until clarified.

MapNav's `caculate_metrics.py:9–14`, commit `4a3acc70270688630b4ca8f6df39f26585898e0d`, turns failures within 3.2m into success and adjusted SPL 1.0 regardless of path length. It prints raw and adjusted metrics. This is not evidence that the paper used the adjusted numbers; enforce the standard evaluator for comparisons.

NavGRPO's invoked `agent_grpo` reward branches at 724–744 do not implement all reward components printed in its paper. PolarMem's release uses a fixed Otsu margin and weighted database ranking instead of the paper's ensemble/dispersion margin and lexicographic ordering. Context-Nav eventually soft-passes repeated Unknown attribute decisions and some unsupported spatial tests. PSG-Nav's failed-episode feedback marks recent detections as false positives even though failure can arise from control; its incidence is not measured here. Each finding has a pinned source trail in its dossier and requires empirical checking before attributing benchmark consequences.

## Corrections that must survive final synthesis

- NavFoM's BATS equation is an expected-budget rule, but Appendix A.2 specifies oldest-frame truncation at extreme horizons. The remaining question concerns evidence preservation and exact draw-level enforcement, not absence of a fallback. Table 11: RTX4090 FP 16 is19.8 GB/218 ms; RTX4090 4bit is10.7 GB/248 ms; Jetson Thor16bit is19.1 GB/566 ms.
- EvoNav supplement C.3 explicitly denies giving/storing ground-truth trajectories in the agent. It uses the agent's rollout plus terminal SR/SPL. Critique outcome feedback and ordering, not oracle-path input at inference.
- AwareVLN's supplement has contradictory RxR training prose versus Table 5's caption. Preserve that uncertainty. Its no-stopping-error ablation has higher R2R SPL than the full model despite lower SR.
- AO-Planner's47% R2R SR belongs to the trained distillation variant; the training-free variant reports 25.5%.
- NavGRPO is marked CVPR2026 Findings. CONSOLE is TPAMI2024. ECoT is CoRL2024 with proceedings published in 2025. CA-Nav's journal-template masthead is not a publication record. HAM-VLN and PolarMem are retained with verified preprint status rather than inferred conference acceptance.
- RL4VLA is manipulation evidence. Its extended appendix has visual and some semantic regressions despite stronger execution robustness; do not generalize its optimizer ranking to VLN.

## Executed checks and their limits

Only the standalone Python checks in [isolated_logic_checks.py](../evidence/03/isolated_logic_checks.py) were executed; results are [isolated_logic_results.json](../evidence/03/isolated_logic_results.json). They import no project models or simulators.

1. A two-expert dispatch toy returns `[10,20]` independently but `[20,20]` when the last expert is reused across the batch. It illustrates the inspected call-path mechanism, not actual SkillNav outputs.
2. A hypothetical failed3.1m MapNav episode has standard SPL 0; the helper's adjusted SPL is1. A path-weighted relaxed-success score would instead depend on path length (the example gives 0.25).
3. Four fusion weights produced by softmax(sigmoid(logits)) have pairwise ratio below e; the maximum possible weight is e/(e+3), approximately 0.4754. This is an algebraic property of the inspected SUSA fusion, not a measured performance limit.
4. A PolarMem weighted-score counterexample shows that five positive matches and one conflict can outrank two compatible matches. This tests ranking semantics, not retrieval accuracy.

The SkillNav Table 10 PDF crop was rendered and visually inspected. No benchmark, training run, API inference, downloaded checkpoint, physical robot or simulator experiment was executed. Root's8 GB laptop and missing Habitat/MatterSim installations are insufficient to reproduce the large-model training recipes without substantial changes.

## Inspection coverage and remaining access limits

All 33 paper entries have method/training/inference, novelty, result/protocol and bottleneck analysis. Twenty-five entries have actual executable-source inspection; R³ has a pinned README-only official release. Progress-Think and Aux-Think share one RoboOrchardLab snapshot. Aux-Think additionally has pinned linked VILA/VLN_CE inspections; NaVILA has pinned linked locomotion inspection. Phase coverage is explicitly limited where the release supplies only wrappers or inference, particularly Aux-Think, Progress-Think, MapNav, and several data-generation paths.

No executable official release was verified for AdaNav, CorrectNav, EvoNav, HAM-VLN, IDEAL-VLN, NavFoM or ProFocus. This is a scoped access finding, not a proof that code cannot exist. R³'s correct repository is IAII-CAS/navigation_R3, but its inspected snapshot contains only a short README.

NaVILA and Uni-NaVid local PDFs had no extractable text. Their accessible author arXiv versions were reviewed and saved in evidence/03; no byte-equivalence claim is made. Distinct AwareVLN, ProFocus and Progress-Think alternatives were checked on relevant pages. EvoNav's separate supplement was read. SUSA's alternate and EvoNav's alternate are exact SHA256 duplicates and were not reread. Detailed scope is recorded in each dossier.

## Assignment ledger

| Paper | Review status | Dossier |
|---|---|---|
| AdaNav | Complete within recorded evidence/access scope | [adanav](../dossiers/adanav.md) |
| Analogical Descriptions | Complete within recorded evidence/access scope | [analogical](../dossiers/analogical.md) |
| AO-Planner | Complete within recorded evidence/access scope | [aoplanner](../dossiers/aoplanner.md) |
| Aux-Think | Complete within recorded evidence/access scope | [auxthink](../dossiers/auxthink.md) |
| AwareVLN | Complete within recorded evidence/access scope | [awarevln](../dossiers/awarevln.md) |
| CA-Nav | Complete within recorded evidence/access scope | [ca_nav](../dossiers/ca_nav.md) |
| CONSOLE | Complete within recorded evidence/access scope | [console](../dossiers/console.md) |
| Context-Nav | Complete within recorded evidence/access scope | [context_nav](../dossiers/context_nav.md) |
| CorrectNav | Complete within recorded evidence/access scope | [correctnav](../dossiers/correctnav.md) |
| DiscussNav | Complete within recorded evidence/access scope | [discussnav](../dossiers/discussnav.md) |
| ECoT | Complete within recorded evidence/access scope | [ecot](../dossiers/ecot.md) |
| EvoNav | Complete within recorded evidence/access scope | [evonav](../dossiers/evonav.md) |
| GC-VLN | Complete within recorded evidence/access scope | [gc_vln](../dossiers/gc_vln.md) |
| HAM-VLN | Complete within recorded evidence/access scope | [ham_vln](../dossiers/ham_vln.md) |
| IDEAL-VLN | Complete within recorded evidence/access scope | [idealvln](../dossiers/idealvln.md) |
| MapGPT | Complete within recorded evidence/access scope | [mapgpt](../dossiers/mapgpt.md) |
| MapNav | Complete within recorded evidence/access scope | [mapnav](../dossiers/mapnav.md) |
| NavFoM | Complete within recorded evidence/access scope | [navfom](../dossiers/navfom.md) |
| NavGPT | Complete within recorded evidence/access scope | [navgpt](../dossiers/navgpt.md) |
| NavGPT-2 | Complete within recorded evidence/access scope | [navgpt2](../dossiers/navgpt2.md) |
| NavGRPO | Complete within recorded evidence/access scope | [navgrpo](../dossiers/navgrpo.md) |
| NaVid | Complete within recorded evidence/access scope | [navid](../dossiers/navid.md) |
| NaVILA | Complete within recorded evidence/access scope | [navila](../dossiers/navila.md) |
| PolarMem | Complete within recorded evidence/access scope | [polarmem](../dossiers/polarmem.md) |
| ProFocus | Complete within recorded evidence/access scope | [profocus](../dossiers/profocus.md) |
| Progress-Think | Complete within recorded evidence/access scope | [progressthink](../dossiers/progressthink.md) |
| PSG-Nav | Complete within recorded evidence/access scope | [psgnav](../dossiers/psgnav.md) |
| R3 | Complete within recorded evidence/access scope | [r3](../dossiers/r3.md) |
| RL4VLA | Complete within recorded evidence/access scope | [rl4vla](../dossiers/rl4vla.md) |
| SkillNav | Complete within recorded evidence/access scope | [skillnav](../dossiers/skillnav.md) |
| StreamVLN | Complete within recorded evidence/access scope | [streamvln](../dossiers/streamvln.md) |
| SUSA | Complete within recorded evidence/access scope | [susa](../dossiers/susa.md) |
| Uni-NaVid | Complete within recorded evidence/access scope | [uninavid](../dossiers/uninavid.md) |
