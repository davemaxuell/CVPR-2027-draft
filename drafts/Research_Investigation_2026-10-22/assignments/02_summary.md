# Assignment 02 — world models, imagination and predictive navigation

Research cutoff: **2026-10-22**. All **33 assigned entries** have substantive dossiers and structured records in `02_results.json`. Included material was read from page-labelled full PDFs, with the actual reviewed page ranges recorded. VLN-SIG's separate supplement was reviewed. The two supplied SC²-WM PDFs were compared token by token: the only extracted-text difference is the first-page arXiv stamp; their substantive method, experiments and supplement are identical.

Implementation findings are linked to immutable commits in [02_repository_audit.md](02_repository_audit.md). These are scoped source inspections, not complete reproductions. Some repositories provide models/inference but omit training or downstream evaluation; several papers have no verified released implementation. These limitations are explicit per paper. No downloaded project code, simulator episode, checkpoint or training run was executed. Four isolated arithmetic/logic checks were executed and saved in `evidence/02/logic_check_results.json`; they establish only the properties described below.

The structured matrix contains **55 separate bottleneck records** with paper-specific conditions, causes, evidence and one scoped status each: 27 uncertain because evidence is insufficient, 13 still unresolved within the reviewed scope, 10 partially addressed, 4 shifted into another cost, and 1 resolved within an explicitly specified release setting. The last concerns SC²-WM's actual delayed feedback timing, not general reliability. These records are saved in `evidence/02/bottleneck_records.json`, repeated in `02_results.json`, and appended to each dossier. Matched follow-up evidence remains distinguishable from a related method with a higher score.

## The distinction that changes the research premise

**SC²-WM already has delayed real-observation residual adaptation.** Its immediate delta in Eq. 9 is a predicted next latent minus a current latent. That endogenous state change is not an observed prediction error. However, the supplement's forecast-to-real alignment and released trainer also compare previous predictions with newly observed visual features and adapt using this residual. Therefore “add observed prediction-error correction to an otherwise entirely open-loop method” is not a defensible novelty premise.

The remaining question is narrower: which observed residual predicts harmful action errors, how to distinguish uncertainty from ordinary state change, and whether adaptation improves decisions under controlled distribution shift. The release also differs from the paper in its MLP versus RSSM realization and trainable-parameter/trigger details. These are reproducibility questions requiring controlled execution, not evidence that the published gains are false. See [SC²-WM](../dossiers/sc2wm.md).

## Group papers by what predictions actually do

These categories overlap; they should not be treated as exclusive architectures.

| Mechanism / role | Papers in this assignment | Comparison that matters |
|---|---|---|
| Explicit online imagined views or scenes | Pathdreamer, DREAMWALKER, SALI, ImagineNav, GenEx | Real observation versus imagined evidence; supplied geometry; downstream scorer quality |
| Online geometric or semantic feature prediction | HNR, NavMorph, NavQ, UNeMo, SC²-WM, NavForesee | Candidate-action prediction versus goal templates; current-state features versus future aggregates |
| Auxiliary prediction mainly used in training | VLN-SIG, MonoDream, FantasyVLN | Same backbone/data with and without future supervision; deployment heads enabled versus disabled |
| Text-conditioned visual prototypes | Do Visual Imaginations Improve VLN Agents? | Prototype grounding and regularization versus actual unseen-state prediction |
| Latent control/planning foundations | World Models, PlaNet, Dreamer, Look Before You Leap, DINO-WM, CVAML | Model bias, value sufficiency, planning cost and policy-induced distribution shift |
| Generated video and joint action/observation modeling | NWM, Vid2World, LCVN, WorldVLN, CoT-VLA | Offline trajectory metrics versus executed control; real feedback frequency; action conditioning |
| Anticipated maps / exploration scores | Learning to Map, SGM, WMNav, WoMAP | Measured versus predicted cells; uncertainty meaning; confidence recovery; target visibility versus reachability |
| Structured belief or symbolic visual planning | Chasing Ghosts, Visual Planning | Belief over an ideal path versus physical dynamics; exact verifiers versus learned outcome checks |
| Adjacent semantic novelty detection | EOE | OOD class scoring is not a navigation transition or action-risk model |

Important corrections: NavQ's “Q” is a **512-dimensional discounted successor-semantic feature**, not a scalar reward value. WMNav uses prompted semantic curiosity and memory, not a learned next-image dynamics simulator. DINO-WM's zero-shot planning follows transition-model training. ImagineNav's open vocabulary does not remove trained waypoint/PointNav modules. EOE is the supplied ICML outlier-exposure paper; treating its acronym as a navigation exploration method would cite the wrong work.

## Strongest evidence about unresolved bottlenecks

1. **Prediction quality and decision quality can separate.** DREAMWALKER's oracle-distance experiment exposes a much larger gap than merely replacing generated views with true views. HNR's matched feature ablation is more informative than its leaderboard score. LCVN's separate world model has better visual fidelity yet worse unseen trajectory success than its unified model. A proposal must test useful action ranking or actual outcomes, not only FVD/LPIPS.

2. **Longer imagination or memory is not monotonically beneficial.** MonoDream's future horizon ablation deteriorates from one to three steps. NavQ peaks near gamma 0.5 rather than its largest tested value. NavMorph's largest memory is not best. SALI already persists imagined nodes, so persistent imagination alone is established prior art. None of these results alone proves a universal optimal horizon; each is scoped to its reported setting.

3. **Model exploitation is old and concrete.** World Models reports virtual versus real survival of 2086 versus 193 steps at low sampling temperature. PlaNet, Dreamer, DINO-WM and WoMAP change representations/objectives but do not establish calibrated off-distribution transition reliability. CVAML corrects a value-learning objective bias; its use of “calibrated” is not probability calibration or a safety guarantee.

4. **Observed, inferred and generated information need distinct treatment.** GenEx's closed imagined loops can be consistent without matching hidden physical facts. Conditional on fixed real history/model, generated samples add computation rather than independent sensor evidence. SGM already preserves observed cells over generated ones; Pathdreamer has an explicit prediction-to-memory flag. A new provenance contribution must exceed these existing distinctions.

5. **Uncertainty-guided exploration is established.** Learning to Map already uses ensemble variance for active collection and target-directed upper-confidence-style scoring. WoMAP has a reward-standard-deviation-weighted planning option. Neither establishes a finite-sample action-risk coverage guarantee. A proposal merely adding an ensemble or confidence penalty is insufficiently differentiated.

6. **Cross-episode adaptation changes the evaluation contract.** NavMorph explicitly acknowledges test-scene information accumulated across episodes. Compare episodic reset, scene-level carryover and continual streams separately; control episode ordering. A higher score after test-scene adaptation does not isolate architecture quality.

7. **Foundation priors, sensing and annotations are major confounders.** HNR adds 800 HM3D training scenes; NavQ optionally adds 1351 scenes; NavForesee uses teacher-VLM planning annotations; MonoDream uses privileged panoramic/depth targets; CoT-VLA uses actionless video pretraining. Compare these under matched data or explicitly study scaling, rather than attributing the whole gain to a novel module.

## Source discrepancies and independent checks

| Finding | Evidence and scope | Consequence |
|---|---|---|
| SC²-WM has both internal refinement and observed-residual adaptation | Main Eq. 9; supplement pp. 15–16; trainer 2246–2308 at pinned commit | Broad “no external correction” premise rejected |
| SGM's inspected IoU term cannot train the model | Main pretrainer imports `models_mae_cross`; loss thresholds and detaches its prediction before IoU | Released objective differs from a differentiable BCE+IoU description; actual loss includes weighted MSE/BCE |
| CVAML finite-sample correction depends on variance normalization | Released code uses population sample variance divided by N; exact enumeration of 16 independent ±1 samples at N=4 leaves 0.0625 bias, versus numerical zero with unbiased variance | Restricted arithmetic check; real policy-dependent critic requires a more careful test, not an accusation of benchmark failure |
| WorldVLN's common reward discount can cancel | Equations 9/13 and isolated standardized-reward arithmetic give identical normalized advantages after a common positive scale | Requires unreleased trainer grouping/epsilon reconciliation; no benchmark result inferred |
| NavQ shortest continuations are not mathematically unique | Diamond-graph counterexample; dataset uses one stored path per endpoint | Operational tie-breaking is valid, but a uniqueness claim needs its assumption |
| WMNav ordinary minimum merging cannot increase a low score | Source and simple 10→2→9 update produce 10→2→2 | Important exception: code resets identified goal-mask cells to 10; do not omit it |
| LCVN actor update is not automatically a Dreamer path gradient | Released actor uses log-probability times detached return because diffusion rollout is no-grad | Inspect REINFORCE and legacy fallback separately |
| NWM released CEM samples a restricted trajectory family | Planner repeats one displacement through the horizon and adjusts final yaw | Do not describe this release as arbitrary unconstrained action-sequence search |
| MonoDream code is available, but training coverage is limited | Official RoboOrchardLab release/project README and source | Correct initial unsuccessful search; released inference/evaluation is not a full auxiliary-loss training reproduction |

DREAMWALKER has apparent printed distance-reward sign inconsistencies and only a README in its pinned author repository. WoMAP prints a positive action-change term while calling it smoothness; the inspected gradient planner instead minimizes distance to reward one and does not contain that term. SALI's main-table and ablation SPL configurations differ materially. These uncertainties are documented rather than silently repaired.

## Priorities for the follow-up and synthesis assignments

The strongest candidate gaps are **decision-relevant prediction reliability**, **recovery from misleading memory under controlled shift**, and **goal-dependent sufficiency of compact future representations**. Each overlaps existing mechanisms and needs a narrow claim. Before proposing a new module, compare actual-outcome residuals, action entropy, ensemble disagreement and goal-conditioned decision error on a matched frozen policy. This inexpensive diagnostic can reject an appealing confidence signal before expensive training.

For experimentation, HNR/NavMorph/SC²-WM provide closely related continuous-navigation interfaces, but release/config discrepancies must first be reconciled. NavQ provides a clean discrete successor-feature target where changing language goals and equal-shortest-path ambiguity can be tested. Keep offline LCVN/NWM scores, manipulation CoT-VLA scores, aerial WorldVLN scores and ObjectNav scores in separate evaluation columns.

WorldMAP (arXiv 2604.07957), PiJEPA (CVPR 2026 workshop) and ImagineNav++ are current leads for assignments 05–06. Their existence prevents broad frontier claims; this assignment does not claim to have completed their full follow-up audit. Novelty confidence should remain provisional until those delegated reviews finish.

## Scope and completion ledger

The assigned 33 are: Chasing Ghosts; CoT-VLA; CVAML; DINO-WM; Dreamer; DREAMWALKER; EOE; FantasyVLN; GenEx; HNR; ImagineNav; LCVN; Learning to Map; Look Before You Leap; MonoDream; NavForesee; NavMorph; NavQ; NWM; Pathdreamer; PlaNet; SALI; SC²-WM; SGM; UNeMo; Vid2World; Do Visual Imaginations Improve VLN Agents?; Visual Planning; VLN-SIG; WMNav; WoMAP; World Models; WorldVLN.

All 33 have problem/method/training/inference/novelty/results/bottleneck dossiers, bibliography and reviewed-page records, initial follow-up relationships, and implementation or explicit access records. Source inspection is selective and phase coverage differs by release; the per-paper audits state the remaining limits. This is completion of assignment 02's evidence package, not a claim of full repository verification, experimental replication, or final novelty clearance.
