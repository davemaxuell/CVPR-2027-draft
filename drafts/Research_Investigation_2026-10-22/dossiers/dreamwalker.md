# DREAMWALKER

## Bibliography and reviewed material

**DREAMWALKER: Mental Planning for Continuous Vision-Language Navigation**. Hanqing Wang; Wei Liang; Luc Van Gool; Wenguan Wang. ICCV 2023 (2023). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-7. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- DREAMWALKER.pdf; Main PDF; SHA256 `49719ea2263fc70b3eafd944b7a2f97c10ecac7f9c4ddd1cb927b1a454392d8f`; 11 pages.

## Problem, task and assumptions

Long-horizon decision making in freely traversable continuous environments; assumes 360°RGB-D, a pretrained waypoint predictor and low-level 15°turn/0.25 m forward actions. Success means stopping within 3 m, at most 500 steps.

## Method, training and inference

Sections 3.1–3.2 pp. 3–6: episodic environment graph stores visited panorama features and geometric edges; a scene synthesizer projects observed RGB-D into a candidate camera frame and inpaints a plausible panorama via a Pathdreamer-style two-stage generator. Imagined observations recursively extend possible paths. MCTS selects using UCT Q+C√(logN/N_a), expands a node, rolls out to horizon K, and backs up discounted reward. A language-conditioned graph-attention network estimates distance-to-go for each graph node; this supplies search rewards and rollout preference. Execute the best root action, observe reality, update the graph, and replan. Distance model training uses true and randomly synthesized nodes with L2 shortest-distance supervision; scene generator/waypoint model are separately trained. Default 50 search iterations, horizon 4, discount 0.98. The policy is not trained end-to-end through a differentiable search.

## Novelty and closest predecessors

The paper integrates an explicit episodic graph and scene generator as a discrete mental testbed for MCTS in VLN-CE; Pathdreamer supplied visual forecasting and Look Before You Leap supplied earlier imagined-planning ideas. The contribution is their structured continuous-navigation integration, not first world-model navigation.

## Results and independent verdict

PAPER EXPERIMENT: Table 2 p. 7 val-unseen SR 49/SPL 44; oracle imagined views 54/49; nearest-memory copied views 27/24; greedy distance selection 42/36. Fig. 4 replacing distance predictions with ground truth reaches 100%SR at full replacement, revealing evaluator quality as a major bottleneck rather than only imagery fidelity. Table 1 test SR 49/SPL 44. Search compute increases with iterations, with diminishing gains beyond 50; p. 7 reports typical response within 1.5 s, not independently reproduced.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/hanqingwangai/Dreamwalker. Snapshot commit: `ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5`. Retrieval: 2026-10-22.

The author repository was pinned and its README inspected; the snapshot contains a README rather than the model, losses, trainer, preprocessing or evaluator. Consequently the paper mechanism is reviewed, but its implementation is unavailable for this audit.

Inspected files: `README.md`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Learned distance/value errors can dominate scene errors; oracle world improves 5 SR whereas oracle distance is much stronger but privileged. Eq. 11 prints D (s′)−D (s) while saying progress toward goal is rewarded, and Eq. 12 softmaxes positive distances; these apparent sign inconsistencies require released implementation or author clarification. The official URL currently exposes only documentation in retrieved snapshot, so behavior cannot be established from equations alone. HNR directly improves representation and branch evaluation, but does not guarantee reliable goal scoring.

Relevant follow-up relationship: HNR Table 3; NavQ's future-distance supervision; SC²-WM. Strong current gap is calibrated decision relevance of predictions, not an absence of imagination.

## Access and uncertainty record

Full text is preserved under `corpus/dreamwalker/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/dreamwalker/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-dreamwalker-scoring

**Partially addressed — Methodological.** Distance prediction remains a stronger bottleneck than future image fidelity in the paper intervention.

Conditions: Unseen continuous VLN episodes requiring branch ranking. Cause: The distance evaluator scores imagined states; a visually reasonable branch can still receive the wrong task-progress score.

Evidence: PAPER EXPERIMENT: Table 2 p.7 gives SR 49 normally versus 54 with oracle views; Figure 4 reaches 100% with full privileged distance replacement. Oracle interventions are not deployable methods.

Follow-up assessment: HNR improves fixed-policy future features; NavQ adds semantic/distance supervision. Neither supplies a matched resolution of DREAMWALKER distance calibration. Confidence: Medium.

### 02-dreamwalker-reproduction

**Uncertain because evidence is insufficient — Implementation-specific.** The printed distance/reward signs cannot be resolved against released model code.

Conditions: Attempting a faithful reimplementation of Eqs.11–12. Cause: The text describes progress toward a goal, while printed reward/softmax signs appear to favor larger distance; retrieved official repository contains documentation rather than model source.

Evidence: UNVERIFIED: Eqs.11–12 versus prose, PDF method section; official repository README at ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5.

Follow-up assessment: No inspected correction or executable official implementation resolves the sign ambiguity. Confidence: High for the documentation gap; unknown for intended implementation.
