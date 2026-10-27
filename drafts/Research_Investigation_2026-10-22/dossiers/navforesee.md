# NavForesee

## Bibliography and reviewed material

**NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction**. Fei Liu; Shichao Xie; Minghua Luo; Zedong Chu; Junjun Hu; Xiaolong Wu; Mu Xu. CVPR 2026 (2026). Research cutoff: **2026-10-22**.

Primary publication/source: Supplied PDF; see local file and SHA256 in corpus metadata. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- NavForesee.pdf; Main PDF; SHA256 `b5d570473be6eb9a04efd5336ce1b4b85f985fd8fdd65c6e8aea2f1e863eda41`; 10 pages.

## Problem, task and assumptions

RGB VLN-CE using observation history and relative pose, with hierarchical instruction planning and continuous waypoint chunks.

## Method, training and inference

Section 3: Qwen2.5-VL 3 B predicts planning text and two sets of future queries: a fixed short horizon and a longer milestone horizon. Gemini 2.5 Pro annotates 10,000 R2R and 20,000 RxR episodes; 16.7% of poor terminal milestones are filtered, leaving about 0.2 million and 1.3 million steps. Structured attention lets long-horizon queries see short-horizon queries, separates semantic/depth branches and lets action queries see all of them. Heads predict DINO/SAM semantic features with MSE and depth with scale-invariant logarithmic loss. An inverse-dynamics head predicts K×5 waypoint vectors (x, y, sin (theta), cos (theta), stop) with MSE; planning text uses cross-entropy. These are goal-conditioned future targets and inverse action prediction, not clearly candidate-action-conditioned counterfactual simulation. New real observations refresh each control cycle, but no calibrated observed-residual objective is specified.

## Novelty and closest predecessors

Combines hierarchical language planning with two-horizon semantic/geometric prediction. Planning-only and prediction-only controls are necessary to distinguish its contribution from HNR, NavMorph and MonoDream.

## Results and independent verdict

Table 1, p. 7: R2R SR 66.2/SPL 59.7 versus CorrectNav 65.1/62.3; RxR 66.3/53.2 versus 69.3/63.3. Table 2 gives 66.2/59.7 for the full system, 48.8/39.4 without VLM planning and 58.6/50.1 without long-horizon prediction. Table 3 gives short-horizon SR 66.2 at five steps, 64.1 at four and 56.5 at three. Ablations support the internal recipe, while the headline comparisons do not show uniform superiority.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: No verified official implementation located in the recorded search.. Snapshot commit: `not available`. Retrieval: 2026-10-22.

The supplied CVPR paper and current preprint identity were checked; no official model/training repository was verified. Future-query attention and inverse-dynamics loss remain paper evidence.

Inspected files: No implementation files inspected; access scope is stated above. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Predicted futures may act as goal templates rather than faithful consequences of alternative actions. Teacher annotation and filtering add cost and data selection. Qualitative geometric fidelity does not validate confidence. No official implementation was verified.

Relevant follow-up relationship: WorldVLN and LCVN unify action/observation prediction differently. SC²-WM adds delayed observed-residual adaptation as well as immediate internal refinement; those mechanisms must be distinguished from simple anticipation.

## Access and uncertainty record

Full text is preserved under `corpus/navforesee/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/navforesee/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-navforesee-counterfactuals

**Uncertain because evidence is insufficient — Insufficiently tested.** Predicted target structures are not established as faithful consequences of alternative candidate actions.

Conditions: Using semantic/depth futures for branch choice under ambiguous or unusual layouts. Cause: Task-conditioned future supervision may encode desired goal templates; it does not alone identify action-specific environment transitions.

Evidence: DEDUCTION: future prediction/action pipeline; PAPER EXPERIMENT: Table 2 p.7 full SR 66.2 versus 58.6 without long-horizon prediction shows utility, not physical counterfactual fidelity.

Follow-up assessment: WorldVLN and LCVN use other action/observation interfaces; SC²-WM adds observed residual adaptation. None is a matched counterfactual-validity test of NavForesee. Confidence: Medium.
