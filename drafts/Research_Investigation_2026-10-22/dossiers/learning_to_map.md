# Learning to Map

## Bibliography and reviewed material

**Learning to Map for Active Semantic Goal Navigation**. Georgios Georgakis; Bernadette Bucher; Karl Schmeckpeper; Siddharth Singh; Kostas Daniilidis. ICLR 2022 (2022). Research cutoff: **2026-10-22**.

Primary publication/source: Supplied PDF; see local file and SHA256 in corpus metadata. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-7. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Learning to Map.pdf; Main PDF; SHA256 `b4e878cad3c07a526e4b1b407a73d7eb0acd69de5b1e2e38a2fb3bf0da6cd1b0`; 19 pages.

## Problem, task and assumptions

ObjectNav with noiseless RGB-D and pose in unseen MP3D scenes. Predict semantic occupancy beyond the field of view and choose exploration goals using ensemble disagreement.

## Method, training and inference

Section 3 pp. 3–7: project depth and semantic segmentation into 10 cm top-down cells. First UNet predicts occupancy over unknown/occupied/free channels; second takes predicted occupancy concatenated with projected semantics and outputs |C_s|×h×w class probabilities. Pixelwise CE trains both; semantic gradients flow through occupancy predictor. Four independently initialized predictors form an ensemble; mean is estimated semantic probability and variance is an epistemic proxy. After about 900 k initial examples, actively collect about 500 k examples maximizing average class variance (Eq. 1) and finetune. At deployment accumulate local maps by Bayes-style registration and select locations maximizing target mean + alpha×standard deviation (Eq. 2); compare lower-bound/mixed/mean variants. Frozen DD-PPO depth PointNav executes the chosen goal. Models do not update online during task evaluation.

## Novelty and closest predecessors

Separates active collection for map learning from uncertainty-guided target search; extends anticipated occupancy to semantic completion with an ensemble. UCB-style terminology is heuristic: the paper acknowledges unknown coverage probability, not a calibrated confidence theorem.

## Results and independent verdict

Table 1 p. 7 active UCB SR 34.3±.9/SPL 13.3±.5 versus offline 30.1±.9/9.6±.4, BALD 30.4/9.6. Table 2 semantic IoU 25.6 versus offline 20.1. About 2480 episodes/11 target classes,56 train/11 val scenes; 500 steps, one-meter visible-object goal. Gains support collection strategy and map quality but not general calibration.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/ggeorgak11/L2M. Snapshot commit: `afa7081241b4e2407286ebe7dc41d2a32f10e906`. Retrieval: 2026-10-22.

INSPECTED CODE: `models/predictors/map_predictor_model.py:45–62` trains occupancy and semantic maps with separate cross-entropy terms. `tester.py:245–282,433–442` preprocesses RGB/depth, runs the ensemble, and registers semantic means and per-class variance. `trainer_active.py:344–352` computes classwise ensemble variance and averages it for active collection; Adam initialization is at 61–65. `tester.py:348–367` reports distance, success, SPL and soft-SPL from the navigation trajectory. This is an ensemble-disagreement heuristic, without a code mechanism proving probabilistic coverage. Data, semantic labels, pose assumptions and the frozen point-navigation controller remain reproduction prerequisites. No simulator episode was executed.

Inspected files: `models/predictors/map_predictor_model.py`, `tester.py`, `trainer_active.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Ground-truth pose/semantic training maps and four-model compute are strong requirements. Repeated correlated predictions registered as independent evidence can become overconfident. Variance can be low for shared bias; UCB scores have no stated finite-sample coverage guarantee.

Relevant follow-up relationship: SGM uses masked-map self-supervision and language priors; WMNav uses heuristic VLM curiosity. A proposal merely adding uncertainty-guided exploration or ensembles would overlap this work.

## Access and uncertainty record

Full text is preserved under `corpus/learning_to_map/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/learning_to_map/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-learning_to_map-overconfidence

**Uncertain because evidence is insufficient — Insufficiently tested.** Ensemble variance and repeated map fusion do not provide validated calibrated exploration uncertainty.

Conditions: Shared model bias or repeated correlated observations in active object-goal navigation. Cause: Ensemble members can agree incorrectly, and registered repeated predictions are not independent evidence; UCB score construction lacks demonstrated coverage.

Evidence: DEDUCTION: ensemble prediction and UCB mechanism; PAPER EXPERIMENT: Tables 1–2 p.7 show SR/SPL 34.3/13.3 and IoU 25.6, not calibration guarantees.

Follow-up assessment: SGM uses masked map completion and WMNav heuristic VLM curiosity; neither resolves posterior coverage or shared-bias detection in this protocol. Confidence: Medium.

### 02-learning_to_map-resources

**Partially addressed — Methodological.** Active semantic mapping requires privileged training maps, accurate pose and multiple predictor evaluations.

Conditions: Deployment without reliable pose or training scenes with annotated semantic occupancy. Cause: Map registration relies on known pose; supervised targets and four-model ensembles introduce annotation and inference costs.

Evidence: AUTHOR METHOD: map construction/training and ensemble details in dossier; no pose-error ablation establishes robustness.

Follow-up assessment: SGM introduces masked-map pretraining and priors, partially changing supervision, but retains semantic-map/perception dependencies. Confidence: Medium.
