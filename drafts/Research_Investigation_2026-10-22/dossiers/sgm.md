# SGM

## Bibliography and reviewed material

**Imagine Before Go: Self-Supervised Generative Map for Object Goal Navigation**. Sixian Zhang; Xinyao Yu; Xinhang Song; Xiaohan Wang; Shuqiang Jiang. CVPR 2024 (2024). Research cutoff: **2026-10-22**.

Primary publication/source: Supplied PDF; see local file and SHA256 in corpus metadata. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-7. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- SGM.pdf; Main PDF; SHA256 `734d3d4e868a3910c34a18d37529c7b6d44609e56406703aedcaf6c044f2c30c`; 12 pages.

## Problem, task and assumptions

ObjectNav from RGB-D, pose and an accumulated semantic map. Predict unobserved object/occupancy regions to select a long-term goal.

## Method, training and inference

Section 3: training global semantic maps are cropped/resized, patchified and randomly masked. ViT encodes visible patches f_v∈R^(n_v×d). Observed categories prompt an LLM for likely neighbors; frozen RoBERTa gives f_l∈R^(n_l×d). LXMERT-style cross-attention (Eq. 1) fuses geometry and language priors, and MAE decoder reconstructs masked multi-channel binary map patches. Loss is pixel BCE plus weighted object IoU (Eqs.2–3). At inference crop local map with enlarged bound (about 140%), resize 224, sample informative/frontier-adjacent patches using weighted normalized occupancy and Laplacian responses (Eq. 4). Predicted unknown cells are merged while observed cells override generated values. Target-channel argmax sets goal; low-confidence initial stages use exploration. Fast Marching replans local actions on occupancy. LLM responses are precomputed over category combinations; no online LLM call is necessary.

## Novelty and closest predecessors

Self-supervised masked-map completion combines episode geometry and generic language relations, versus L2 M supervised local anticipation/ensembles. Self-supervised means no goal-action labels, not no map construction or pretrained segmentation requirements.

## Results and independent verdict

Table 2 p. 7 Gibson full SR 78.0/SPL 44.0, episodic-only 71.5/40.8, plus general knowledge 74.4/41.3; MP3D full 37.6/14.7 versus 31.9/12.2. Oracle target location Gibson 91.7/72.3 reveals substantial remaining localization/planning gap. Combined exploration and priors matter, not only decoder capacity.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/sx-zhang/SGM. Snapshot commit: `579982dae2fe040d2fa26e82778352d950e31a19`. Retrieval: 2026-10-22.

INSPECTED CODE: `SGM_train/main_pretrain.py:38,170–198` selects `models_mae_cross` and AdamW; `engine_pretrain.py:75–99` applies its masked-map loss and optimizer update. A material discrepancy appears in `models_mae_cross.py:254–295`: the objective includes channel-weighted squared error and BCE, and its IoU term is computed after hard thresholding and `.detach()`. Consequently the IoU addend provides no gradient in this released path, despite the paper describing IoU as part of the optimization objective. `SGM_train/sgm/dataset.py:819–859` reads precomputed semantic maps and samples masks. `semexp/eval_sgm.py:178–184,604–606,944–947` reconstructs resized map crops and passes them to target selection; lines 1073–1080 aggregate ObjectNav metrics. Ground-truth map helper functions also exist for visualization/oracle paths; their presence alone does not establish test leakage. Released assets and exact flags need checking in an executable reproduction.

Inspected files: `SGM_train/engine_pretrain.py`, `SGM_train/main_pretrain.py`, `SGM_train/models_mae_cross.py`, `SGM_train/sgm/dataset.py`, `semexp/eval_sgm.py`, `semexp/sgm_adds.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Argmax predicted target can pursue hallucinated context; scores are confidence-like outputs without calibration evidence. Closed category combinations and semantic segmentation constrain open vocabulary. Resizing crops can distort metric precision; observed-value override helps provenance but does not validate unknown predictions.

Relevant follow-up relationship: WMNav replaces learned map completion with VLM scores; WoMAP optimizes detector-derived latent rewards. Neither establishes that semantic prior maps recover correctly after confidently wrong predictions.

## Access and uncertainty record

Full text is preserved under `corpus/sgm/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/sgm/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-sgm-target_bias

**Uncertain because evidence is insufficient — Methodological.** Predicted semantic targets can attract exploration toward confidently wrong room/object context.

Conditions: Unusual layouts, segmentation errors or category combinations outside training. Cause: Map completion and argmax target selection use scores without demonstrated probability calibration; observed-map override protects observed cells but not unknown predictions.

Evidence: PAPER EXPERIMENT: Table 2 p.7 full Gibson SR/SPL 78.0/44.0 versus oracle target 91.7/72.3. DEDUCTION: oracle gap does not uniquely identify calibration as the cause.

Follow-up assessment: WMNav uses VLM curiosity and L2M ensembles uncertainty; neither establishes reliable recovery after confidently wrong semantic-map completion. Confidence: Medium.

### 02-sgm-iou_gradient

**Still unresolved in the reviewed literature — Implementation-specific.** The inspected hard IoU term cannot train the decoder through its detached thresholded mask.

Conditions: The released main masked-map pretraining loss. Cause: Thresholding and detach remove the differentiable path for IoU, although weighted MSE and BCE still supply gradients.

Evidence: INSPECTED CODE: main pretrainer imports models_mae_cross (lines 38,172); models_mae_cross.py:254–295 computes detached hard IoU. No training run was executed.

Follow-up assessment: No inspected patch or follow-up establishes the empirical effect of replacing the detached term with a differentiable surrogate. Confidence: High for gradient path; unknown for benchmark impact.
