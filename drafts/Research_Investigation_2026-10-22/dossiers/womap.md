# WoMAP

## Bibliography and reviewed material

**WoMAP: World Models for Embodied Open-Vocabulary Object Localization**. Tenny Yin; Zhiting Mei; Tao Sun; Lihan Zha; Miyu Yamane; Emily Zhou; Jeremy Bao; Ola Shorinwa; Anirudha Majumdar. CoRL 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: Supplied PDF; see local file and SHA256 in corpus metadata. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- WoMAP.pdf; Main PDF; SHA256 `7cfa9321714ceca03c20a5dc1e4c1cc7278dcaafcb88874ccb0b6db3c98f3041`; 26 pages.

## Problem, task and assumptions

Active 6 DoF camera localization of a language-specified object, maximizing visibility/detection reward. Primarily tabletop/robot active perception, not route instruction navigation.

## Method, training and inference

Section 3: construct Gaussian Splat scenes from real video and sample arbitrary observation/pose pairs, deriving relative actions; no expert demonstrations required. Grounding DINO confidence×bounding-box size supplies dense target rewards. Frozen DINOv2 patch embeddings form z∈R^d; ViT dynamics predicts Gaussian next-latent parameters and a language-conditioned head predicts reward. Recurrent dynamics supervision is described as KL to true transition (Eq. 1), but empirical target distribution/variance must be checked in code. VLM chooses coarse textual action proposals; each is refined by differentiable MPC through latent dynamics, selecting highest predicted reward. Eq. 2 prints maximizing reward PLUS positive action-change L1 while calling it smoothness: the sign is inconsistent and requires source resolution. No pixel decoder is required.

## Novelty and closest predecessors

Combines real-to-sim data generation, detector-reward distillation and VLM-initialized latent action optimization. Closest DINO-WM predicts features; here explicit task rewards and proposal priors guide active perception.

## Results and independent verdict

Figures 6–7 report 150 tasks/environment; Table 1 p. 8 hardware 20 trials per scene, WoMAP real 55/65/63% versus VLM 0/5/0. Table 2 nominal 63% falls to 50% under lighting and 30% under background shift, substantial residual sensitivity. Success is detector confidence/box-size threshold, potentially sharing reward bias; not object-goal navigation SR .

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/irom-princeton/womap. Snapshot commit: `3f75333d788ac3fe0cd05bed3781104b7a0b859a`. Retrieval: 2026-10-22.

INSPECTED CODE: `src/models/worldmodel.py:260–300,677–700` uses sampled Gaussian next latents with smooth-L1 reconstruction and KL regularization, alongside confidence/bounding-box losses. This is more specific than a generic KL between learned and true transition distributions. `joint_gradient_planner` at 1180–1250 samples next latent and reward distributions, optionally divides reward by predicted standard deviation, and minimizes abs(1 - weighted_reward) by differentiating through the model. No action-change L1 smoothing term is present in this inspected planner, so the paper's printed positive sign cannot be explained as a matching implementation. Other planner branches and all deployment dispatches were not exhaustively traced; this is a precise source discrepancy, not proof that every reported run used this branch. Detector-reward calibration, stochastic uncertainty coverage, preprocessing of every Gaussian scene, and actual hardware evaluation remain unexecuted. The model/loss/planner audit is substantive, but complete end-to-end reproduction is not certified.

Inspected files: `src/models/worldmodel.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Reward detector errors can be amplified by gradient optimization; predicted Gaussian spread is not shown calibrated. Object visibility differs from collision-free reachability. VLM proposals restrict explored modes; background shift halves success. The printed smoothing sign and exact dynamics likelihood need code audit.

Relevant follow-up relationship: DINO-WM,NWM,CVAML provide representation/planning/objective contrasts. A navigation proposal about model-basedVLM proposal refinement overlapsWoMAP strongly.

## Access and uncertainty record

Full text is preserved under `corpus/womap/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/womap/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-womap-reward_exploitation

**Still unresolved in the reviewed literature — Methodological.** Optimizing detector-derived reward can favor visible/confident objects without ensuring physically valid goal reaching.

Conditions: Background/lighting shift or candidate actions outside the learned dynamics support. Cause: Gradient search amplifies errors in learned dynamics and rewards; evaluation success also uses detector confidence/box-size thresholds, potentially sharing reward bias.

Evidence: PAPER EXPERIMENT: Table 2 p.8 nominal success 63% falls to 50% lighting and 30% background shift. INSPECTED CODE: optional reward/std score exists; joint planner differs from printed smoothing objective.

Follow-up assessment: DINO-WM/NWM offer other models and CVAML addresses value-objective bias. None is a matched resolution of detector-based goal/reward validity. Confidence: Medium.
