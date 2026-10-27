# NWM

## Bibliography and reviewed material

**Navigation World Models**. Amir Bar; Gaoyue Zhou; Danny Tran; Trevor Darrell; Yann LeCun. CVPR 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-7. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- NWM.pdf; Main PDF; SHA256 `2cfe50f6350a9bd68f8bfa8fbc61d38bf2c186106ecb28fb6e37ab716bf34d26`; 11 pages.

## Problem, task and assumptions

Goal-image navigation and action-conditioned video prediction over heterogeneous robot datasets; no route-language input.

## Method, training and inference

Section 3: a Stable Diffusion VAE encodes RGB images. CDiT processes target patches with self-attention and historical context with cross-attention; its interaction cost scales approximately as O(m n² d) for m context frames, n patches and width d, instead of self-attention over all concatenated frames. Four historical frames, displacement/yaw action and signed time offset up to 16 seconds condition adaptive layer normalization. Training uses clean-latent denoising MSE plus a variance variational-bound term. Actions are normalized by robot step size; actionless Ego4D data can supply temporal supervision. At inference CEM or reranking uses generated endpoint similarity to a goal image. The largest 1B-parameter training uses 64 H100 GPUs and batch 4096. The inspected released CEM planner restricts candidate trajectories more than an unconstrained sequence optimizer; see code audit.

## Novelty and closest predecessors

Cross-context diffusion with action/time conditioning supports flexible generative horizons and goal-image planning. It generates pixels, unlike DINO-WM's feature-only planner, and uses substantially larger visual priors/data than PlaNet without learning a task reward.

## Results and independent verdict

Table 1, p. 6: LPIPS is 0.760 with time only, 0.318 with action only and 0.295 with both. Table 2, p. 7: RECON two-second ATE/RPE is 1.13/0.35 versus NoMaD 1.93/0.52; reranking 32 proposals gives 1.78/0.48. These are offline trajectory errors, not measured physical success or collision guarantees.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/facebookresearch/nwm. Snapshot commit: `3f6cd8e70d6f2d1e2b9684acff510710135f0f41`. Retrieval: 2026-10-22.

INSPECTED CODE: `models.py:233–241` adds diffusion-time, relative-time and action embeddings and attends to encoded visual context. `train.py:127–138,188–258,290–303` instantiates CDiT, combines robot datasets and optimizes the diffusion loss with AdamW. Its periodic helper at 366–405 evaluates only the first sampled batch; this is training-time monitoring, not necessarily the paper's full evaluation. `planning_eval.py:217–299` implements LPIPS-driven CEM, but its candidate displacement is repeated across the predicted horizon, with a final-yaw adjustment. Therefore this released planner is more restricted than unconstrained optimization of every action in an arbitrary eight-step trajectory. Lines 366–376 evaluate offline ATE/RPE against logged trajectories. README records unavailable redistributable high-resolution data. No online robot navigation, training or checkpoint evaluation was executed.

Inspected files: `models.py`, `planning_eval.py`, `train.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Diffusion×CEM is costly; endpoint metrics and unsupported action rollouts can be exploited. Logged action/time dependence does not identify all interventions. OfficialREADME notes nonredistributable high-resolution SACSoN/HuRoN data, limiting exact reproduction.

Relevant follow-up relationship: Vid2World adapts pretrained video generators to causal action conditioning; LCVN adds language, actor–critic learning and unified prediction. Their datasets and evaluation protocols differ.

## Access and uncertainty record

Full text is preserved under `corpus/nwm/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/nwm/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-nwm-planning_cost

**Shifted into another cost or limitation — Implementation-specific.** Diffusion-based trajectory evaluation and restricted CEM candidates impose compute and expressivity costs.

Conditions: Online action planning over multiple imagined time steps. Cause: Each candidate requires generated predictions; the released planner repeats one displacement across the horizon with a final yaw adjustment instead of optimizing fully independent per-step actions.

Evidence: INSPECTED CODE: planner.py:217–299 at the dossier commit; PAPER EXPERIMENT: Table 2 p.7 is offline ATE/RPE, not closed-loop SR.

Follow-up assessment: LCVN trains an actor and unifies language/video/action generation, moving cost into training and changing the evaluation. No matched cost/expressivity resolution was reviewed. Confidence: Medium.

### 02-nwm-physical_validity

**Uncertain because evidence is insufficient — Insufficiently tested.** Offline video/trajectory errors do not establish safe causal action predictions in deployment.

Conditions: Actions outside logged support, moving obstacles or long physical navigation. Cause: Observational action-conditioned training plus endpoint error metrics does not identify unsupported interventions or validate all intermediate transitions.

Evidence: PAPER EXPERIMENT: Tables 1–2 pp.6–7 report LPIPS and ATE/RPE; no physical safety claim follows. Access limit: high-resolution SACSoN/HuRoN data are not redistributable in official README.

Follow-up assessment: Vid2World improves causal masking/guidance; LCVN adds language. Different datasets and offline metrics do not resolve this validity gap. Confidence: Medium.
