# Vid2World

## Bibliography and reviewed material

**Vid2World: Crafting Video Diffusion Models to Interactive World Models**. Siqiao Huang; Jialong Wu; Qixing Zhou; Shangchen Miao; Mingsheng Long. ICLR 2026 (2026). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/1214d67bb7fa38d40e0b3c2be677e39b-Abstract-Conference.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-9. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Vid2World.pdf; Main PDF; SHA256 `a292ed40793aa2af24ee3ff6cd8b3f280a7fa37ef555e610601333eda61c745e`; 44 pages.

## Problem, task and assumptions

Convert pretrained passive video generation into temporally causal, frame-level action-conditioned simulation for manipulation, games and visual navigation.

## Method, training and inference

Sections 4.1–4.2: mask temporal attention and transform temporal convolution weights. Shift reuses weights at earlier indices; masked transfer discards future-facing weights; extrapolative transfer replaces future features by a local linear combination of past features and redistributes convolution weights. For a three-tap example z_(t+1)≈2 z_t−z_(t−1), effective weights become w_−1−w_1 and w_0+2 w_1. This preserves responses only under approximation assumptions. Diffusion forcing samples independent per-frame noise levels; action embeddings at time t use preceding action a_(t−1). Independent action dropout trains conditional/unconditional scores; guidance epsilon=(1+lambda) epsilon_cond−lambda epsilon_uncond. Theorem 4.1 interprets this as probability tilting toward action-consistent outcomes, not causal identification of unsupported interventions. DynamiCrafter 1.1 B initialization,100ksteps/about 7 days on 4 A100 forRT1. Inference autoregressively denoises latest frame with clean history.

## Novelty and closest predecessors

Systematic architecture and training adaptation of passive video priors, plus fine-grained action guidance. Distinct from training a navigation diffusion model from scratch or only adding a global action embedding.

## Results and independent verdict

Table 1, p. 8: RT1 FVD is 18.5 versus 24.2 for the action-conditioned baseline. Some FID comparisons are flagged because baseline code did not set the Inception network to evaluation mode. On RECON, Vid2World FVD is 59.4 versus NWM 31.2, and LPIPS 0.3236 versus 0.295, although SSIM improves from 0.389 to 0.481. Thus there is no uniform dominance. Table 2, p. 9: extrapolative transfer plus guidance gives FVD 22.4 versus masked transfer plus guidance 25.8. The real-to-sim experiment ranks three RT1 drawer-policy checkpoints; this is a narrow validation setting.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/thuml/Vid2World. Snapshot commit: `ae24077adbd0783bc6e1b79f705d9b8fad5888ce`. Retrieval: 2026-10-22.

INSPECTED CODE: `main/utils_train.py:342–366` implements three-tap extrapolative transfer as weights [0, w0-w2, w1+2w2], matching a linear future-feature approximation. `configs/navigation/config_recon_train.yaml` selects extrapolative transfer, diffusion-forcing loss, causal attention, three-dimensional actions and 0.2 action dropout. `lvdm/models/ddpm3d.py:1144–1158` supports independent per-time-step action masks. `lvdm/data/reconvid.py:178–210` reads video/action pairs, applies transforms and normalizes frames to [-1,1]. Sampler code implements conditional guidance; the full action-specific evaluation branch was not exhaustively traced. Causal masking prevents future-frame leakage in this architecture, but neither these files nor the paper's guidance identity establish intervention identification from observational data. Exact video/checkpoint benchmarks were not executed.

Inspected files: `configs/navigation/config_recon_train.yaml`, `lvdm/data/reconvid.py`, `lvdm/models/ddpm3d.py`, `lvdm/models/samplers/ddim_multiplecond.py`, `lvdm/modules/networks/openaimodel3d.py`, `main/utils_train.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Guidance can overconcentrate probabilities and does not establish calibrated transition uncertainty. Local linear extrapolation fails under abrupt changes. Video metrics and a few policy-ranking examples do not demonstrate reliable VLN planning or all-action counterfactual validity.

Relevant follow-up relationship: NWM,LCVN and WorldVLN are navigation interfaces to compare. Proposed 'causal' world modeling must distinguish temporal masking, conditional guidance and actual intervention identification.

## Access and uncertainty record

Full text is preserved under `corpus/vid2world/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/vid2world/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-vid2world-causality

**Uncertain because evidence is insufficient — Insufficiently tested.** Temporal causal masking and guidance do not establish intervention-valid or calibrated transition probabilities.

Conditions: Unsupported actions, abrupt dynamics changes or physical policy deployment. Cause: Chronological conditioning prevents future leakage but does not remove confounding in logged action data; guidance changes conditional distributions and linear extrapolation assumes local regularity.

Evidence: DEDUCTION: transfer/masking/guidance mechanism. PAPER EXPERIMENT: Table 1 p.8 RECON FVD/LPIPS 59.4/.3236 versus NWM 31.2/.295, and the real-to-sim study ranks only three drawer-policy checkpoints.

Follow-up assessment: NWM/LCVN/WorldVLN use video prediction differently; no reviewed cross-task intervention study establishes broad causal validity. Confidence: Medium.
