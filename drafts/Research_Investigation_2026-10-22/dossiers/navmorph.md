# NavMorph

## Bibliography and reviewed material

**NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments**. Xuan Yao; Junyu Gao; Changsheng Xu. ICCV 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/ICCV2025/html/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- NavMorph.pdf; Main PDF; SHA256 `b203428708b07f0a7291efae7b9a06c279b008e55f1f4ea7abf0547fd1bc977d`; 11 pages.

## Problem, task and assumptions

Continuous instruction navigation with RGB-D observations, predicted waypoints and accumulated experience. Addresses latent foresight and online scene adaptation without repeatedly synthesizing RGB panoramas.

## Method, training and inference

Sections 3.1–3.4 pp. 3–6: a recurrent state-space model combines deterministic history h_t with Gaussian stochastic state s_t. Observation-conditioned posterior and action-conditioned prior share the latent dynamics. Contextual Evolution Memory stores N_m scene vectors; top-K cosine retrieval enriches h_t with a weighted memory sum (Eq. 2), and retrieved slots receive exponential forward updates v_k←(1−β) v_k+βh_t (Eq. 3). The prior then rolls out T_p future states/actions without future observations. A decoder reconstructs visual embeddings rather than RGB. Eq. 5 combines observed/future feature likelihoods, action likelihoods and Gaussian posterior/prior KL. Eq. 6 adds NDTW-style temporal alignment, with L2 supervision for continuous displacement, plus DAgger-like teacher-node cross-entropy. At test time memory is updated forward; this 'self-evolution' is not gradient-based TTA. The policy still depends on pretrained navigational architectures ETPNav/HNR or monocular VLN-3DFF.

## Novelty and closest predecessors

The navigation-specific contribution is context-memory augmentation of RSSM and online memory evolution, relative to PlaNet/Dreamer recurrent dynamics and HNR feature anticipation. Future-feature losses and memory changes must be separated experimentally; they are not a new general principle of world modeling.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p. 7 monocular reproduced VLN-3DFF SR 43.8/SPL 29.4 versus NavMorph 47.9/33.2 on R2R-CE val-unseen; test 43.8/28.6 versus 45.7/30.2. HNR panoramic 61/51 to 64/53. Table 2 RxR-CE HNR56.48/46.62 to 58.02/48.98. Section 4.3 describes loss and fixed-memory ablations; numerical Table 3 is not transcribed here. Results establish additive gains on these architectures, not uniformly superior cross-protocol navigation. Reported single RTX3090 training; compute not reproduced. Table 3, p. 8: removing reconstruction/action/KL yields SR 45.51/44.81/44.10 versus 47.91 full; without self-evolution it is 45.08. Table 4 gives 21.22 seconds per instruction for contextual memory versus 44.56 for the LSTM adaptation variant. Memory size 1000 gives the best SR in Figure 4, while both smaller and larger settings can hurt. The authors explicitly acknowledge cross-episode test-scene memory and evaluation-fairness concerns.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/Feliciaxyao/NavMorph. Snapshot commit: `7fb0f4aa44ff3e042335d8de0d1b2197c6f1f657`. Retrieval: 2026-10-22.

INSPECTED CODE: `vlnce_baselines/models/etp/vilmodel_cmt.py:741–759,942–966` concatenates 768-dimensional visual and position embeddings, learns 1536-dimensional Gaussian prior/posterior projections, samples with bounded standard deviation, and decodes a three-dimensional displacement. `utils_p/losses.py:56–98` implements Gaussian KL with separate stop-gradient terms. `vlnce_baselines/ss_trainer_ETP.py:1352–1392` constructs the real-state posterior and invokes probability/action losses; notably the probability-loss call precedes assignment of the current posterior, so time indexing requires verification before equating it with the printed current posterior/prior objective. Lines 1389–1394 push and retrieve memory during evaluation. Lines 1421–1454 use nearest reference-path coordinates during training and retrieve memory before imagination. `utils_p/memory.py:125–175,220–246` confirms finite-capacity nearest-neighbor updates and cosine retrieval. This is forward memory adaptation, not evidence that model weights are optimized at deployment. The release includes its training/evaluation trainer and configs, but no benchmark was run and the detailed training-time alignment warrants an execution audit.

Inspected files: `utils_p/memory.py`, `vlnce_baselines/models/etp/vilmodel_cmt.py`, `vlnce_baselines/ss_trainer_ETP.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

CEM may retain scene/context biases across episodes and depends on slot replacement, reset policy and episode order. No confidence gate distinguishes helpful versus harmful stored features in Eq. 2–3. Predicted-state alignment is not a calibrated uncertainty estimate. SC²-WM subsequently adds latent refinement and gradient adaptation, partly addressing online correction but retaining uncertainty about endogenous feedback validity. Methodological limitation; confidence medium.

Relevant follow-up relationship: SC²-WM Sections 3 and Table 3 directly extend the world-model adaptation theme; HNR remains a necessary controlled baseline.

## Access and uncertainty record

Full text is preserved under `corpus/navmorph/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/navmorph/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-navmorph-correction

**Partially addressed — Methodological.** Online context adaptation improves scores but does not establish reliable correction of confidently wrong predictions.

Conditions: Unseen CE scenes and biased stored context. Cause: Contextual memory supplies retrieved features without a calibrated confidence gate; latent alignment can preserve shared prediction bias.

Evidence: PAPER EXPERIMENT: Table 3 p.8 SR 47.91 versus 45.08 without self-evolution; DEDUCTION: Eq.2–3 memory weights are not correctness probabilities.

Follow-up assessment: SC²-WM adds learned feature/state refinement and delayed real-observation TTA, with further CE gains. Residual-to-correct-action reliability is still not isolated. Confidence: Medium.

### 02-navmorph-episode_protocol

**Still unresolved in the reviewed literature — Methodological.** Cross-episode memory makes evaluation depend on reset, ordering and prior test-scene exposure.

Conditions: Transductive test evaluation across multiple episodes in a scene. Cause: Memory slots retain scene/context information, and replacement behavior changes available knowledge.

Evidence: AUTHOR ACKNOWLEDGEMENT: PDF p.8 discussion of test-scene memory/fairness; Figure 4 memory size 1000 performs best and smaller/larger settings can hurt.

Follow-up assessment: SC²-WM also adapts online, making episode-reset and matched exposure controls necessary. No reviewed ordering/reset factorial resolves comparability. Confidence: Medium.
