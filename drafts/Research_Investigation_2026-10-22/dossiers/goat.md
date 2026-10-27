# GOAT

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Vision-and-Language Navigation via Causal Learning**. Liuyi Wang, Zongtao He, Ronghao Dang, Mengjiao Shen, Chengju Liu, Qijun Chen. CVPR 2024.

Primary paper: [Vision-and-Language Navigation via Causal Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Vision-and-Language_Navigation_via_Causal_Learning_CVPR_2024_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: GOAT.pdf (Main PDF; 12 pages; SHA256 a3b3f2e96335236ffedb46422b79c857517dc4e837154ef9e702881163069d07).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

GOAT targets dataset-dependent visual/language/history shortcuts in DUET-family navigation. It assumes room/keyword confounders and latent style/trajectory confounders in a structural causal model; these are modeling assumptions, not variables randomized by its benchmark.

## 3. Mechanism, training, and inference

§4 pp 3–6: extract RoBERTa direction/landmark dictionaries and CLIP room dictionaries labeled by BLIP VQA. Back-door-inspired features add fx(x)+Σz wzfz(z), using empirical frequencies or query-dependent attention weights. Front-door-inspired modules attend both current-input features and K-means-selected global training features, then sum the resulting expectations. A sigmoid gate mixes adjusted and original features. Language, local panorama and global/history features enter DUET-style dual-scale fusion and action CE. Contrastive cross-modal feature pooling uses normalized bidirectional batch InfoNCE between pooled language and local/global/fused embeddings, temperature t; it pretrains the representations used to build dictionaries. Reported 6 text/2 panorama/3 cross layers, CLIP-B/16, METER initialization, batch 12 (5 for RxR/SOON), learning rate 2e−5, max 100k; dynamic speaker augmentation is also used. Inference reads learned dictionaries and predicts graph actions; it does not perform physical randomized interventions.

Section 5.1, PDF p 6, specifies pretraining for up to 300k iterations with AdamW, batch 48 and learning rate 5e−5 on one V100, using MLM, SAP and contrastive pooling, plus object grounding for REVERIE/SOON. EnvEdit visual augmentation and external synthetic datasets are also used. The 74 keyword classes, 50 room classes and InfoNCE temperature 1 are part of the experimental configuration.

## 4. Novelty and evidence

The engineering contribution is dictionary-conditioned multimodal feature adjustment, contrastive pooling, and gated integration on a dual-scale graph backbone. The author’s claim of unbiased causal effects is stronger than evidence: front-door identification requires no unblocked X→M confounding, all causal X→Y paths mediated by M, and back-door blocking between M, Y by X. These assumptions are not established for learned attention/dictionaries; query-dependent weighting is not automatically the marginal P(z), and residual original-feature paths persist. Benchmark gains support useful regularization, not identification.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: §5 compares R2R, RxR-English, REVERIE, SOON and reports intervention/pooling ablations. The source README also warns speaker augmentation coupled with intervention is critical. VERDICT: compelling multimodal regularizer baseline for future causal claims, but one should test controlled counterfactual invariance and preserve task-relevant direction/landmark effects rather than treating all frequent keywords as removable nuisance.

Tables 1–2, PDF p 7, report R2R test SR 74.57/SPL 64.94 and REVERIE test RGS 38.32/RGSPL 26.70. Table 5 is a more controlled comparison: no adjustment gives unseen SR 75.27/SPL 64.69; both adjustment modules give 77.82/68.13. Table 7/p 8 shows text-attention plus vision-frequency weighting performing best at 68.13 SPL, versus 65.83 with attention for both. These experiments support the utility of the particular feature adjustments; neither improved benchmark scores nor object-keyword box plots identify a causal effect under the stated structural model.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/CrystalSixone/VLN-GOAT](https://github.com/CrystalSixone/VLN-GOAT); inspected snapshot **c083bf458e6d2041fbdf91c2a8036f458faa95b4**. Snapshot metadata and source are under `repositories/01/goat`.

Inspected actual back/front dictionary operations, adaptive gates, confounder extraction and room-label preprocessing. The code confirms residual mixing and attention-based conditioning; the word do in function names does not establish causal identification.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [map_nav_src/models/vilmodel_GOAT.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/models/vilmodel_GOAT.py) — lines 94–162: Frequency-weighted or attention-based dictionary features are added/gated with original text embeddings, including direct residual paths.
- [map_nav_src/do_utils/do_intervention.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/do_utils/do_intervention.py) — dictionary construction/extraction routines: Confounder representations are estimated from training samples, not externally randomized assignments.
- [map_nav_src/do_utils/extract_room_type.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/do_utils/extract_room_type.py) — BLIP room-type extraction: Room labels are model predictions and can carry perception errors.
- [map_nav_src/r2r/agent.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/r2r/agent.py) — 306–350;616–621;697: Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/pretrain_src/data/dataset.py) — 68–132;616–721: Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [map_nav_src/r2r/eval_utils.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/r2r/eval_utils.py) — 6–25: Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [map_nav_src/scripts/run_r2r_goat.sh](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/scripts/run_r2r_goat.sh) — 15–72: Activates front/back-door branches, updated text dictionary, generated speaker labels and augmented data; these additions must be matched in causal-module comparisons.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **goat-B1 — Uncertain because evidence is insufficient**. Causal effect identification is unverified; representation regularization can improve SR without removing the hypothesized confounding. Evidence: §3 SCM/§4 approximations; residual source path. Type: causal identification. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.
- **goat-B2 — Partially addressed**. Dictionary construction and multimodal attention add compute and depend on augmentation/backbone quality. Evidence: §4–5; official README dependency note. Type: compute/data. Follow-ups: Walk and Read Less token pruning. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://aclanthology.org/2025.emnlp-main.1202/
