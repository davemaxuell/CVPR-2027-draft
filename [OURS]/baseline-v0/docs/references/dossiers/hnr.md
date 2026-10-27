# HNR

## Bibliography and reviewed material

**Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation**. Zihan Wang; Xiangyang Li; Jiahao Yang; Yeqi Liu; Junjie Hu; Ming Jiang; Shuqiang Jiang. CVPR 2024 (2024). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Lookahead_Exploration_with_Neural_Radiance_Representation_for_Continuous_Vision-Language_Navigation_CVPR_2024_paper.pdf. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- HNR.pdf; Main PDF; SHA256 `8b5ad4072b2acf6b5e534b2c6b03945d7f2b4155da8f18c4bf42be36f107de07`; 10 pages.

## Problem, task and assumptions

A waypoint's current single-view feature misses occluded future context; RGB generation can be slow and distorted. Assumes panoramic 12-view RGB-D, camera pose/intrinsics, predicted candidate waypoints and low-level control.

## Method, training and inference

Section 3 pp. 3–6: frozen CLIP ViT-B/32 grid features are backprojected with depth/pose into an incremental 3D feature cloud (Eq. 1–3). For a candidate camera ray, k-nearest cloud features plus relative position/orientation/scale feed an MLP predicting feature vector and volume density; volumetric compositing gives region embeddings (Eq. 4–8). Empty neighborhoods skip the MLP. A four-layer view Transformer inpaints contextual information in feature space; region/view cosine losses match CLIP targets, with RGB-D rendering auxiliary loss (Eq. 9–12). HNR pretraining uses 800 HM3D training scenes and random 15-step episodes. Candidate 12-view features and predicted depth generate next-level waypoint/lookahead nodes. A panorama encoder and graph-aware cross-modal Transformer score visited, candidate, lookahead and stop nodes. Branch scores are max (candidate, descendant scores), Eq. 15. Soft-target cross-entropy supervises the branch leading closest to the goal; Dijkstra plus a controller executes the chosen reachable waypoint. This is explicit inference-time feature lookahead, not merely a training loss.

## Novelty and closest predecessors

Compared with DREAMWALKER's RGB scene synthesis/MCTS and RNR-Map's 2 D latent map, HNR uses a 3D feature cloud, multiscale semantic rendering, and parallel branch scoring. It inherits graph-aware navigation machinery from ETPNav/DUET lineage. Extra HM3D pretraining must be accounted for when attributing all gains to geometry.

## Results and independent verdict

PAPER EXPERIMENT: Table 3 p. 7 holds lookahead policy constant: single-view SR 57.21/SPL 49.15; NeRF RGB56.55/48.61; image generation 58.35/50.96; HNR60.74/51.27; oracle future features 63.13/54.59. This is stronger mechanism evidence than leaderboard comparison. Removing positional embeddings gives 56.17/48.91; removing region loss 60.20/50.74. Fig. 5 p. 7 reports feature similarity decreasing with prediction distance. Table 1 R2R-CE unseen SR 61/SPL 51 versus ETPNav 57/49; Table 2 RxR-CE56.39/46.73 versus 54.79/44.89. Table 5, p. 8 reports rendering at 11.5 ms on one RTX 3090, versus 169.5 ms without sparse sampling and 42.3 ms without KD-tree search. This is component timing. Table 6 gives SR 60.74 for soft supervision versus 56.72 for the hard-target variant; the authors attribute the latter to occlusion and prediction noise.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/MrZihan/HNR-VLN. Snapshot commit: `6c935c1f6f0d715b1c6bef28724bdfabb48b159e`. Retrieval: 2026-10-22.

INSPECTED CODE: `NeRF/models/etp/nerf.py:148–195` computes alpha/transmittance weights and normalized rendered features; sparse neighbor search uses geometric distances and peak selection. `NeRF/ss_trainer_ETP.py:1065–1106` crops and encodes CLIP regions, preprocesses depth, and invokes rendering supervision. Lines 1162–1177 explicitly use 2×2 rendering for fast feature evaluation versus 224×224 for visualization. `run_r2r/iter_train_nerf.yaml` exposes 15,000 iterations, batch one and learning rate 1e-5. The available snapshot is centered on NeRF training/rendering evaluation; the complete lookahead navigation tree/policy reproduction was not established from the inspected files. Its 11.5 ms paper timing concerns rendering, not a certified end-to-end robot control cycle. Historical CUDA neighbor-search and Habitat dependencies remain prerequisites.

Inspected files: `NeRF/models/etp/nerf.py`, `NeRF/ss_trainer_ETP.py`, `run_r2r/iter_train_nerf.yaml`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Blind-region semantic extrapolation deteriorates with distance, and max over noisy descendant scores can favor an overoptimistic branch. No uncertainty penalty or calibration is included in Eq. 15. Large geometric feature memory and 800-scene pretraining are costs. NavMorph and SC²-WM add adaptation; NavQ replaces rollouts with semantic successor features in a different discrete/object-goal protocol. No evidence here that those eliminate HNR's unseen-occlusion error under equal sensing/data.

Relevant follow-up relationship: NavMorph; SC²-WM; NavQ. DREAMWALKER's oracle-world and oracle-distance analyses motivate separating prediction from scoring errors.

## Access and uncertainty record

Full text is preserved under `corpus/hnr/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/hnr/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-hnr-extrapolation

**Partially addressed — Methodological.** Semantic feature prediction degrades with distance and occlusion.

Conditions: Longer lookahead branches entering unobserved regions in R2R-CE/RxR-CE. Cause: Rendered features must extrapolate geometry/semantics not directly contained in the observed feature field.

Evidence: PAPER EXPERIMENT: Figure 5 p.7 shows declining similarity with distance; Table 3 predicted SR 60.74 versus oracle 63.13 under fixed lookahead policy; Table 6 soft targets outperform hard targets.

Follow-up assessment: NavMorph adds contextual adaptation and improves HNR; SC²-WM adds calibration and delayed real-observation adaptation. Gains do not eliminate the equal-sensing occluded-region oracle gap. Confidence: Medium.

### 02-hnr-branch_selection

**Uncertain because evidence is insufficient — Insufficiently tested.** Maximizing descendant scores can favor an overoptimistic noisy branch.

Conditions: More candidate descendants or correlated errors in unseen-branch evaluation. Cause: Eq.15 takes a maximum without a demonstrated uncertainty correction or calibrated confidence bound.

Evidence: DEDUCTION: Eq.15, method section; no risk-calibration or search-width bias experiment was identified in reviewed tables.

Follow-up assessment: NavQ avoids explicit rollout through successor semantics; SC²-WM adapts features. Neither directly tests or removes the maximum-selection bias in the same planner. Confidence: Medium.
