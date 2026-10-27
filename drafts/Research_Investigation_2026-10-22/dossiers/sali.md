# SALI

## Bibliography and reviewed material

**Planning from Imagination: Episodic Simulation and Episodic Memory for Vision-and-Language Navigation**. Yiyuan Pan; Yunzhe Xu; Zhe Liu; Hesheng Wang. AAAI 2025; supplied PDF is arXiv2412.01857v2,25Dec2024 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://ojs.aaai.org/index.php/AAAI/article/view/32679. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-7. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- SALI.pdf; Main PDF; SHA256 `eea88ef1fe0d39c7cc7b5a3a3c5d2a8d1a47c34c22f116d8b521d5167d40b826`; 9 pages.

## Problem, task and assumptions

Discrete R2R/REVERIE with RGB-D, semantic segmentation and position. Addresses transient imagined views by retaining them in persistent hybrid memory.

## Method, training and inference

Sections 3.1–3.3, pp. 3–5: graph memory distinguishes visited/current, reachable-candidate and imagined locations. Features, positions and temporal markers feed separate real/imagined node encoders and instruction-conditioned graph attention. Imagined scores are gated and added to the nearest reachable candidates (Eqs. 3–4). A recurrent imagination tree projects historical depth/semantics into future poses, completes missing structure, adjusts semantics using a room/object dictionary, synthesizes RGB with SPADE and predicts additional waypoints. The tree is merged into persistent memory; cosine similarity minus positional MSE controls capping/pruning (Eq. 1). Separate pretraining uses semantic cross-entropy, depth L1, room-type cross-entropy, GAN/perceptual/feature-matching losses and waypoint heatmap MSE. Navigation pretraining uses masked language, region, action and object-grounding tasks before encoder fine-tuning. The authors report 100,000 pretraining iterations at batch 32 and 20,000 fine-tuning iterations at batch four, using one then four Quadro RTX 8000 GPUs. Inference repeatedly updates the hybrid graph and selects a reachable or stop action.

## Novelty and closest predecessors

Combines Pathdreamer-style generative components with persistent imagined nodes and adaptive graph action fusion, distinguishing it from transient DREAMWALKER branches and HNR feature-only anticipation. It is already strong prior art against proposing simply storing imagination in memory.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p. 6 reports R2R unseen SR 82/SPL 78 and test 79/74; ScaleVLN79/70 and 77/68. REVERIE unseen SR 58/RGS 38/RGSPL 28, test 56/34/25. These are author claims, not reproduced. Table 1 also attributes an implausibly strong REVERIE Seq 2 Seq row relative to other named methods; treat cross-paper baseline labels with caution. Fig. 5 stratifies complex room/object instructions. Generated-image quality is visual/PSNR evidence, not proof of correct hidden room layouts. No verified implementation currently permits independent reproduction. Table 2 on p. 7 reports full-system ablation SPL of 70/71 (SR 80/82), whereas the main R2R headline gives SPL 78. These configurations cannot be silently treated as the same experiment; the source does not permit a complete reconciliation.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: No verified official implementation located in the recorded search.. Snapshot commit: `not available`. Retrieval: 2026-10-22.

The full paper and its ablations were reviewed; no verified official model/training repository was located. Different headline and ablation configurations remain unresolved from source.

Inspected files: No implementation files inspected; access scope is stated above. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Persistent imagined nodes risk treating hallucinated evidence as measured truth; score fusion and pruning use heuristics without calibrated source confidence. Room-object dictionary can reinforce atypical-layout errors. Multiple generation modules and sensors add training/data/compute cost; large reported gains need matched-code replication and uncertainty estimates. Later latent/predictive methods shift the pixel-generation cost, but do not settle provenance-aware imagined-memory reliability.

Relevant follow-up relationship: HNR; NavMorph; SC²-WM; GenEx. Each changes prediction/memory representation, not a demonstrated solution to false persistent memory.

## Access and uncertainty record

Full text is preserved under `corpus/sali/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/sali/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-sali-persistent_hallucination

**Still unresolved in the reviewed literature — Methodological.** Persistent imagined nodes can retain incorrect room/object structure as planning evidence.

Conditions: Atypical layouts or errors in generated unseen views. Cause: Heuristic pruning and score fusion do not provide calibrated provenance weights separating observed from synthesized content.

Evidence: DEDUCTION: imagined graph/memory method; qualitative imagery and PSNR are not hidden-world truth tests.

Follow-up assessment: HNR/NavMorph/SC²-WM alter representations and adaptation; GenEx improves consistency. None directly demonstrates correct recovery from false persistent SALI memory. Confidence: Medium.

### 02-sali-replication

**Uncertain because evidence is insufficient — Implementation-specific.** Headline and ablation performance cannot be fully reconciled from the inspected material.

Conditions: Attributing the large reported gain to imagination rather than a configuration or baseline difference. Cause: Main R2R SPL 78 differs from full-system ablation SPL 70/71, and no verified official code resolves the configuration mapping.

Evidence: PAPER EXPERIMENT: Table 1 p.6 versus Table 2 p.7; source/access record in dossier. These numbers are author-reported and unexecuted.

Follow-up assessment: No reviewed follow-up supplies a matched independent reproduction or configuration reconciliation. Confidence: Medium.
