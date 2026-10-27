# VLN-SIG

## Bibliography and reviewed material

**Improving Vision-and-Language Navigation by Generating Future-View Image Semantics**. Jialu Li; Mohit Bansal. CVPR 2023 (2023). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/CVPR2023/html/Li_Improving_Vision-and-Language_Navigation_by_Generating_Future-View_Image_Semantics_CVPR_2023_paper.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-6. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: VLN-SIG (supplement).pdf, pp. 1–3: objectives, weighted-patch argument, hyperparameters and additional evaluations.

- VLN-SIG.pdf; Main PDF; SHA256 `2a63b71dc0198335d65038decc2cf64d5e97b0cf4c42443559118405571caf37`; 10 pages.
- VLN-SIG (supplement).pdf; Supplement; SHA256 `d92c70a88b3e13e6cf3ac9900167798ebc9a3e8bf9aca86298352c9cb9803289`; 3 pages.

## Problem, task and assumptions

Improve HAMT/DUET representations through semantic image-token prediction in graphVLN. This is auxiliary training, not online imagined search.

## Method, training and inference

Section 3 pp. 3–5: frozen dVAE supplies 8192-token distributions over 16 pixel patches; mean, sampled or weighted distributions provide targets. HAMT adds masked-trajectory (MTM), masked-panorama (MPM) and action-conditioned future-image-semantic (APIG) tasks. Whole-image masking supports cached CLIP features. Weighted-patch heads condition on patch weights. Dynamic 1000-token vocabulary follows EMA frequency and difficulty (Eq. 6); soft-target CE/KL (Eqs.7–9) trains prediction. Finetuning retains IL/RL plus APIG. DUET omits MTM due to different history encoding. Inference uses the trained backbone without decoded future images or rollout planning.

## Novelty and closest predecessors

Prediction targets preserve more visual-token semantics than a single pooled feature; weighted-patch identifiability is a mathematical target property under rich weighting, not a finite-SGD guarantee.

## Results and independent verdict

Table 1 p. 6 HAMT unseen SR 66→68,SPL 61→62; test 65/60 unchanged. DUET test 69/59→72/60. Table 2 task baseline 64.4/58.8 versus all 68.1/62.3. Results support regularization with split-dependent gains.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/jialuli-luka/VLN-SIG. Snapshot commit: `603bbd2f40fe47b62c787af567d0cef5a462b301`. Retrieval: 2026-10-22.

INSPECTED CODE: `pretrain_src/model/pretrain_cmt.py:475–520,595–663` selects a dynamic vocabulary, applies log-softmax and KL loss against future visual-token distributions, and returns detached target/difficulty statistics. This verifies auxiliary semantic prediction rather than an inference-time image planner. The repository contains pretraining and finetuning entrypoints, but their complete dynamic-vocabulary update schedule, preprocessing caches and every HAMT/DUET evaluation branch were not exhaustively audited. The separately reviewed supplement documents the task mixture, vocabulary settings, and the fact that displayed decoded images retain 70% ground-truth tokens. No cached features, checkpoints or navigation evaluation were executed.

Inspected files: `pretrain_src/model/pretrain_cmt.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Token targets can prioritize common visual content rather than action-critical details; pretrained dVAE/CLIP and dynamic vocabulary determine information retained. No causal intervention establishes that deployment decisions rely on correct future predictions.

Relevant follow-up relationship: MonoDream and FantasyVLN also use training-only prediction; NavQ supplies inference-time future semantic graphs; VisualImaginations supplies generated landmark prototypes.

## Access and uncertainty record

Full text is preserved under `corpus/vlnsig/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/vlnsig/`. Supplement p. 2 gives dynamic vocabulary size 1000, frequency/difficulty weights gamma=1 and lambda=0.5, task-mixture/masking settings, and about 30 hours on two A6000 GPUs. Its weighted-patch result assumes equality over sufficiently rich weights and a linear decomposition; it is not a finite-optimization guarantee. Supplement p. 3 decodes visual examples retaining 70% ground-truth tokens, so these images do not demonstrate autonomous future-image synthesis. No benchmark was executed.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-vlnsig-causal_use

**Uncertain because evidence is insufficient — Insufficiently tested.** Auxiliary future-token prediction has not been shown to improve actions specifically by forecasting correct future content.

Conditions: Distinguishing representation regularization from deployed foresight. Cause: Future-token and semantic objectives train the encoder; dVAE/CLIP targets and vocabulary select what information is retained, while prediction heads are not an explicit deployment simulator.

Evidence: PAPER EXPERIMENT: Table 2 p.6 baseline SR/SPL 64.4/58.8 versus all 68.1/62.3; supplement p.3 visual reconstructions use 70% ground-truth tokens, so they are not autonomous predicted images.

Follow-up assessment: MonoDream/FantasyVLN share training-time prediction; NavQ inserts predicted semantics during inference. No matched intervention demonstrates causal future-content use in VLN-SIG. Confidence: Medium.
