# CONSOLE

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Correctable Landmark Discovery via Large Models for Vision-Language Navigation**. Bingqian Lin, Yunshuang Nie, Ziming Wei, Yi Zhu, Hang Xu, Shikui Ma, Jianzhuang Liu, Xiaodan Liang IEEE TPAMI 2024 (author publication record); DOI10.1109/TPAMI.2024.3407759 (2024).

Primary record: https://arxiv.org/abs/2405.18721

The first author's publication page confirms IEEE TPAMI2024: https://expectorlin.github.io/. The supplied paper is the expanded journal treatment, not a CVPR main-conference paper.

Supplied main file: [CONSOLE.pdf](../../Paper/CONSOLE.pdf). Pages actually reviewed: 1, 3-8. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Use broad landmark co-occurrence knowledge to improve supervised graph VLN without blindly trusting language-model priors. Operates on panoramic candidate views with HAMT or DUET , not a training-free navigator.

## Method, training, inference, and equations

PAPER METHOD (§D): ChatGPT extracts navigation-order landmarks and likely co-occurring objects before navigation. A frozen CLIP representation compares the current panoramic average with adjacent landmarks to advance a pointer; a forced shift after 1–2 steps prevents stalling. Per-view CLIP similarities create landmark/co-occurrence view distributions. A learned state projection of instruction CLS and visual features produces relevance scores; weighted landmark text features are added to each observation feature before the backbone's cross-modal action predictor (Eqs.11–16). Training combines the backbone imitation/ A 2C objective, action-label consistency for corrected landmark scores, and symmetric observation/landmark contrastive loss (Eqs.14,17,18). Temperature.5, auxiliary weights.1, SGD learning rate.1 for the scoring projection, batch 4 and 300 K iterations on 3090 are specified. Large augmented sets reuse a landmark vocabulary instead of new ChatGPT calls for every item.

## Novelty and closest predecessors

A learned correction layer makes open-world co-occurrence priors useful to existing specialist policies; contrasts ESC's exploration and KERM's external knowledge. New contribution lies in weighting and representation enhancement, not language extraction alone. Component ablations separate learnable scoring and observation enhancement.

## Results and independent verdict

PAPER EXPERIMENT: Table 2 p.8 reports DUET val-unseen 72 SR/60 SPL versus CONSOLE 73/63; test 69/59 versus 72/61. HAMT val-unseen 66/61 versus 69/63. ScaleVLN 's larger-data 81/70 is explicitly excluded from fair same-data SOTA interpretation. RxR uses English only. DEDUCTION: Eq.13 is a signed weighted sum of distributions without explicit positivity/normalization, yet Eq.14 calls it a probability inside log; implementation should clarify logits versus probabilities. The method's hard pointer may skip an unmet landmark despite corrected visual scores.

## Implementation audit

INSPECTED CODE. finetune_src/models/vln_clip.py:33–75 computes CLIP-normalized landmark similarities, selects the current/next landmark, and combines co-occurrence features. Candidate scores are softmax-normalized over candidates, whereas co-occurrence weights are scaled dot products and are not softmax-normalized at 57. Thus the latter can be signed; calling them probabilities or a convex mixture would overstate the code. agent_cmt.py:347–424 advances the landmark index with both visual evidence and a no-shift counter.

agent_cmt.py:436–439 adds supervised action CE;487 adds candidate consistency CE;490–511 builds a candidate-by-candidate similarity matrix and symmetric contrastive CE with padded targets ignored. Lines 700–709 add consistency only when train_ml is active and contrastive loss only when train_rl is active. The inspected agent is the HAMT/CMT integration, not a verification that every DUET variant follows identical control flow. Model, feature construction, loss and rollout deltas were inspected; complete preprocessing of generated co-occurrence annotations and a matched DUET evaluator were not independently traced. No training/evaluation executed.

Repository: https://github.com/expectorlin/CONSOLE. Inspected snapshot commit: `7417d4f4d454c8157b913a03b04838f9d4527b54`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [finetune_src/models/vln_clip.py:1-90](https://github.com/expectorlin/CONSOLE/blob/7417d4f4d454c8157b913a03b04838f9d4527b54/finetune_src/models/vln_clip.py)
- [finetune_src/r2r/agent_cmt.py:93-108,347-424,436-439,480-511,700-709](https://github.com/expectorlin/CONSOLE/blob/7417d4f4d454c8157b913a03b04838f9d4527b54/finetune_src/r2r/agent_cmt.py)

## Bottlenecks and follow-up chain

- **03-console-prior: Learned co-occurrence correction does not verify actual landmark completion** Conditions: Repeated landmarks or unusual co-occurrences Cause: Heuristic monotone pointer and prior-based feature fusion can encode wrong stage Evidence: §D.2.1 p.6 forced shifting and Eq.13 Type: methodological. Follow-ups:  CA-Nav introduces explicit checks; GC-VLN uses graph feasibility; SkillNav specializes grounding skills. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

Principal method and results of the supplied expanded paper and the HAMT/CMT code integration were reviewed. The author publication record confirms TPAMI2024. All alternative integrations and preprocessing assets were not independently reconstructed. No benchmark was executed.

