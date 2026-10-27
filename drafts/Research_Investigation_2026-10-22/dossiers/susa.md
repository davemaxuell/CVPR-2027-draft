# SUSA

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Agent Journey Beyond RGB: Hierarchical Semantic-Spatial Representation Enrichment for Vision-and-Language Navigation**. Xuesong Zhang, Yunbo Xu, Jia Li, Ruonan Liu, Zhenzhen Hu AAAI 2026 (2026).

Primary record: https://github.com/HCI-LMC/VLN-SUSA



Supplied main file: [SUSA.pdf](../../Paper/SUSA.pdf). Pages actually reviewed: 1,3–7. Associated local versions are recorded below with SHA256 identifiers. Exact duplicate bytes were not reread. Distinct alternatives were checked for the relevant method/results; their complete bibliographies were not separately reread.

Associated material:

- SUSA (alternate).pdf; reviewed pages/status: Exact SHA256 duplicate; no separate reread; SHA256 `07e8ca776eb6b523355eb41cdbfca8180ca312acd78be90f431dd0be98e4f48c`

## Problem, motivation, and operating assumptions

DUET RGB-only representations under use semantics and geometry. SUSA adds view-level caption and trajectory-level depth branch while preserving dual-scale action fusion. Requires GT simulator depth and pretrained caption features; no human annotations does not mean no extra sensors/pretraining.

## Method, training, inference, and equations

Section 3 augments DUET at two scales. The local semantic branch generates 36 BLIP2-FlanT5-XXL view captions and encodes them with CLIP. Static caption/instruction matching takes maximum word similarities from a 36×L matrix, where L is instruction length. Dynamic cross-attention adds context-dependent matching. The printed Eqs 1–3 are not completely explicit about the projection needed to align vector scores and transformer outputs; the released branch fusion is described separately below.

The global spatial branch uses a Gibson-pretrained ResNet50 depth encoder alongside CLIP RGB features. A shared two-layer panorama transformer and average pooling produce768-dimensional topological node representations; cross-modal and graph attention relate them to the instruction. The four branches are global RGB, global depth, local RGB and local semantic features. Their pooled representations and language feed fusion and symmetric contrastive alignment (Eqs 4–9).

The reported fine-tuning objective is0.2 times expert-action CE plus DAgger-action CE plus 0.8 times symmetric contrastive loss (Eqs 10–11). For an action distribution p_t over candidates, CE is −log p_t(a*_t); the contrastive objective aligns matched visual/language representations while using other batch examples as negatives. The paper reports partial RGB-base pretraining with MLM/MRC/SAP/object-grounding tasks and 100K iterations at batch 32. Fine-tuning uses 25K iterations and task-dependent batches on one RTX4090. Continuous-environment transfer uses an ETPNav planner/executor. Inference fuses branch logits within the DUET policy; the captions are preprocessed representations, not generated step-by-step reasoning.

## Novelty and closest predecessors

Multi scale complementarity: local caption matching and global depth memory rather than naive feature concatenation; contrastive fusion and partial pretraining. DUET derivative with VLM preprocessing, not standalone generative VLM policy. CONSOLE already text knowledge feature enrichment; SUSA adds depth branch and caption matching.

## Results and independent verdict

Table 1 R2R 73 SR/64.8 SPL vs DUET 72/60; test 72.5/63.8 vs 69/59. REVERIE test 41.5 SPL /27.3 RGSPL vs 36/22. Table 3 CE 52.7/43.6 vs DUET 51.9/41.2, but below ETPNav 57/49. Table 4 upgrading RGB ViT→ CLIP alone 46.9→51.5 SR; Gibson depth 52.0/39.2 versus ImageNet depth 50.1/34.6. Table 6 HAP reduces val 55→51.7 SR but improves test 51.9→54.3; this is not monotonic component benefit. §4.3 text reversesδstatic/dynamic relative Eq 3; exact config code important.

## Implementation audit

INSPECTED CODE. map_nav_src/r2r/agent.py:75–120 loads separate RGB, depth and semantic feature arrays. models/vilmodel.py:978–1049 uses distinct global RGB/depth and local RGB/semantic encoders, predicts four fusion weights and sums branch logits. The weights are sigmoid outputs followed by softmax at 1025–1028, so their relative ratio is bounded by e; this is a mathematical consequence of the inspected implementation, not a measured performance limit.

agent.py:27–36 attention-pools each representation and defines symmetric cross-batch CE;551–585 adds action CE and a 0.8-weighted contrastive term. The contrastive fusion in 583 is an unweighted sum, with the weighted alternative commented out. pretrain_cmt.py exposes inherited MLM, MRC, SAP and object-grounding losses, but no explicit HAP task appears in its inspected dispatch at 71–124. Therefore the paper's historical-action pretraining provenance is not established by this file. The feature-generation pipelines and exact HAP training configuration remain unverified. No project execution occurred.

Repository: https://github.com/HCI-LMC/VLN-SUSA. Inspected snapshot commit: `03b48cb289650374f85ef017c549ddab5174141a`. Retrieval status: Commit-pinned source retrieved through sparse git checkout. Project code was not executed.

- [map_nav_src/models/vilmodel.py:800-810,978-1049](https://github.com/HCI-LMC/VLN-SUSA/blob/03b48cb289650374f85ef017c549ddab5174141a/map_nav_src/models/vilmodel.py)
- [map_nav_src/r2r/agent.py:27-58,75-120,551-585,659-664](https://github.com/HCI-LMC/VLN-SUSA/blob/03b48cb289650374f85ef017c549ddab5174141a/map_nav_src/r2r/agent.py)
- [pretrain_src/model/pretrain_cmt.py:71-124,145-149,198-207,264-269](https://github.com/HCI-LMC/VLN-SUSA/blob/03b48cb289650374f85ef017c549ddab5174141a/pretrain_src/model/pretrain_cmt.py)

## Bottlenecks and follow-up chain

- **SUSA-1: Caption/depth branches bring pretrained-data/sensor dependence and inconsistent split gains** Conditions: Domain-shifted depth, missed caption landmarks, fusion tuning Cause: Frozen perception and weighting can amplify semantic error; CLIP upgrade confounds module claim Evidence: Table 4 feature ablation, Table 6 val/test tradeoff, Eq 3/proseδmismatch Type: Methodological / protocol. Follow-ups:  CONSOLE / MapNav / SkillNav different text interfaces; sensor-matched and caption-error controlled analysis still needed. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

The main file and its byte-identical alternate were accounted for. The four-branch model, fine-tuning losses and inherited pretraining dispatch were inspected; complete caption/depth generation and the claimed historical-action pretraining configuration remain unverified. No project code was executed.

