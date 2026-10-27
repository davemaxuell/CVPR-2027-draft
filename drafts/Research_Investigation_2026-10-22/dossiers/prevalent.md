# PREVALENT

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-Training**. Weituo Hao, Chunyuan Li, Xiujun Li, Lawrence Carin, Jianfeng Gao. CVPR 2020.

Primary paper: [Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-Training](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hao_Towards_Learning_a_Generic_Agent_for_Vision-and-Language_Navigation_via_Pre-Training_CVPR_2020_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: PREVALENT.pdf (Main PDF; 10 pages; SHA256 3a00fee944f0860a1e9bd816fea2d6026bec69ac163161c7b6f50c6b4d9ebe64).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Pretrain representations using navigation-specific image–instruction–action triplets to transfer across embodied instruction tasks. Unlike later history-aware policies, its pretraining action prediction conditions on a current observation and instruction.

## 3. Mechanism, training, and inference

PDF pp 3–5: 36 panoramic ResNet view features (2048 visual+128 angle dimensions) are projected into a 768-dimensional Transformer with 9 language, 1 vision and 3 cross-modal layers. Masked language modeling hides 15% of words; action prediction uses a CLS representation to classify the next panoramic direction. For triplet (W, V, a), L_AP=−log pθ(a|W, V); MLM is cross entropy of masked words conditioned on unmasked language and views. These objectives do not explicitly supervise temporal memory. Training uses 104k human-derived triplets and approximately 6.482M synthetic triplets from 1.02M generated instructions, totaling 6.582M. Reported pretraining uses 8 V100 GPUs, 20 epochs, AdamW at 5e−5. Downstream R2R integrates pretrained language features into an EnvDrop/LSTM agent; this differs from later recurrent VLN-BERT, which recurrently uses the multimodal backbone. Inference follows the trained downstream policy.

Section 6.5, PDF p 8, clarifies that most downstream experiments use frozen pretrained features for computational efficiency. R2R additionally uses a two-stage adaptation that unfreezes cross-attention after feature-based training. These schedules should not be collapsed into a single claim that all pretrained parameters are always updated.

## 4. Novelty and evidence

Navigation-specific multimodal/action pretraining is the contribution, rather than a new online map or world model. Transfer across R2R, CVDN, HANNA supports usefulness of the representation; the data scale and downstream policy remain material parts of the result.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 reports test SR 54/SPL 51 for the single-instruction setting and 59/56 for the multiple-instruction setting. They are separate protocols. VERDICT: effective task-oriented pretraining, but its state-independent pretraining cannot establish history-sensitive reasoning and synthetic instruction quality is not independently certified.

Table 5, PDF p 8, gives test SR/SPL 54/51 for two-stage R2R adaptation versus 52/49 using fixed features. Table 4 shows small and nonuniform increments from adding action pretraining on CVDN: for all dialog text and mixed supervision, goal progress is 3.15 versus 3.14 with multimodal MLM alone. The much larger comparison against ordinary BERT cannot be attributed solely to the action objective. HANNA Table 3/p 7 increases unseen SR from 47.45 to 52.91 while assistance requests also rise from 5.8 to 6.6.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/weituo12321/PREVALENT](https://github.com/weituo12321/PREVALENT); inspected snapshot **4ccf59be11a9f7fb6b28f43631fd147edbaecf37**. Snapshot metadata and source are under `repositories/01/prevalent`.

The pinned main repository contains preprocessing assets/connectivity and README links to official PREVALENT_R2R and PREVALENT_HANNA applications. It is inaccurate to claim that the main repository itself contains the complete pretrained model/trainer. Application source inspection is tracked in the shared implementation supplement; release links alone are not execution evidence.

The top-level official repository points to PREVALENT_R2R. That application repository was pinned separately at 868fb53d6b7978bbb10439a59e65044c811ee5c2 and its pretraining heads, listener, evaluator, training driver and launcher were inspected. Links below explicitly use this separate commit rather than incorrectly attributing the files to the top-level repository.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [r2r_src/r2rpretrain_class.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/r2rpretrain_class.py) — 104–145: Application repository has MLM/action heads with positive CE sum; this is inspected source, not a full reproduction of original pretraining.
- [r2r_src/agent.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/agent.py) — 167–174;353–503: Listener training combines teacher CE and actor-critic policy/value/entropy terms.
- [r2r_src/eval.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/eval.py) — 51–107: Final/nearest goal distances and graph path length yield SR/OSR/SPL.
- [r2r_src/train.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/train.py) — 181–250;444–479: Training loop, validation-SPL checkpoint selection and split/vocabulary construction.
- [run/agent.bash](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/run/agent.bash) — 1–9: Sample-feedback listener configuration, RMS optimizer and80k iterations; exact pretrained model branch must be checked before reproduction.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **prevalent-B1 — Partially addressed**. Current-state action pretraining does not explicitly resolve visually aliased states requiring past observations. Evidence: §3 image–text–action triplet objective. Type: architectural/training. Follow-ups: Recurrent VLN-BERT, HAMT. Confidence: high.
- **prevalent-B2 — Partially addressed**. Millions of synthetic triplets still originate from a restricted set of environments and an imperfect speaker. Evidence: §4 pretraining data. Type: data. Follow-ups: ScaleVLN environment expansion, SRDF iterative consistency filtering. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/weituo12321/PREVALENT_R2R; https://github.com/YicongHong/Recurrent-VLN-BERT
