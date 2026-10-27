# VLN-BERT

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**VLN BERT: A Recurrent Vision-and-Language BERT for Navigation**. Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, Stephen Gould. CVPR 2021.

Primary paper: [VLN BERT: A Recurrent Vision-and-Language BERT for Navigation](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_VLN_BERT_A_Recurrent_Vision-and-Language_BERT_for_Navigation_CVPR_2021_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: VLN-BERT.pdf (Main PDF; 11 pages; SHA256 65ed08cfa0495690d46362215014729f145e7b4d2c1ee8e9b9cd8cc981a57899).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Adapt pretrained vision–language Transformers to sequential, partially observed action selection without an additional LSTM navigation backbone. This is the recurrent VLN-BERT paper by Hong et al., not the earlier discriminative VLN-BERT path-ranking paper.

## 3. Mechanism, training, and inference

PDF §3 pp 3–5: encode the instruction once and use its CLS output as initial state s_0∈R^d. At each step a Transformer consumes recurrent state st−1 and current panoramic candidates, attending to fixed language features. Text and visual attention summaries are multiplied elementwise and concatenated with the raw state before projection (Eq 12); the selected action angle is incorporated into the next state (Eq 13). Attention from the state to visual candidates supplies the action distribution. REVERIE additionally predicts the target object and adapts state refinement. OSCAR and PREVALENT initializations are separate variants; the PREVALENT implementation adapts its cross-modal architecture for recurrent state+vision inputs. Eq 14 combines actor-critic RL with supervised imitation CE; the scalar policy term uses discounted advantage and the supervised term −logπ(a*t). Object grounding adds CE on REVERIE. Reported AdamW learning rate is 1e−5; R2R 300k steps/batch 16 and REVERIE 200k/batch 8 on one 2080Ti. Inference recurrently updates the state and greedily acts, rather than ranking complete paths.

## 4. Novelty and evidence

The contribution is recurrent reuse of a pretrained multimodal Transformer and state-directed attention for navigation. Initialization and architecture are jointly important: Table 1 compares random, OSCAR, and PREVALENT variants. It should not be grouped as a general-purpose instruction-following VLM with training-free inference.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 reports R2R test SR/SPL 63/57 for PREVALENT initialization, 57/53 for OSCAR, 51/47 for random initialization. Table 2 reports REVERIE test SR 29.61, remote-grounding success 16.50 and RGSPL 13.51. VERDICT: substantial pretrained-representation benefit is evident, but comparisons do not independently establish semantic reasoning or resilience to long-history aliasing.

Table 3, PDF p 8, holds data and training strategy fixed for OSCAR-based architecture ablations: recurrent attention without the explicit matching refinement gives unseen SR 57.22/SPL 52.57, while the full model gives 58.71/53.41. Table 4 compares text-attention schedules under an 11GB budget, but batch size differs (6 versus 16); thus it is not a pure attention-only causal comparison. The full fixed-text variant uses 9.2GB at batch 16 in that setup.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/YicongHong/Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT); inspected snapshot **ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e**. Snapshot metadata and source are under `repositories/01/vlnbert`.

Inspected PREVALENT recurrent Transformer, policy/optimization loop, training driver, and evaluation file. Paper and code agree that state and visual inputs evolve while the language representation serves as persistent context.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [r2r_src/vlnbert/vlnbert_PREVALENT.py](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e/r2r_src/vlnbert/vlnbert_PREVALENT.py) — forward lines 322–327 and mode dispatch near384: The recurrent state is concatenated with current visual features for the visual/cross-modal branch.
- [r2r_src/agent.py](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e/r2r_src/agent.py) — lines 527–544: Imitation and RL losses are optimized with gradient clipping40.
- [r2r_src/eval.py](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e/r2r_src/eval.py) — Evaluation scoring functions: Scoring computes endpoint and path measures; no instruction counterfactual evaluation is implemented in the inspected evaluator.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **vlnbert-B1 — Partially addressed**. A fixed-dimensional state must compress the full trajectory, so histories requiring distinct decisions can become aliased. Evidence: Eq 12–13 and recurrent state architecture; this is a capacity deduction, not a demonstrated universal failure. Type: architectural. Follow-ups: HAMT explicit history, DUET graph memory. Confidence: medium.
- **vlnbert-B2 — Partially addressed**. Headline accuracy depends strongly on pretrained initialization; pure architecture claims need matched initialization/data controls. Evidence: Table 1 p 6 random versus OSCAR/PREVALENT. Type: experimental confounding. Follow-ups: HAMT Table 2 matched-feature comparison. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/cshizhe/VLN-HAMT
