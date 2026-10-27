# RCM

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation**. Xin Wang, Qiuyuan Huang, Asli Celikyilmaz, Jianfeng Gao, Dinghan Shen, Yuan-Fang Wang, William Yang Wang, Lei Zhang. CVPR 2019.

Primary paper: [Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Reinforced_Cross-Modal_Matching_and_Self-Supervised_Imitation_Learning_for_Vision-Language_Navigation_CVPR_2019_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: RCM.pdf (Main PDF; 10 pages; SHA256 38dd3f1c3766a8a1174c80fda463351d9d1a9648d7aef87f4b8820901be4de23).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Use both goal-reaching reinforcement and route–instruction consistency to train a navigator, then adapt using unlabeled instructions in additional environments. Access to unseen environments during adaptation is a different evaluation regime from zero-shot unseen-scene generalization.

## 3. Mechanism, training, and inference

PDF pp 3–5: a recurrent reasoning navigator attends to instruction words and then to visual candidates. A fixed cross-modal matching critic estimates length-normalized instruction likelihood given an executed trajectory using a reverse sequence model. Supervised action-CE warm-start is followed by REINFORCE with extrinsic distance-progress/goal reward and intrinsic critic reward; γ=.95. For trajectory τ and instruction W, the policy term is −Σt logπθ(at|ht, W)At, where At includes return from these rewards. Self-supervised imitation learning (SIL) samples routes for unlabeled instructions, retains high-critic-score trajectories in replay, and imitates them. Eq 16’s reward-weighted objective and Eq 17’s simplified imitation update should not be described as algebraically identical without the authors’ approximation. Reported Adam rates are 1e−4 supervised and 1e−5 RL/SIL; ResNet is frozen and GloVe is fine-tuned. Inference uses the resulting trained navigator, not online gradient-free VLM prompting.

## 4. Novelty and evidence

Cycle-style matching reward and self-supervised route replay combine language consistency with goal reinforcement. The critic is a learned proxy, not an independent semantic oracle; self-training may inherit its preferences.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 test RCM SR 43.1/SPL 35; SIL in training environments 43/38; SIL with unseen environments 60.5/59. The latter explicitly uses unseen-environment adaptation and must not be compared as the same zero-shot condition. VERDICT: evidence for adaptation and consistency training, but not proof that the matching score certifies instruction entailment or that gains transfer without target-environment access.

Table 2, PDF p 7, gives unseen SR 42.5 for RCM, 40.6 after removing intrinsic reward and 37.7 for supervised-only training. Figure 6/p 8 explicitly shows a success with matching reward 0.53 and a failure with reward 0.54 that stops 5.5m from the goal. This author-provided example directly shows that a high consistency reward does not certify correct final grounding.

## 6. Implementation and reproducibility

An official executable repository was not verified in the assignment search scope. No implementation behavior is inferred from similar repositories. The paper’s objectives and experiment protocols are available; code reproducibility remains an evidence gap.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

**Execution status:** No model source inspection is claimed beyond the explicit scope above. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **rcm-B1 — Partially addressed**. Learned critic and policy can agree on semantically wrong but likely trajectories; no independent entailment guarantee is supplied. Evidence: Matching likelihood and SIL replay §3; Figure 6/PDF p 8 explicitly shows high intrinsic reward despite incorrect stopping. Type: self-training/verifier bias. Follow-ups: SRDF bidirectional consistency filtering, CAST counterfactual relabeling lead. Confidence: high for the published counterexample; unknown frequency.
- **rcm-B2 — Shifted into another cost or limitation**. Unseen-environment SIL trades zero-shot generalization for target-environment interaction/training. Evidence: Table 1 evaluation footnote and SIL setup. Type: protocol/resource tradeoff. Follow-ups: IVLN persistent experience, test-time adaptation methods. Confidence: high.

## 8. Access limitations and source record

Official code not located in documented search scope; this is not a claim that none exists. No benchmark executed.

The coordinator checked Xin Wang’s primary homepage (https://eric-xw.github.io/), whose RCM entry links paper/video/bibliography but no Code link. Nearby entries explicitly link code. This supports the bounded “official executable implementation not located” status, not a claim of universal nonexistence. The same page lists the TPAMI2020 extension Vision-Language Navigation Policy Learning and Adaptation as a follow-up lead.

Related primary sources: https://github.com/wz0919/VLN-SRDF; https://arxiv.org/abs/2508.13446
