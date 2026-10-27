# AdaNav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**AdaNav: Adaptive Reasoning with Uncertainty for Vision-Language Navigation**. Xin Ding, Jianyu Wei, Yifan Yang, Shiqi Jiang, Qianxi Zhang, Hao Wu, Fucheng Jia, Liang Mi, Yuxuan Yan, Weijun Wang, Yunxin Liu, Zhibo Chen, Ting Cao ICML 2026; preprint first posted 2025 (2026).

Primary record: https://arxiv.org/abs/2509.24387

ICML's 2026 downloads listing and Microsoft Research's publication record identify ICML 2026. An older anonymous ICLR submission is not the final venue.

Supplied main file: [AdaNav.pdf](../../Paper/AdaNav.pdf). Pages actually reviewed: 1-8, 12-17. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

A trained monocular VLN-CE policy must choose when additional language reasoning is worth its latency. Inputs are instruction, RGB history, and internal action-token uncertainty; outputs are low-level navigation commands and optional descriptions, summaries, or corrections. Navigation training is required; transfer to RxR without RxR training is a different setting from the main mixed R2R / RxR training.

## Method, training, inference, and equations

AUTHOR CLAIM / PAPER METHOD (§3, pp.3–5): factor the policy into action generation, reasoning-mode selection, and conditional reasoning text. The selector has four outcomes: no reasoning, description, summary, correction. Token entropy H=−(1/N)Σ_tokenΣ_vocabulary p log p is computed from action-generation probabilities, then its history, current observation representation, and instruction are linearly combined to produce mode scores. A cold-start prior places soft mass near mode-specific entropy thresholds τ_m=τ_0+mδ. The heuristic weight is annealed toward zero so learned selection takes over. Reward combines geodesic progress with clipped relative reasoning-length cost; the latter uses the shortest successful response in a group (Eq.11). Inference runs the selector, conditionally generates reasoning, and conditions the action policy on accumulated reasoning.

Training (§4.1): 3,000 episodes each from R2R and RxR , five rollouts per episode, learning rate 10^-6 on four A 100 GPUs. Cross-dataset experiments instead use 3,000 R2R episodes. The 6 K count excludes backbone pretraining. DEDUCTION: the derivation leaves the dimensions/encoding of variable-length entropy history unspecified. Eq.6 calls its output logits, Eq.8 defines a normalized probability, Eq.9 mixes them, and Eq.10 applies another softmax, whereas Algorithm 1 samples the mixture directly. These alternatives are not equivalent. No optimality guarantee follows from using RL.

## Novelty and closest predecessors

Relative to fixed-interval NavGPT -style reasoning and direct-action NaVid / NaVILA , the contribution is a learned uncertainty-conditioned mode gate with a heuristic curriculum, not the invention of reasoning or uncertainty monitoring. Tables 6 and 8 separately test gate/prior/RL and input signals. It strongly overlaps any proposal to 'reason only when action entropy is high.'

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.6 reports NaVid 37.0 SR/35.0 SPL versus NaVid - AdaNav 47.7/42.34 on R2R-CE val-unseen; NaVILA improves 54.0 to 60.19 SR. These are different pretrained backbones and additional training, not equal-total-data comparisons. Table 6 p.8 compares fixed-five-step 40.12 SR, heuristic-only 42.53, no-prior RL 43.82, and full 47.7. Table 7 reports mean gate 0.0029 s, action 0.76 s, and reasoning 2.94 s; 2.5 reasoning calls per average 55-step trajectory lowers cost relative to dense reasoning. Appendix B calibrates thresholds on 1,000 validation episodes; the exact split must be disclosed to assess tuning leakage. Appendix F acknowledges missed transient ambiguity and adds 500 targeted examples; this is mitigation through extra training, not proof of calibrated uncertainty.

## Implementation audit

ACCESS LIMITATION: the paper advertises https://github.com/xinding-sys/AdaNav, but git ls-remote returned repository-not-found on 2026-10-22. No selector, optimizer, or reward implementation was inspected. The mathematical ambiguity therefore remains unresolved.

Repository: https://github.com/xinding-sys/AdaNav. Inspected snapshot commit: `unavailable`. Retrieval status: Access limitation. Command '['git', 'ls-remote', 'https://github.com/xinding-sys/AdaNav.git', 'HEAD']' returned non-zero exit status 128. Project code was not executed.


## Bottlenecks and follow-up chain

- **03-adanav-uncertainty: Confidently wrong or brief ambiguity can escape the reasoning gate** Conditions: Novel semantics, short lighting/view changes, or transient uncertainty Cause: Action token entropy is not calibrated probability of navigation failure; temporal averaging suppresses spikes Evidence: Author-acknowledged Appendix F, Table 17; deduction from Eq.5 and entropy averaging Type: methodological. Follow-ups:  AwareVLN uses supervised key-event triggers; Progress-Think models semantic progress. Neither establishes calibration of this gate.. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

The main paper and relevant appendices were reviewed. The advertised official code URL was inaccessible at the cutoff; no implementation or benchmark was executed.

- https://gwxuan.github.io/AwareVLN/
- https://horizonrobotics.github.io/robot_lab/progress-think
