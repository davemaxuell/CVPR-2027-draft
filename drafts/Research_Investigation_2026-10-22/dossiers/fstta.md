# FSTTA

Fast-Slow Test-Time Adaptation for Online Vision-and-Language Navigation

Junyu Gao; Xuan Yao; Changsheng Xu. ICML 2024. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [FSTTA.pdf](../../Paper/FSTTA.pdf). Pages reviewed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18].

## Problem and operating conditions

FSTTA adapts a pretrained navigation policy during a stream of unlabeled test episodes. The policy sees instructions, panoramas, object/history features and a changing candidate graph; no success feedback is required. The targeted tradeoff is rapid per-step adaptation versus accumulation of erroneous updates. It is gradient-based test-time learning, not training-free inference.

## Mechanism, training, and inference

The original policy (principally DUET, also HM3D-scaled variants and continuous baselines) supplies action probabilities. Entropy H(p)=−Σp log p gives a self-supervised loss. Only a small set of late normalization affine parameters is intended to update. Every M steps, centered recent gradients form G∈R^(M×D); covariance eigenvectors define axes, and the mean gradient is reweighted inversely by eigenvalue before length normalization (Eqs 3–5). Gradient-variance departure from a historical EMA scales the learning rate within bounds (Eq 6).

Every N episodes, a slow stage analyzes N+1 parameter snapshots. A recency-weighted displacement from a historical reference supplies a direction; high-variance parameter axes receive more weight, sign-aligned to that displacement (Eqs 8–11). Thus fast and slow stages intentionally favor opposite covariance eigenspaces. Default M=3,N=4, fast/slow rates 6e−4/1e−3 and q=.1 apply to DUET; R2R-CE uses M=7 and different rates. Batch 1 evaluation shuffles each split five times; parameters persist across episodes (§4, AppendixA).

DEDUCTION: centered covariance rank is at most M−1, so for M=3 and D=768 the literal inverse-eigenvalue formula has at least 766 zero eigenvalues. Numerical stabilization is essential, not optional. The released fast helper floors eigenvalues, but its omission of the paper’s norm calibration changes the update magnitude. SVD/PCA requires additional memory and computation; no measured runtime was reproduced.

## Novelty and closest predecessors

Unlike Tent-style independent entropy steps or resetting at each episode, FSTTA couples recent within-episode gradients and cross-episode parameter trajectories. Its novelty is an optimization schedule and geometric update transformation, not a new semantic navigation model. Later FeedTTA/ATENA add external outcome feedback, changing the information available at deployment.

## Results and independent verdict

PAPER EXPERIMENT: REVERIE unseen validation DUET SR 46.98/SPL 33.73 becomes 54.15/36.41; test SR 52.51 becomes 53.40 (Table 1). The smaller test gain matters. Ablation Table 2/p7 gives 49.74 SR with fast updates,49.82 with dynamic rate and 54.15 with slow updates. R2R unseen SR 72→75/SPL 60→62 also increases path length 13.94→14.64m (Table 6), consistent with an exploration/efficiency tradeoff. Continuous BEVBert test SR 59→60 while NE worsens 4.70→5.45m (Table 12); gains are not uniform across metrics.

The forgetting test returns to seen validation after adapting on unseen data and retains similar SR (Table 3). This supports retention for that finite sequence, not a guarantee under arbitrary lifelong shifts. Authors explicitly acknowledge shuffled static test episodes as an imperfect online deployment simulation, fixed update frequencies and added cost (§5). Some copied baseline rows violate OSR≥SR (for example DREAMW Table 8), so comparative table transcription needs cross-checking. Verdict: meaningful unlabeled TTA idea with controlled ablations, but confidence optimization can still reinforce a wrong action and the current release has material reproducibility defects.

## Implementation and reproducibility

INSPECTED CODE: Official pinned source is read-only. map_nav_src/models/FSTTA.py implements the active FAST class, covariance eigendecomposition with 1e−6 floor (lines 25–69), entropy backward and update (76–169), and the slow module (254–336). The fast path writes the inverse-covariance direction directly into p.grad at 138 without Eq 5 length calibration. Its entropy helper at 344–346 computes softmax times log_softmax without masking negative infinities; DUET masked logits can therefore require additional numerical handling.

The agent initializes FAST once per rollout (agent_obj.py:374–382), preserving its gradient window within an episode, and maintains SLOW snapshots across episodes (384–408,588–628). However agent_obj.py:37 imports principal_grad from FSTTA.py, and the inspected module defines no such function. The slow PCA helper at 602–609 uses norm/vector rather than vector/norm. These are static release findings, not evidence that published experiments used this exact snapshot. Model/pretrain and graph evaluator paths inherit DUET. Dataset preparation remains marked TODO in the tree. Reproduction requires repairing and separately validating the release, obtaining simulator/features/checkpoints, and resolving update-normalization differences. No fetched code was executed.

Inspected release reference: [https://github.com/Feliciaxyao/ICML2024-FSTTA at 3481c5457c00f2a12a09264111816e9cdf3a6bf7](https://github.com/Feliciaxyao/ICML2024-FSTTA/tree/3481c5457c00f2a12a09264111816e9cdf3a6bf7). File-level inspection is restricted to the paths listed below.

- map_nav_src/models/FSTTA.py:25–169,254–346,461–735
- map_nav_src/reverie/agent_obj.py:29–37,350–455,588–628
- map_nav_src/reverie/agent_base.py:40–90
- map_nav_src/run_reverie_tta.sh

## Specific bottlenecks

**fstta-B1: Unlabeled entropy adaptation has no semantic signal identifying confidently wrong decisions.**

Conditions: Policy is confident on a wrong landmark or route in a shifted environment. Cause: Entropy loss rewards sharper predictions independently of instruction completion. Evidence: Eq 2; later FeedTTA/ATENA explicitly study outcome-conditioned alternatives.

Type: methodological. Status: Shifted into another cost or limitation. Confidence: High for objective information content; failure frequency is setting dependent.. Follow-ups: FeedTTA and ATENA use binary external or self-predicted outcome labels, trading the unlabeled setting for feedback dependence.

**fstta-B2: The inspected release does not directly instantiate a reproducible paper-equivalent update path.**

Conditions: Running the provided REVERIE agent at the pinned commit. Cause: Missing imported function and differences in normalization/numerical handling. Evidence: agent_obj.py:37; FSTTA.py:138,344–346,602–609; later papers reference issue1.

Type: implementation-specific. Status: Uncertain because evidence is insufficient. Confidence: High for static code differences; no full execution attempted.. Follow-ups: FeedTTA and ATENA state they reimplement FSTTA; their results must identify which implementation is compared.

## Sources and limitations

- [Supplied main paper](https://raw.githubusercontent.com/mlresearch/v235/main/assets/gao24p/gao24p.pdf)
- No simulator, benchmark or fetched project code executed. Repository snapshot findings do not establish which code produced the paper results.
