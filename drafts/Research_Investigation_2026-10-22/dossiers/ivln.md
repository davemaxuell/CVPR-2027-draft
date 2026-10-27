# IVLN

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Iterative Vision-and-Language Navigation**. Jacob Krantz, Shurjo Banerjee, Wang Zhu, Jason Corso, Peter Anderson, Stefan Lee, Jesse Thomason. CVPR 2023.

Primary paper: [Iterative Vision-and-Language Navigation](https://openaccess.thecvf.com/content/CVPR2023/papers/Krantz_Iterative_Vision-and-Language_Navigation_CVPR_2023_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: IVLN.pdf (Main PDF; 10 pages; SHA256 c9fe6ac41c9f67d0c5bb7ea7c33f2e29ff8baa776c6143a2813533f517a6d414).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Iterative VLN asks whether a robot can exploit persistent experience across sequential instructions in the same building. Episodes form tours rather than independent resets. The protocol includes an oracle correction to the true goal and transport to the next start while the agent passively observes; this is meaningful additional supervision/experience.

## 3. Mechanism, training, and inference

§3 pp 3–4 constructs tours by approximately solving an asymmetric traveling-salesperson ordering over existing R2R paths, using connectivity partitions and a Lin–Kernighan heuristic. t-nDTW computes exp(−DTW(Rtour, Qtour)/(|Rtour|dth)), forbids cross-episode alignment by infinite costs, excludes oracle transport, then averages tours weighted by episode count. This discourages sacrificing one episode for unlimited exploration. TourHAMT keeps up to 50 previous steps with boundary tokens; TourCMA retains a visual GRU state; pooling variants store max-pooled latent history. MAP-CMA projects semantics/occupancy into a persistent metric map, crops 14×64×64 tensors, encodes to 128×4×4, and replaces CMA RGB inputs. Training distinguishes episodic/iterative/known maps and ground-truth/RedNet semantics. Latent baselines use teacher forcing with episode-truncated gradients; map models additionally use augmented pretraining, progress loss and DAgger. Inference persists maps across episodes but resets instruction-specific state.

## 4. Novelty and evidence

Persistent multi-episode benchmark and tour-level metric, with controlled memory-reset variants. The main scientific result is that naive longer latent history can hurt while explicit map reuse helps; it is not evidence that all unstructured memories fundamentally fail.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: §5.1 extending pretrained HAMT history roughly halves t-nDTW without adequate adaptation. §5.2 iterative map evaluation improves 3–4 points in t-nDTW over episodic resets when trained for iteration. Authors explicitly acknowledge map versus CMA comparisons are confounded by DAgger training and identify ground-truth/known-map variants as invalid leaderboard submissions. VERDICT: useful persistent-memory testbed, but oracle inter-episode correction and static scenes differ from autonomous lifelong deployment.

Table 4/p 8 separates train/evaluation map regimes and predicted versus ground-truth semantics. For predicted semantics, iterative-map training/evaluation reaches 35 SR/32 SPL/47 tour-nDTW, while training/evaluating with a fully known map reaches 31/28/44. More map information alone does not guarantee better use of it. The human R2R comparison in the conclusion uses a different benchmark/protocol and should not be interpreted as a controlled IVLN human gap.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/jacobkrantz/IVLN-CE](https://github.com/jacobkrantz/IVLN-CE); inspected snapshot **14417fc0930fc273183667c09e3e3564adc823a0**. Snapshot metadata and source are under `repositories/01/ivln`.

Official project links both discrete Bill1235813/IVLN and continuous jacobkrantz/IVLN-CE. The latter is pinned here; map policy, mapper, iterative trainer, tour dataset, measure definitions and configuration were retrieved for phase audit.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [ivlnce_baselines/models/map_cma_policy.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/models/map_cma_policy.py) — 276–368: Map/depth/text attention and episodic recurrent-state masks; optional tanh progress monitor with positive MSE.
- [ivlnce_baselines/trainers/iterative_dagger_trainer.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/trainers/iterative_dagger_trainer.py) — 33–94: Tour memory versus episodic resets, truncated gradient state, weighted action CE and auxiliary losses.
- [ivlnce_baselines/common/environments.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/common/environments.py) — 36–110: Iterative environment includes agent, oracle_goal and oracle_start phases, explicitly using oracle transit between episodes.
- [ivlnce_baselines/common/base_il_trainer.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/common/base_il_trainer.py) — 173–218;585–914: Weighted imitation update and iterative evaluation maintaining tour/episode masks and aggregating tour-nDTW.
- [ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/1_ftune_dagger.yaml](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/1_ftune_dagger.yaml) — 1–38: Predicted-semantics iterative mapper, 10 DAgger iterations, 5000 collected trajectories and progress auxiliary enabled.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **ivln-B1 — Partially addressed**. Memory contents learned under short episodic training do not automatically transfer to persistent tours. Evidence: §5.1 negative TourHAMT/TourCMA results. Type: methodological. Follow-ups: OVER-NAV structured open-vocabulary memory. Confidence: high.
- **ivln-B2 — Uncertain because evidence is insufficient**. Persistent map gains are established under oracle corrections, static scenes and accurate pose, leaving autonomous updating/forgetting under changes insufficiently tested. Evidence: §3 oracle phases and §4 maps. Type: protocol. Follow-ups: Context-Nav/continual-navigation frontier audit. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://arxiv.org/abs/2403.17334; https://jacobkrantz.github.io/ivln
