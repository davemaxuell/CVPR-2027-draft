# NavGRPO

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Trajectory-Diversity-Driven Robust Vision-and-Language Navigation**. Jiangyang Li, Cong Wan, Songlin Dong, Chenhao Ding, Qiang Wang, Zhiheng Ma, Yihong Gong CVPR 2026 Findings (explicit supplied-PDF watermark; not main-conference assumption) (2026).

Primary record: https://arxiv.org/abs/2603.15370



Supplied main file: [NavGRPO.pdf](../../Paper/NavGRPO.pdf). Pages actually reviewed: 1, 3–8. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Improve DUET / ScaleVLN goal seeking and recovery after discrete action perturbations using RL beyond expert imitation. Privileged goal geodesic distance, shortest path and expert trajectories are training rewards; inference has no GT goal position.

## Method, training, inference, and equations

§3 Eqs 1–11: stochastic K=8 rollouts per instruction from DUET / ScaleVLN policyπ. Group advantage A_k=r_k−mean(r) without std normalization. Trajectory reward r=exp(−d_final²/(2ε²))1[d_final<ε]−.25 max(L−L*,0)/L*. The indicator makes it discontinuous atε despite'differentiability'prose; RL does not require path wise reward gradient. Step modifierγ=1+sign(A)(d_(t−1)−d_t)/L* yieldsγA=A+|A|Δd/L*. For negative A, approaching goal reduces penalty magnitude, contrary to universal'magnitude increases'wording. If|Δd|>L*,γcan flip; no clipping reported. PPO clipped ratio objective sums steps(no trajectory length normalization), KL to fixed referenceβ=.01,δ=.2. This remains credit assignment via shared trajectory advantage, not complete elimination of credit assignment difficulty. All-fail groups enter hard case buffer 200, supervised expert CE replay;30 K warm up then 200 K overall updates, lr 1 e−5, B 8, three seeds. At inference greedy graph policy; CE transfer uses waypoint predictor.

## Novelty and closest predecessors

Applies Dr. GRPO -style group relative optimization, geometric reward shaping and hard case imitation to existing DUET / ScaleVLN . Not a novel backbone; related RCM/HAMT already use RL. Closest novelty is controlled trajectory diversity/relative credit and robustness study.

## Results and independent verdict

Table 2 ScaleVLN 79/70→82/73 unseen SR/ SPL ; DUET 72/60→74/63. Table 3 continuous ScaleVLN 55/51→57/53. Table 1 least-probable first 3 actions 47.77→62.66 SPL . However perturbations depend on each policy's distribution, so neither sampled noise nor argmin actions are matched physical disturbances. Table 7 PPO 70.68 versus GRPO 72.65 uses different rewards; claim terminal rewards'cannot'be used with PPO / A 2C bootstrap is mathematically false. Table 8 hard replay adds 72.08→72.65 SPL . Table 6 K 16 only+.22 SPL over K 8 at double cost. Improvements are supported under reported protocol; optimizer-specific causal claim is not isolated.

## Implementation audit

INSPECTED CODE. main_nav.py:25 imports agent_grpo.GMapNavAgent. Its rollout_grpo_sample:508–665 samples trajectories and stores old-policy log probabilities; compute_advantages:959–971 subtracts the within-instruction reward mean without standard-deviation normalization. train_grpo:915–935 recomputes selected log probabilities, clips their ratio to the old policy, and minimizes the clipped surrogate. The actual reward switch at 724–744 offers success, SPL, progress, mixed or dense rewards; none of these inspected branches implements the paper's exponential terminal reward plus per-step progress coefficient. The README writes the paper equations, but this does not make them executable.

run_r2r.sh uses CLIP-B/16, DAgger-initialized checkpoints, batch 8, lr 1e-5,200k iterations and ScaleVLN annotations. main_nav.py:189–212 evaluates with argmax and selects the best unseen SPL. The source therefore provides a concrete GRPO baseline but does not yet establish exact reproduction of all claimed NavGRPO components. The README TODO still leaves perturbation-evaluation scripts and pretrained checkpoints unchecked despite a news line announcing release. This audit does not infer which private version produced reported results. No training or benchmark was executed.

Repository: https://github.com/Songlin-Dong/NavGRPO-CVPR2026. Inspected snapshot commit: `4668d42506858ea971c62e26fdc6e4b71b75064e`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [map_nav_src/r2r/agent_grpo.py:508-665,667-748,750-973](https://github.com/Songlin-Dong/NavGRPO-CVPR2026/blob/4668d42506858ea971c62e26fdc6e4b71b75064e/map_nav_src/r2r/agent_grpo.py)
- [map_nav_src/r2r/main_nav.py:19-70,150-212](https://github.com/Songlin-Dong/NavGRPO-CVPR2026/blob/4668d42506858ea971c62e26fdc6e4b71b75064e/map_nav_src/r2r/main_nav.py)
- [map_nav_src/scripts/run_r2r.sh:1-75](https://github.com/Songlin-Dong/NavGRPO-CVPR2026/blob/4668d42506858ea971c62e26fdc6e4b71b75064e/map_nav_src/scripts/run_r2r.sh)
- [README.md:49-63,108-122](https://github.com/Songlin-Dong/NavGRPO-CVPR2026/blob/4668d42506858ea971c62e26fdc6e4b71b75064e/README.md)

## Bottlenecks and follow-up chain

- **NAVGRPO-1: Robustness gains confound policy sharpness, perturbation location and reward choice** Conditions: Comparing noise policies and RL algorithms Cause: Each policy receives its own sampled/argmin perturbations; PPO reward differs Evidence: Table 1/§4.2 and Table 7/§4.4 Type: Methodological / evaluation. Follow-ups: Needs common externally specified perturbations, equal reward and rollout budget, recovery/route fidelity metrics. Status: **Partially addressed**, confidence Moderate.

## Source/access record and remaining uncertainty

The PDF advertises a different GitHub account from the inspected Songlin-Dong release. This provenance difference and the reward-code discrepancy are recorded; neither proves what private implementation produced the tables. No benchmark was executed.

