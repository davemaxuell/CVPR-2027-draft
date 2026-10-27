# RL4VLA

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**What Can RL Bring to VLA Generalization? An Empirical Study**. Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, Yu Wang NeurIPS 2025; reviewed arXiv v4 from January 2026 (2025).

Primary record: https://arxiv.org/abs/2505.19789



Supplied main file: [RL4VLA.pdf](../../Paper/RL4VLA.pdf). Pages actually reviewed: 1,3–8,17–24. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Systematically compare SFT / PPO / GRPO / DPO on vision, semantic and execution distribution shifts for OpenVLA manipulation. Findings are transfer hypotheses for VLN, not navigation benchmark results.

## Method, training, inference, and equations

Section 3 uses OpenVLA: SigLIP and DINOv2 image features feed Llama2-7B, which predicts a seven-dimensional end-effector/gripper action. Each action coordinate is discretized into256 bins between training quantiles. The base setting observes one 640×480 image and an instruction. All compared fine-tuning methods use rank 32 LoRA; the value head is fully trained.

Appendix A.1, Eqs 1–2 defines PPO with scalar ratio r_t=πθ(a_t|s_t)/πold(a_t|s_t), where π(a_t|s_t) is the product of the seven action-token probabilities. The maximized surrogate is E[min(r_t A_t, clip(r_t,1−ε,1+ε)A_t)]. GAE estimates A_t by summing discounted temporal-difference residuals, Σ_l(γλ)^l[r_(t+l)+γV(s_(t+l+1))−V(s_(t+l))]. These are policy-gradient estimators, not a convergence guarantee. The three-layer shared critic reads the first action-token hidden state. GRPO instead uses 32 groups of 8 trajectories and group-normalized terminal outcomes; TPO/DPO compares whole trajectory preferences using the same sparse reward.

The base pick-and-place reward gives partial credit for grasping/holding and a larger terminal placement reward. A 140-demonstration warm start reduces interaction requirements. Motion-planner demonstrations are filtered when translation is below 0.01 and Euler-angle change below 0.06 (Appendix A.2), removing about one third of idle actions. This preprocessing is important when comparing SFT and RL. The shared critic uses 44.4 GB versus 81.3 GB for a separate critic; the main run takes roughly 42 hours on one A 100. Appendix B.5 adds four-action chunks with action-dimension clipping and retuned γ/λ, so it is a separate recipe. Appendix B.6 adds articulated-object opening on a Franka robot, extending the original pick-and-place scope.

## Novelty and closest predecessors

Controlled generalization benchmark and practical PPO recipe rather than a new navigation architecture. Strong counterweight to claims GRPO universally best for embodied policies. Authors'POMDP/nonstationarity explanation is a hypothesis: state evolution does not itself make MDP transition nonstationary.

## Results and independent verdict

Figure 3 compares PPO, GRPO and DPO under the authors' setup; it supports PPO as a strong baseline, not a universal ordering of optimizers. Appendix Table 1 reports task success: in-distribution SFT0.781 versus RL0.938; unseen objects0.453 versus 0.714; changed robot pose0.339 versus 0.797; mid-episode object repositioning0.286 versus 0.745. Because base success differs, relative degradation as well as absolute SR matters. Figure 7 reports similar relative visual robustness and stronger semantic/execution transfer.

The extended tests qualify the conclusion. With action chunks, Table 3 shows a regression on unseen multiple receptacles (0.365 SFT versus 0.297 RL). On articulated objects, Table 4 reports visual-average SR 0.646 versus 0.591 and rephrased-instruction SR 0.703 versus 0.620; RL improves execution-average SR 0.300→0.398. Thus RL does not uniformly improve every semantic or visual shift. Thirty preliminary real trials in Table 2 yield pick-and-place SR 0 versus 0.27, which is promising but small. These manipulation findings motivate matched tests in VLN; they are not navigation results.

## Implementation audit

INSPECTED CODE. SimplerEnv/simpler_env/policies/openvla/openvla_train.py:49–75 inserts LoRA and exposes the trainable value head;104–109 creates separate AdamW optimizers. Its action evaluation returns scalar values/log probabilities, and 273–319 implements clipped PPO, clipped Huber value loss and entropy regularization, followed by shared gradient clipping and both optimizer steps. The total minimized loss is policy_loss + value_loss − entropy_coefficient × entropy; accumulation divides this loss before backward.

openvla/prismatic/extern/hf/modeling_prismatic.py:627–642 selects the value-head mode. In the a 0 path at 690–716, the pre-action hidden state gives a B×1 value, action logits have shape B×7×256, and seven token log probabilities are summed to form the joint action log probability. This matches Appendix A.1's product-of-token-probabilities convention. The fetched eval_ms3_collect.py:23–44,62–78,90–128 is an Octo data-collection utility, not an RL4VLA performance evaluator; it is explicitly not treated as validation of the paper's PPO evaluation. Full main training orchestration, all reward/task implementations and the exact final evaluator remain outside this targeted source audit. No project code was executed.

Repository: https://github.com/gen-robot/RL4VLA. Inspected snapshot commit: `de46e0174e7ac310255a2836718fcefab28b8963`. Retrieval status: Commit-pinned source retrieved through sparse git checkout. Project code was not executed.

- [SimplerEnv/simpler_env/policies/openvla/openvla_train.py:49-75,104-109,152-163,197-209,273-319](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/policies/openvla/openvla_train.py)
- [openvla/prismatic/extern/hf/modeling_prismatic.py:627-642,690-716,824-835](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/openvla/prismatic/extern/hf/modeling_prismatic.py)
- [SimplerEnv/simpler_env/eval_ms3_collect.py:23-44,62-78,90-128](https://github.com/gen-robot/RL4VLA/blob/de46e0174e7ac310255a2836718fcefab28b8963/SimplerEnv/simpler_env/eval_ms3_collect.py)

## Bottlenecks and follow-up chain

- **RL4VLA-1: RL's execution robustness does not imply visual or navigation semantic robustness** Conditions: Appearance change, route-language following, partial history Cause: On-policy interaction learns recovery but reward does not guarantee perceptual invariance or instruction faithfulness Evidence: Fig 1 and§4 PPO vs GRPO setting; single-task single-image formulation Type: Methodological / protocol. Follow-ups:  NavGRPO studies graph VLN with different reward; controlled matched optimizer/reward perturbation study needed. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

Main method, Appendix A and additional results in Appendix B were read. The targeted actor/critic/PPO implementation was inspected at the pinned commit, but the full released task suite and evaluation orchestration were not reproduced. All reported benchmark/robot numbers are authors' experiments. No project code was executed.

