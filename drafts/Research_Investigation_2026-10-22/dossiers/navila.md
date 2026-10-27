# NaVILA

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**NaVILA: Legged Robot Vision-Language-Action Model for Navigation**. An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, Xiaolong Wang Robotics: Science and Systems 2025 (2025).

Primary record: https://roboticsproceedings.org/rss21/p018.html



Supplied main file: [NaVILA.pdf](../../Paper/NaVILA.pdf). Pages actually reviewed: Alternate arXiv v2 pages1,3–8 (local image-only proceedings PDF not text-reviewed). Local supplied PDF has no extractable text. Reviewed alternate official-author arXiv2412.04453v2 (17Feb2025), downloaded to evidence/03/navila_alternate.pdf and text. Numerical claims here refer to that version; equivalence to supplied proceedings bytes is not assumed.

Associated material:

- evidence/03/navila_alternate.pdf; reviewed pages/status: 1,3–8; version arXiv2412.04453v2

## Problem, motivation, and operating assumptions

Connects semantic VLN planning to legged robot joint control. Separates RGB VLA high level commands from LiDAR /proprioceptive PPO locomotion. Therefore high-level RGB -only benchmark description must not be applied to the full real robot stack.

## Method, training, inference, and equations

§II-A VILA image- VLM encodes current frame plus uniform history(always retain first);196 tokens/frame,8–64 frames, textual roles rather than special history tokens. All vision/projector/ LLM weights SFT one epoch from stage 2 VILA . Data blend: R2R / RxR oracle, EnvDrop augmented instructions, trajectory summarization, ScanQA , general VQA , and 20 K trajectories from 2 K YouTube tours. MASt3R metric pose creates continuous actions; VLM captions+ LLM rephrases instructions. Consecutive same actions merge≤3 and Stop is rebalanced. Regex outputs distance/angle. §II-B converts forward to.5 m/s and turn to±π/6 rad/s for command duration. PPO actor predicts 12 desired joint positions; PD controller converts motor torque. Actor uses proprio history and 2.5 D LiDAR height map; critic gets privileged linear velocity/terrain. Lowest voxel height plus max filter over 5 clouds build map. One-stage RL trained in IsaacLab ; exact reward weights require appendix/legged-loco repo. Full system requires LiDAR and controller dynamics, not just VLM .

## Novelty and closest predecessors

Two-level language action interface joins general VLA with perception-conditioned legged locomotion; human video action labeling and VLN-CE - Isaac benchmark are major contributions. Direct joint prediction was not used. New benchmark physical embodiment controls are as important as VLM architecture.

## Results and independent verdict

Table I R2R 54 SR/49 SPL , RxR^49.3/44 trained both; Table II R2R -only RxR 34.3/28.2 must not be confused with 49.3. Isaac 1077 selected traversable mesh trajectories: Go 2 blind 36.2 SR→ LiDAR 50.2, oracle 51.3; H 1 blind 24.4→45.3. Prose claims 15/27 point oracle-gap but those are blind policy gaps, not vision policy gaps(1.1/6.0); Table IV clarifies this. Table V collision.81 versus ROA 3.09. Real 25 instructions×3 trials, human video data helps outdoor strongly; platform transfer uses new low level control but shared VLA . Small selected real world test not general safety guarantee.

## Implementation audit

INSPECTED CODE. evaluation/vlnce_baselines/navila_trainer.py:38–52 pads short episodes with black frames and samples historical images plus the current image. Lines 163–211 format the instruction and greedily generate a language action with a 32-token cap. Lines 221–234 parse stop/forward/left/right by regular expressions before conversion to simulator actions; Habitat episode metrics are subsequently recorded. This is a high-level VILA navigation implementation. It is not the complete quadruped locomotion system. The low-level controller has a separate linked legged-loco release; the high-level RGB-only input does not imply an RGB-only physical robot. No checkpoint, simulator or robot code was executed.

The high-level scripts/train/sft_8frames.sh:10–41 specifies eight frames, SigLIP, all three VLM components trainable, one epoch and lr 1e-4, with a mixture including R2R, RxR, EnvDrop, human videos and general QA. The linked yang-zj1026/legged-loco commit87b0d3d18404e784abc0a62227bc41c940f29ecc was also inspected. go2_low_vision_cfg.py:301–324 explicitly consumes angular velocity, orientation, commands, joint positions/velocities, previous actions and a LiDAR-derived height map. Its reward block205–260 includes termination, joint deviation, base-height, action-smoothness, power, foot-sliding and collision penalties. actor_critic_depth_cnn.py:13–60 separately embeds proprioception and depth, concatenates two 128-dimensional latents and predicts joint commands;84–166 adds a Gaussian policy and scalar critic. rsl_rl/algorithms/ppo.py:151–175 implements the clipped surrogate/value/entropy update. This checks the sensing and learning mechanism, not the physical calibration or exact paper checkpoint.

Pinned related sources: [robot observations and rewards](https://github.com/yang-zj1026/legged-loco/blob/87b0d3d18404e784abc0a62227bc41c940f29ecc/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/config/go2/go2_low_vision_cfg.py), [actor/critic](https://github.com/yang-zj1026/legged-loco/blob/87b0d3d18404e784abc0a62227bc41c940f29ecc/rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py), [PPO](https://github.com/yang-zj1026/legged-loco/blob/87b0d3d18404e784abc0a62227bc41c940f29ecc/rsl_rl/rsl_rl/algorithms/ppo.py).

Repository: https://github.com/AnjieCheng/NaVILA. Inspected snapshot commit: `76b98f233dd0fff05dfcd69435eec6740febff9d`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [evaluation/vlnce_baselines/navila_trainer.py:38-52,163-234,313-330](https://github.com/AnjieCheng/NaVILA/blob/76b98f233dd0fff05dfcd69435eec6740febff9d/evaluation/vlnce_baselines/navila_trainer.py)
- [scripts/train/sft_8frames.sh:10-41](https://github.com/AnjieCheng/NaVILA/blob/76b98f233dd0fff05dfcd69435eec6740febff9d/scripts/train/sft_8frames.sh)

## Bottlenecks and follow-up chain

- **NAVILA-1: Semantic command can be physically infeasible despite robust local avoidance** Conditions: Narrow doors, embodimentshape, blocked route, camera height changes Cause: High level VLA does not explicitly receive body geometry or locomotion feasibility Evidence: Isaac Table IV Go 2 vs H 1 difference and traversable subset selection Type: Methodological / evaluation. Follow-ups:  NavFoM unifies embodiments but still task scaling/controllers; CorrectNav acknowledges physical body collisions. Status: **Partially addressed**, confidence Moderate.

## Source/access record and remaining uncertainty

The local image-only PDF was replaced for text review by author arXiv2412.04453v2; later proceedings changes are not assumed reviewed. High-level inference/SFT configuration and linked low-level observations/rewards/PPO were inspected. Physical calibration, all training assets and robot results were not reproduced. No Isaac or robot experiment was executed.

