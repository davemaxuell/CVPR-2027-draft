# ECoT

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Robotic Control via Embodied Chain-of-Thought Reasoning**. Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, Sergey Levine CoRL 2024; proceedings published as PMLR270 in 2025 (2025).

Primary record: https://proceedings.mlr.press/v270/zawalski25a.html

The official PMLR270 index identifies the8th Conference on Robot Learning, held6–9 November 2024, with proceedings published12 January 2025. The supplied text is the March 2025 arXiv revision; proceedings byte equivalence was not assumed.

Supplied main file: [ECoT.pdf](../../Paper/ECoT.pdf). Pages actually reviewed: 1, 3-10. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Ground language reasoning in physical manipulation rather than only high-level subtask prose. Uses a third-person image and instruction to control a WidowX 6 DoF arm; this is adjacent VLA prior art, not demonstrated R2R / RxR navigation.

## Method, training, inference, and equations

PAPER METHOD (§4): OpenVLA 's fused SigLIP / DINOv2 visual encoder and Llama2 -7 B predict a sequence containing task paraphrase, plan, current subtask, movement reasoning, gripper pixel location, object boxes, then seven discretized action tokens (256 bins per dimension). Post-hoc annotations for 2.5 M Bridge V2 transitions use Prismatic captions, GroundingDINO boxes, proprioception-derived motion primitives, OWLv2/SAM gripper detections and per-trajectory RANSAC projection; Gemini 1.0 builds reasoning chains using those signals. Standard causal next-token supervision trains reasoning and actions jointly. Inference caches high-level reasoning for five steps or runs a second asynchronous reasoning instance; token generation grows from 7 to about 350 per step. No formal faithfulness guarantee is claimed.

## Novelty and closest predecessors

Unlike purely linguistic CoT/Naive-CoT, supervise explicit low-level spatial grounding as intermediate text. Controls share OpenVLA base and Bridge V2 robot trajectories, making grounding ablations meaningful; synthetic annotation and test-time compute still differ.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.7 aggregates 314 trials per method: ID-view ECoT 66% versus OpenVLA 44%, OOD-view 64% versus 30%; task-specific failures remain (tall cup). Table 2 p.9's 25-trial efficiency subset reports five-step caching 72% success/+24%speed and asynchronous 65%/+40%speed, with asynchronous doubling policy instances. Text correction improves a selected hard subset 32%→80%, not full-benchmark accuracy. §5.6 p.10 explicitly says reasoning transfer to other embodiments was largely visualization/proxy evidence; SIMPLER Google-Robot rollouts did NOT improve over OpenVLA . Hence language transfer is not validated policy transfer.

## Implementation audit

INSPECTED CODE. prismatic/vla/datasets/datasets.py:32–48 controls reasoning dropout. Its RLDS transform at 69–108 combines the image, instruction, annotated reasoning and tokenized action; labels are copied from input_ids, with optional EOS masking. This inspected transform does not independently mask the instruction prompt. Lines 137–151 specify a one-frame window, no depth/proprioception and action normalization by quantiles. vla-scripts/finetune.py:167–169,233–247,295–297 runs AdamW, the model loss, backward and optimizer steps. It is a released fine-tuning path, not verification of the entire original pretraining run.

openvla.py:94–111 builds the task prompt;128–156 and 188–205 implement reuse of asynchronously generated reasoning;215–229 converts seven discrete action tokens into normalized continuous controls. The reasoning server at experiments/bridge/reasoning_server.py:65–81 loads the model and serves raw generation over ZMQ. The Bridge evaluator at 84–87 defaults to a 2 Hz control loop,600 steps and 500 episodes;139–165 times action calls. These files substantiate the reasoning/action interface and deployment mechanism. Complete original reasoning-label generation, all pretraining configurations and physical robot results were not independently reproduced. No project code was executed.

Repository: https://github.com/MichalZawalski/embodied-CoT. Inspected snapshot commit: `1813ad76001f1e08095088f94a86c43fc0e457a3`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [prismatic/vla/datasets/datasets.py:32-48,69-108,137-151](https://github.com/MichalZawalski/embodied-CoT/blob/1813ad76001f1e08095088f94a86c43fc0e457a3/prismatic/vla/datasets/datasets.py)
- [vla-scripts/finetune.py:167-169,233-247,295-297](https://github.com/MichalZawalski/embodied-CoT/blob/1813ad76001f1e08095088f94a86c43fc0e457a3/vla-scripts/finetune.py)
- [prismatic/models/vlas/openvla.py:33-46,94-111,128-156,188-229](https://github.com/MichalZawalski/embodied-CoT/blob/1813ad76001f1e08095088f94a86c43fc0e457a3/prismatic/models/vlas/openvla.py)
- [experiments/bridge/reasoning_server.py:65-81](https://github.com/MichalZawalski/embodied-CoT/blob/1813ad76001f1e08095088f94a86c43fc0e457a3/experiments/bridge/reasoning_server.py)
- [experiments/bridge/eval_model_in_bridge_env.py:84-87,126-175](https://github.com/MichalZawalski/embodied-CoT/blob/1813ad76001f1e08095088f94a86c43fc0e457a3/experiments/bridge/eval_model_in_bridge_env.py)

## Bottlenecks and follow-up chain

- **03-ecot-faithfulness: Correct-looking reasoning may not control the action and becomes stale between updates** Conditions: Unseen objects/embodiments and fast-changing scenes Cause: Autoregressive language intermediates are not verified physical state; fixed-step caching trades freshness for latency Evidence: §5.3 caveat p.7; §5.6 and limitations p.10 Type: methodological. Follow-ups:  Aux-Think removes test-time text; AdaNav / AwareVLN trigger reasoning adaptively; RL4VLA tests generalization axes. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

No robot experiments executed. Appendix training specifics and all 24 pages not fully audited. Do not generalize manipulation evidence into navigation performance.

