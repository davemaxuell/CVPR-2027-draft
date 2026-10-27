# AwareVLN

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation**. Wenxuan Guo, Xiuwei Xu, Yichen Liu, Xiangyu Li, Hang Yin, Huangxing Chen, Wenzhao Zheng, Jianjiang Feng, Jie Zhou, Jiwen Lu CVPR 2026 (2026).

Primary record: https://openaccess.thecvf.com/content/CVPR2026/html/Guo_AwareVLN_Reasoning_with_Self-awareness_for_Vision-Language_Navigation_CVPR_2026_paper.html



Supplied main file: [AwareVLN.pdf](../../Paper/AwareVLN.pdf). Pages actually reviewed: 1–7,12–15; alternate version4–7. Associated local versions are recorded below with SHA256 identifiers. Exact duplicate bytes were not reread. Distinct alternatives were checked for the relevant method/results; their complete bibliographies were not separately reread.

Associated material:

- AwareVLN (alternate).pdf; reviewed pages/status: 4–7; SHA256 `ef14b2f1014e871dd2907f330d288d9a4541766406c27d5e267b7c7ccaf46473`

## Problem, motivation, and operating assumptions

A monocular RGB navigation policy should recognize subtask completion, deviations, and terminal mistakes while retaining low-latency actions. At inference no depth or pose is used; training data generation uses simulator privileged semantics and reference waypoints.

## Method, training, inference, and equations

PAPER METHOD (§3, pp.3–6): one VLM predicts either [REASON] or [ACT]. Inputs include the instruction, eight sampled RGB frames, last reasoning text, and elapsed steps since that reasoning. Reasoning has three ordered fields: scene description, progress assessment, next plan; generated text is reused as future context. Action mode emits parsed movement chunks. Eq.3 chooses the larger of two mode logits. An automatic engine collects expert trajectories and DAgger -like rollouts corrected to reference waypoints. Room-category transitions and completed recovery define subtask events; reference-path distance defines deviations and stopping errors. Qwen -VL-Max labels reasoning with full-episode context, privileged room/progress information and, for mistakes, future correction observations. SFT follows NaVILA / VILA pretraining plus generated reasoning and human videos. Four nodes of H20 GPUs are reported; inference is about 1 FPS on 4090. The main paper does not specify a separate RL objective.

## Novelty and closest predecessors

Unified special-token mode selection plus event-specific structured data extends direct-action NaVILA and fixed-interval Nav-R1. The key contribution is supervision construction and sparse scheduling, not proof of introspective self-awareness. Tables 3–4 isolate event categories and scheduling/token choices.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.6 reports R2R 65.4 SR/55.1 SPL and RxR 67.6/56.1. Table 3 p.7: removing subtask events yields 52.3 R2R SR; removing deviations 55.1; removing stopping-error examples 60.0. However the latter gives 57.5 SPL , ABOVE the full model 55.1, contradicting the prose's 'best across all metrics' claim. Table 4 dense reasoning 63.8 SR versus sparse 65.4 supports sparse control, but latency distributions and equal-generation-budget controls are absent. Real-world 18 instructions across three scenes are useful demonstrations with small cell counts, not broad reliability evidence.

Version-specific caution: the main PDF's supplementary Section B (p.12) says training exclusively on RxR followed by RxR evaluation, while Table 5 (p.13) is captioned as evaluation without RxR training. The latter reports 39.8 SR/36.0 SPL. These statements conflict, so that transfer number is not treated as a settled protocol. The11-page proceedings alternate's method and Tables 1,3,4 were checked and retain the main65.4/55.1 R2R and 67.6/56.1 RxR results, including the ablation where removing stopping-error data improves R2R SPL to 57.5.

## Implementation audit

INSPECTED CODE. evaluation/vlnce_baselines/awarevln_trainer.py:31–50 strips reasoning markers and pads/samples RGB history;81–99 loads dedicated REASON/ACT token IDs and four action IDs. At 193–267 the prompt includes the previous reasoning and its age, then greedily generates at most256 tokens. A REASON result updates context and loops without moving; more than three consecutive reasoning calls forces STOP at 172–179. This establishes an actual repeated inference loop and a hard fallback beyond the conceptual event selector.

Training token registration is in llava/train/train.py:631–653. scripts/train/sft_8frames.sh:69–106 uses eight frames, freezes the vision tower/projector, tunes the language model, and defaults to 24 GPUs, one epoch, batch 8 per GPU with accumulation4 and lr 5e-5. llava/model/loss.py:11–48 implements a generic Gaussian soft-token CE utility; its presence alone does not establish that the AwareVLN event-token path calls it. The event-supervision construction and exact final model loss call require the dataset/model-specific path, and are not claimed verified from this utility. Habitat evaluation is wired in the trainer; no environment or checkpoint was executed.

Repository: https://github.com/GWxuan/AwareVLN. Inspected snapshot commit: `8f14eb8450cdd14ff4e3ba1ea04824255121e6be`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [evaluation/vlnce_baselines/awarevln_trainer.py:31-50,81-99,160-267,269-353](https://github.com/GWxuan/AwareVLN/blob/8f14eb8450cdd14ff4e3ba1ea04824255121e6be/evaluation/vlnce_baselines/awarevln_trainer.py)
- [scripts/train/sft_8frames.sh:1-106](https://github.com/GWxuan/AwareVLN/blob/8f14eb8450cdd14ff4e3ba1ea04824255121e6be/scripts/train/sft_8frames.sh)
- [llava/train/train.py:631-653](https://github.com/GWxuan/AwareVLN/blob/8f14eb8450cdd14ff4e3ba1ea04824255121e6be/llava/train/train.py)
- [llava/model/loss.py:1-48](https://github.com/GWxuan/AwareVLN/blob/8f14eb8450cdd14ff4e3ba1ea04824255121e6be/llava/model/loss.py)

## Bottlenecks and follow-up chain

- **03-awarevln-progress: Learned progress and trigger labels depend on privileged, hindsight-derived supervision** Conditions: Subtasks without room transitions, novel rooms, off-route states absent from correction collection Cause: Room change is an imperfect proxy for semantic completion; future correction observations help the annotator beyond inference information Evidence: §3.3 pp.4–5; deduction on train/test information asymmetry Type: methodological. Follow-ups:  Progress-Think uses learned monotone semantic alignment instead; AdaNav uses uncertainty gating. Neither alone proves event calibration.. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

The main paper, embedded supplementary annotation/cross-dataset discussion (pp.12–15), alternate proceedings method/results and targeted training/inference code were reviewed. The contradictory RxR training description is preserved as uncertainty. Dataset generation, exact loss masking and all annotation examples were not independently reproduced. No benchmark was executed.

- https://horizonrobotics.github.io/robot_lab/progress-think
- https://arxiv.org/abs/2509.24387
