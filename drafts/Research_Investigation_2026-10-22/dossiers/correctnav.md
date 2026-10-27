# CorrectNav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model**. Zhuoyuan Yu, Yuxing Long, Zihan Yang, Chengyan Zeng, Hongwei Fan, Jiyao Zhang, Hao Dong AAAI 2026 (2026).

Primary record: https://ojs.aaai.org/index.php/AAAI/article/view/38942



Supplied main file: [CorrectNav.pdf](../../Paper/CorrectNav.pdf). Pages actually reviewed: 1-7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

A monocular navigator trained only on demonstrations lacks recovery behavior once it departs the oracle path. The target is recovering without an extra reasoning model at inference. Simulator reference paths and a planner are available while collecting training data.

## Method, training, inference, and equations

PAPER METHOD (pp. 3–5, Algorithm 1): the navigator connects a SigLIP visual encoder to Qwen2 through a two-layer projector, initialized from LLaVA-Video-7B. Initial fine-tuning combines 2.1 million R2R/RxR step examples with camera and lighting randomization, 30,000 trajectory-to-instruction generation examples, and 240,000 general multimodal examples. These data sources are part of the intervention and must be matched when attributing later gains to recovery training.

The flywheel then runs the current policy on training instructions and identifies a deviation. Reference-path points are interpolated; the first predicted position whose distance to the nearest reference segment exceeds threshold S is treated as the deviation. The procedure projects that position onto a reference segment and plans a correction through the segment's next waypoint and the remaining reference waypoints. This produces a recovery suffix, rather than relabeling the entire trajectory as an expert demonstration. Earlier frames are retained as context, while only the correction suffix contributes supervised action targets.

A second branch targets perception around the error. Qwen-VL-Plus captions three key frames near the deviation and generates visual question-answer supervision. The next training round mixes sampled correction and perception examples with oracle demonstrations. Repeating collection and training changes the state distribution on which the policy receives supervision. The objective remains supervised token prediction; the reviewed method does not establish a reinforcement-learning update. In particular, the recovery target is constructed by a reference-path planner, not discovered from an unconstrained learned reward.

Inference uses the trained navigator directly, with a context of 16 frames and four-action output chunks. The data-generation model and reference-path correction procedure are training-time components. The paper reports eight A100 GPUs, 80 hours for initial training and 20 hours for each flywheel iteration. These are author-reported costs, not local measurements; reproducing the data-generation component also requires its simulator and labeling access.

## Novelty and closest predecessors

The method extends NaVid-style navigation tuning with an iterative, DAgger-like error-data pipeline. Its more specific contributions are localization of the first deviation, supervision restricted to the recovery suffix, and balanced action/perception data. The ablations can support those choices. A broader claim that collecting policy mistakes or learning from failed trajectories is new would overlook established interactive imitation-learning methods.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.5 reports R2R 65.1 SR/62.3 SPL and RxR 69.3/63.3. First-round Table 2 gives 63.0 SR, falling 59.2 without trajectory correction and 60.1 without key frame perception. Figure 5/p.6 states performance drops at the fourth iteration, where training stops; improvement is not monotonic indefinitely. Real-world 120 instructions across three environments are useful transfer evidence but do not isolate correction from domain randomization/backbone/data changes. Limitations p.7 acknowledge body-relative geometry errors, including hind-leg scraping.

## Implementation audit

ACCESS SCOPE: current paper-title/project searches did not locate a verifiable released official model/training repository. No official implementation was inspected or executed. Absence from this search is not proof no repository exists.


## Bottlenecks and follow-up chain

- **03-correctnav-recovery: Nearest-path recovery can mistake geometric closeness for valid instruction progress** Conditions: Looping/repeated corridors, skipped landmarks, alternative valid routes Cause: Projecting to a reference segment assumes preceding reference points were correctly traversed Evidence: Deviation and recovery equations p.4; deduction, not reported failure rate Type: methodological. Follow-ups:  AwareVLN supervises semantic progress; Progress-Think learns progress alignment; NavGRPO explores non expert routes. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

The paper mechanism and data/iteration ablations were reviewed. No official executable implementation was verified in the scoped source search. Data regeneration and benchmark reproduction were not attempted.

- https://arxiv.org/abs/2603.15370
- https://gwxuan.github.io/AwareVLN/
