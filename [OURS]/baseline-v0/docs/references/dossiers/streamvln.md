# StreamVLN

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling**. Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, Xihui Liu, Jiangmiao Pang ICRA 2026 according to the official project/repository; reviewed arXiv v2 from July 2026 (2026).

Primary record: https://arxiv.org/abs/2507.05240



Supplied main file: [StreamVLN.pdf](../../Paper/StreamVLN.pdf). Pages actually reviewed: 1,3–7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Repeated history pre fill makes video VLA too slow; continuous stream requires bounded recent context and retained history. Distinguish RGB -only core from optional depth/odometry voxel pruning.

## Method, training, inference, and equations

SectionIII organizes observations and predicted actions as an interleaved conversation. KV states are reused within an eight-turn window. At a boundary, the model discards the old prompt/action context, samples eight visual memory frames and starts a new window from those features. This lowers repeated prefilling cost while retaining a limited history. The policy predicts four actions per generation; eight turns therefore need not mean eight simulator steps.

An optional geometric variant (Algorithm 1) back-projects image patches using depth and poses into3D voxels. It retains the latest patch token for each stride-bin/voxel pair and can drop frames retaining too few patches. This is deterministic geometric selection, not a learned world model or a guarantee that the remaining patches preserve task-relevant detail. The RGB-only deployed agent must be distinguished from this depth/odometry variant.

The LLaVA-Video-7B/Qwen2 backbone receives two supervised fine-tuning stages: oracle navigation first, then DAgger and general vision-language data. Reported learning rates are2e-5 for language and 5e-6 for vision, with about 1,500 A 100 GPU-hours. The mixture includes450K MP3D navigation examples,300K ScaleVLN examples from 700 HM3D scenes,240K DAgger examples,248K video/ScanQA examples and 230K MMC4 examples. These data additions explain part of the headline gain. At inference, streaming cache reuse controls repeated computation, while window resets and finite memory introduce latency spikes and potential forgetting.

## Novelty and closest predecessors

Streaming dialogue KV reuse plus bounded visual memory consolidation, and 3 D latest patch selection. NaVid / Uni-NaVid compress features but repeatedly refresh LLM context; StreamVLN changes causal prompt organization. Not all history information can be preserved under fixed 8 frame budget.

## Results and independent verdict

Reviewed v 2 Table I extra-data RGB 56.4 SR/50.2 SPL R2R,54.4/45.4 RxR ; Prune Mem 57.4/51.1 and 53.9/45.1. Main text claims 56.9/51.9 and 52.9/46.0 which do not match Table I; cite version+table. Table IV 2 memory frames 37.3 SR→8 frames 45.5, all history 40.0. Data control Table III 45.6 oracle→50.8 DAgger +Vid QA→52.8 MMC4 →56.4 Scale; architectural gain not isolated by full number. Table III prose increment values also differ literal rows. Real 20 trials/scenario. Timing low mean and window boundary spikes must be both reported.

## Implementation audit

INSPECTED CODE. streamvln/streamvln_agent.py:169–174 substitutes zero depth and identity pose because this deployed variant does not use voxel pooling. Lines 191–198 clear the conversation/KV cache at the configured window boundary;223–232 reintroduce sampled historical image features;244–253 pass and update past_key_values. The agent defaults to 32 simulator steps per window and four predicted future actions, consistent with eight action-generating turns rather than eight raw simulator steps. stream_video_vln.py:111–142 pools historical visual features into memory tokens.

streamvln_train.py:1743–1766 explicitly selects trainable multimodal components and 1816 constructs supervised data;1852–1855 runs the trainer. This verifies training wiring and streaming inference, but the precise loss masking, full dataset recipe and optional geometric voxel-pruning implementation were not all independently traced in this targeted pass. The paper's optional RGB-D/odometry variant must be distinguished from the RGB-only agent above. No training, checkpoint inference or benchmark executed.

Repository: https://github.com/InternRobotics/StreamVLN. Inspected snapshot commit: `e48f6ff7e9201d93aae003e8f64b04c00cec13bc`. Retrieval status: Commit-pinned source retrieved through sparse git checkout. Project code was not executed.

- [streamvln/streamvln_agent.py:169-258,265-290](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/streamvln/streamvln_agent.py)
- [streamvln/model/stream_video_vln.py:111-152](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/streamvln/model/stream_video_vln.py)
- [streamvln/streamvln_train.py:1743-1766,1816,1852-1855](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/streamvln/streamvln_train.py)

## Bottlenecks and follow-up chain

- **STREAM-1: Fixed memory and geometric pruning can erase decision-critical appearance or earlier actions** Conditions: Revisits, subtle objects, dynamic changes, pose/depth noise Cause: Recent token per voxel assumes redundancy; old action states discarded; uniform 8 frames not instruction aware Evidence: Algorithm 1; Table IV memory size;§III-B Type: Methodological / protocol. Follow-ups:  PolarMem /HAM retrieve explicit context; NavFoM BATS recency weight; no faithful bounded memory guarantee. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

The paper's metric-version inconsistencies are recorded. Targeted streaming inference and training wiring were inspected, but complete loss masking, data generation and optional voxel pruning were not exhaustively audited. No independent runtime or benchmark was measured.

