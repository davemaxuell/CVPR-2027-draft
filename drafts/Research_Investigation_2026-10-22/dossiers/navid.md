# NaVid

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation**. Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, He Wang Robotics: Science and Systems 2024 (2024).

Primary record: https://roboticsproceedings.org/rss20/p079.pdf



Supplied main file: [NaVid.pdf](../../Paper/NaVid.pdf). Pages actually reviewed: 1, 3–7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Transfer instruction-following from simulator to real monocular RGB navigation without depth, map or odometer policy inputs. Unlike graph agents, outputs parameterized move/turn commands; lower robot controller still must execute them.

## Method, training, inference, and equations

§IV Eqs 1–3: LLaMA-VID/ EVA-CLIP encodes frame X_t∈R^(256×C). Q-Former / BERT instruction query Q∈R^(M×C) cross attends image, mean queries, and projects one instruction-queried token. Parallel grid pool/project gives 64 instruction-agnostic current tokens and 4 per history frame. <HIS>,<OBS>,<NAV>distinguish history/current/instruction before Vicuna 7 B. Autoregressive language outputs Forward(distance), Left(angle), Right(angle), Stop; regex parser converts commands. Training 320 K oracle R2R steps+180 K DAgger non oracle+10 K trajectory-to-instruction reasoning+763 K web VQA , total 1.273 M;24 A 100 ×28 h, one epoch, train LLaMA /text encoder while other pretrained modules frozen per§IV-D. No additional test time reasoning loop is required; 'instruction reasoning' is an auxiliary training task. History tokens grow linearly with frames despite per frame compression. 1.2–1.5 s/action reported.

## Novelty and closest predecessors

Task-specific video-token allocation and action-language formulation atop LLaMA-VID, with non oracle collection/auxiliary instruction reconstruction. NaVILA changes backbone/data blend/physical controller; Uni-NaVid adds semantic token merging; StreamVLN segment memory; NavFoM cross camera/cross task trajectory head.

## Results and independent verdict

Table I R2R 37.4 SR/35.9 SPL , comparable SR but higher SPL than WS-MG Map 38.9/34.3. Table II RxR zero-shot from R2R 23.8/21.2. Table III 100 sample baseline LLaMA-VID- Nav 29/27.5 vs NaVid 38/35.4 supports VLN modeling. Fig 4 stable horizon bins does not prove arbitrary length generalization; >90 step train cases 5.91%. Real 200 instructions, four scenes, success 1.5 m versus simulator 3 m. Pretrained web knowledge and training data differ strongly from old RGB -CMA baseline.

## Implementation audit

INSPECTED CODE. navid/model/navid_arch.py:200–225 pools the latest frame to an 8-by 8 grid and history to the configured smaller grid, then projects visual features into the LLM. Lines 233–244 map grid:2 to four history tokens. language_model/llava_navid.py:99–112 applies shifted autoregressive CE to labeled output tokens. train/train.py:1165–1186 exposes backbone freezing, quantization and optional LoRA; these options alone do not identify the exact published training run.

agent_navid.py:54–69 preprocesses newly arrived RGB images and appends them to a cached history tensor. Its generation path at 128–135 samples with temperature 0.2 and at most1024 tokens, rather than being intrinsically deterministic. Reset at 231–238 clears episode history. This confirms video-history action generation and exposes history-memory growth; it does not independently reproduce the paper's complete data mixture, annotation generation or benchmark evaluator. The released file also contains a Uni-NaVid agent, so care is needed to select the intended class. No checkpoint, training or navigation run was executed.

Repository: https://github.com/jzhzhang/NaVid-VLN-CE. Inspected snapshot commit: `ce2f804a43fb23f1d52431ec24be1acf74c3f4f4`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [navid/model/navid_arch.py:126-196,200-254](https://github.com/jzhzhang/NaVid-VLN-CE/blob/ce2f804a43fb23f1d52431ec24be1acf74c3f4f4/navid/model/navid_arch.py)
- [navid/model/language_model/llava_navid.py:55-120](https://github.com/jzhzhang/NaVid-VLN-CE/blob/ce2f804a43fb23f1d52431ec24be1acf74c3f4f4/navid/model/language_model/llava_navid.py)
- [navid/train/train.py:1165-1186](https://github.com/jzhzhang/NaVid-VLN-CE/blob/ce2f804a43fb23f1d52431ec24be1acf74c3f4f4/navid/train/train.py)
- [agent_navid.py:54-69,128-141,231-238](https://github.com/jzhzhang/NaVid-VLN-CE/blob/ce2f804a43fb23f1d52431ec24be1acf74c3f4f4/agent_navid.py)

## Bottlenecks and follow-up chain

- **NAVID-1: History cost grows and compressed tokens can miss spatial details** Conditions: Long episodes, high precision turn/stop decisions Cause: 4 history tokens per frame and autoregressive 7 B decoding Evidence: §IV-B, D; Table III and Fig 4 Type: Methodological / evaluation. Follow-ups:  Uni-NaVid token merging, MapNav ASM, StreamVLN stream memory, NavFoM BATS; each introduces forgetting/sensor/data assumptions. Status: **Partially addressed**, confidence Moderate.

## Source/access record and remaining uncertainty

Main method/results and targeted video-token/model-loss/train/inference code were inspected. Complete annotation generation and exact evaluation assets were not reconstructed. No independent runtime or robot measurement was made.

