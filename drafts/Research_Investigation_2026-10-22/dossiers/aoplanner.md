# AO-Planner

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Affordances-Oriented Planning Using Foundation Models for Continuous Vision-Language Navigation**. Jiaqi Chen, Bingqian Lin, Xinmin Liu, Lin Ma, Xiaodan Liang, Kwan-Yee K. Wong AAAI 2025 (2025).

Primary record: https://ojs.aaai.org/index.php/AAAI/article/download/34526/36681



Supplied main file: [AO-Planner.pdf](../../Paper/AO-Planner.pdf). Pages actually reviewed: 1, 3-7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Replace graph-provided candidate waypoints with foundation-model low-level planning in continuous environments. Assumes four directional RGB views, depth, calibrated camera intrinsics, and a rotate-then-forward controller. Evaluates R2R-CE and sampled English RxR-CE , not a monocular RGB -only setting.

## Method, training, inference, and equations

PAPER METHOD (pp.3–5): GroundingDINO plus SAM segments navigable ground in each 90° view. Uniformly sampled pixels on the floor mask are labeled with IDs. Gemini-1.5-Pro receives the instruction, annotated view, and waypoint/path definitions; it returns candidate waypoint IDs and connecting image-plane paths. A separate GPT-4o Path Agent sees all four images with candidate paths and uses history/instruction to select one. Depth and camera intrinsics back-project selected pixels to 3 D; ETPNav -style rotation/forward control executes the piecewise path. Eq.1 specifies low-level generation; Eq.2 specifies high-level selection. No navigation loss is optimized in the zero-shot variant.

A distinct trained variant collects approximately 10 K locations/40 K images and 40 K Gemini calls as pseudo waypoint labels, trains the ETPNav waypoint predictor, and combines it with an ETPNav high-level policy. This is waypoint distillation, not a training-free system. The paper refers optimizer details to its arXiv appendix; conference text alone does not specify them.

## Novelty and closest predecessors

Visual affordance prompting and explicit RGB -space paths extend MapGPT 's high-level reasoning into waypoint proposal and continuous movement. Table 2 removes floor masks, instructions, and waypoint definitions; Table 3 crosses trained/zero-shot waypoint generators with baseline/Path Agent. These are stronger mechanism controls than comparing only final benchmark scores.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.6 separates zero-shot R2R SR 25.5/ SPL 16.6 and RxR 22.4/15.1 from distilled R2R^47/33 and RxR^43.3/30.5. Treating 47 SR as the training-free score would be incorrect. RxR uses 500 sampled episodes and ablations use 100 sampled R2R episodes (§experiments p.5). Table 2: removing affordances lowers subset SR 27 to 15; Table 4: Gemini proposer+ GPT-4o decider reaches 27 versus same-Gemini 14 and same- GPT-4o 16. Strong modular evidence, but API capability and sensing differ across baselines; ground segmentation alone does not guarantee path feasibility.

## Implementation audit

INSPECTED CODE. vlnce_baselines/zero_shot_agent.py:690–825 obtains current camera position/heading, renders four directional views, calls llm_waypoint_predictor_single_view for each, builds bidirectional path-graph edges and submits the resulting graph prompt to GPT-4o. Failed JSON parsing falls back to STOP. The code therefore contains a concrete perception-to-path-to-decision pipeline, rather than only an LLM wrapper. Policy_ViewSelection_ETP.py:150–214 exposes the inherited 120-angle by 12-distance waypoint heatmap and NMS; its llm_waypoint branch aligns the external proposals with visual features. Evaluation in zero_shot_agent.py:926–939 calculates final-distance success, oracle success, path length, SPL and DTW.

The run_r2r/zero_shot.bash example enables sliding, sets max_traj_len=5 and uses gpt-4o-2024-05-13. fast_eval selects every fifth episode at zero_shot_agent.py:383–386. These flags must be matched when comparing subsets and physical feasibility. This audit directly covers the zero-shot inference/evaluation path and inherited waypoint representation. It does not independently reconstruct the separate waypoint-distillation training run or establish that the released example is the exact main-table configuration. No project code was executed.

Additional preprocessing inspection: llm/grounded_sam_Gemini.py:313–316,430–437 runs GroundingDINO and SAM; llm/utils.py:22–69,87–142 converts normalized depth and image-space waypoints into world coordinates using camera intrinsics and poses. llm/prompting/Gemini.py:14 selects gemini-1.5-pro-latest, an unpinned service alias. These dependencies constrain reproducibility even when navigation itself is training-free. Distillation-training provenance was not independently reconstructed.

Repository: https://github.com/chen-judge/AO-Planner. Inspected snapshot commit: `719f42a1c9bbb3ebee4815287713ec6c73c0e468`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [vlnce_baselines/zero_shot_agent.py:77-106,255-305,383-386,579-633,690-825,926-939](https://github.com/chen-judge/AO-Planner/blob/719f42a1c9bbb3ebee4815287713ec6c73c0e468/vlnce_baselines/zero_shot_agent.py)
- [vlnce_baselines/models/Policy_ViewSelection_ETP.py:109-214,317-340](https://github.com/chen-judge/AO-Planner/blob/719f42a1c9bbb3ebee4815287713ec6c73c0e468/vlnce_baselines/models/Policy_ViewSelection_ETP.py)
- [run_r2r/zero_shot.bash:1-24](https://github.com/chen-judge/AO-Planner/blob/719f42a1c9bbb3ebee4815287713ec6c73c0e468/run_r2r/zero_shot.bash)
- [llm/grounded_sam_Gemini.py:313-316,430-437](https://github.com/chen-judge/AO-Planner/blob/719f42a1c9bbb3ebee4815287713ec6c73c0e468/llm/grounded_sam_Gemini.py)
- [llm/utils.py:22-69,87-142](https://github.com/chen-judge/AO-Planner/blob/719f42a1c9bbb3ebee4815287713ec6c73c0e468/llm/utils.py)
- [llm/prompting/Gemini.py:14,43-47](https://github.com/chen-judge/AO-Planner/blob/719f42a1c9bbb3ebee4815287713ec6c73c0e468/llm/prompting/Gemini.py)

## Bottlenecks and follow-up chain

- **03-aoplanner-motion: A plausible floor-projected path may still be unexecutable or inefficient** Conditions: Occluded obstacles, poor depth, narrow traversable regions, simulator reconstruction artifacts Cause: 2 D mask and language path selection do not prove collision-free 3 D dynamics Evidence: Method pp.4–5; author discussion p.5 of image quality and remaining supervised gap; deduction on feasibility Type: methodological. Follow-ups:  CA-Nav adds persistent semantic/value maps; GC-VLN adds graph constraints and backtracking; NaVILA learns a physical locomotion controller.. Status: **Shifted into another cost or limitation**, confidence high.

## Source/access record and remaining uncertainty

Conference PDF reviewed; arXiv implementation appendix remains an explicit access-scope limitation. No API calls or navigation benchmark executed.

- https://arxiv.org/abs/2412.10137
- https://proceedings.mlr.press/v305/yin25a.html
