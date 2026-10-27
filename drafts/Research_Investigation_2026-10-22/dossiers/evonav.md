# EvoNav

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**History to Future: Evolving Agent with Experience and Thought for Zero-shot Vision-and-Language Navigation**. Guangzhao Dai, Shuo Wang, Zihan Wang, Guo-Sen Xie, Yang Yang, Jinshan Pan, Qianru Sun, Xiangbo Shu CVPR 2026 (2026).

Primary record: https://openaccess.thecvf.com/content/CVPR2026/html/Dai_History_to_Future_Evolving_Agent_with_Experience_and_Thought_for_CVPR_2026_paper.html



Supplied main file: [EvoNav.pdf](../../Paper/EvoNav.pdf). Pages actually reviewed: 1, 3-7. Associated local versions are recorded below with SHA256 identifiers. Exact duplicate bytes were not reread. Distinct alternatives were checked for the relevant method/results; their complete bibliographies were not separately reread.

Associated material:

- EvoNav (supplement).pdf; reviewed pages/status: 1–4; SHA256 `716c3c0d20a588f474dbaae8d8507ea28f89e9b583de51f4d24fa3d3a058f901`
- EvoNav (alternate).pdf; reviewed pages/status: Exact SHA256 duplicate; no separate reread; SHA256 `482fa22bc5fee6999276ef6c7040bb6166b4814f42a26a8a1d02062b946e3cc9`

## Problem, motivation, and operating assumptions

Improve continuous-navigation decisions with prior-episode lessons and textual future landmarks. Assumes 12-view RGB -D panoramas and a pretrained waypoint predictor; no prebuilt map, but experience persists across episodes.

## Method, training, inference, and equations

PAPER METHOD (§4): parse instruction into ordered subactions and landmarks. Predict future action, infer completed actions by comparing history and predicted future, then identify the next landmark. This F-CoT is language prediction, not learned dynamics. An external evaluator computes terminal navigation metrics using benchmark ground truth (Eq.7); the supplement specifies that the agent receives its own rollout and terminal SR/SPL, not a ground-truth trajectory. H-CoE then asks an LLM to generate correction advice, reproduce high-quality paths, or improve inefficient successes. Store instruction/landmarks/path/advice in Chroma and retrieve by landmark embedding. Separate banks store textual visual descriptions and CLIP image embeddings across completed steps. Future landmark queries both banks. Final LLM sees retrieved trajectory/scene evidence, current views, progress and instruction. Model weights are not updated; the experience store is.

## Novelty and closest predecessors

Combines future-text guidance with cross-episode trajectory and scene retrieval, beyond Open-Nav 's direct reasoning. The distinctive protocol includes privileged post-episode metric feedback; it is not a fresh-agent one-shot condition. Broad proposals for 'learn from prior failures by retrieval' overlap directly.

## Results and independent verdict

PAPER EXPERIMENT: sampled R2R-CE Table 1 p.6: same Gemini 2.5-Pro Open-Nav 23 SR/19.90 SPL versus EvoNav 43/37.77. NavRAG-CE 24/13.39 versus 30/17.60. Table 4 p.7: future-only 27 SR, adding trajectory experience 31, scene-description 39, both scene forms 40, full 43. These support memory utility conditional on episode ordering/outcome access. Main table includes some cited results using different waypoint predictors. No reset-memory or no-oracle-outcome control is established in reviewed main text. Source says code will be released.

The supplement distinguishes online accumulation from a larger offline bank. Outcome-derived feedback and episode order must be matched against frozen-memory baselines. Its correction text is LLM-generated from the agent's own failed experience; it is not an oracle-corrected demonstration path.

## Implementation audit

ACCESS SCOPE: CVPR main paper states code will be released. Title and project searches did not verify an official implementation. No retrieval bank, feedback computation or evaluation ordering was inspected in code.


## Bottlenecks and follow-up chain

- **03-evonav-feedback: Experience improvement depends on reliable previous-episode outcome labels and retrieval relevance** Conditions: Deployment without reliable terminal outcome feedback; changed scenes or reset-memory evaluation Cause: Advice is generated from privileged navigation metrics and can carry hallucinated correction into later episodes Evidence: §4.3.1 Eq.7–9 p.4 Type: methodological. Follow-ups:  HAM-VLN uses within-episode advisory failure memory; PolarMem adds consistency filtering but not embodied coverage evidence. Status: **Shifted into another cost or limitation**, confidence high.

## Source/access record and remaining uncertainty

Main paper and the separate four-page supplement were read. Supplement A.1 uses 100 sampled episodes for R2R-CE and 100 random NavRAG episodes. Section C.3 explicitly says no ground-truth trajectory is supplied to or stored by the agent: feedback comprises its own rollout plus terminal SR/SPL. The critique therefore concerns ground-truth-derived outcome feedback and episode ordering, not use of an oracle path at inference. No verified executable official release was inspected and no benchmark was executed.

