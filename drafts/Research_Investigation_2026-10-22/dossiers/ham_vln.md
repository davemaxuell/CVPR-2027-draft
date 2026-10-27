# HAM-VLN

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**HAM-VLN: Harnessing Hierarchical Agentic Memory for Zero-Shot Vision-and-Language Navigation**. An Liu, Bingxi Liu, Hongyu Ding, Yixuan Jiang, Yaran Chen, Fulin Tang, Cong Leng, Hong Zhang, Jian Cheng arXiv preprint v1,31 July 2026 (2026).

Primary record: https://arxiv.org/abs/2607.29600v1



Supplied main file: [HAM-VLN.pdf](../../Paper/HAM-VLN.pdf). Pages actually reviewed: 1, 3-7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

HAM-VLN aims to bound the context passed to a language-model navigator while retaining semantic progress and knowledge of previously abandoned branches. It assumes RGB-D panoramas, odometry and a physical path planner. Its persistent place graph operates within an episode; this should not be conflated with learning from feedback across independent test episodes.

## Method, training, inference, and equations

PAPER METHOD (§3): a high-level Gemini 3.1-Pro agent produces a direction, backtrack or stop decision together with structured memory writes. Qwen3.6-35B grounds the selected intent to an image box. Depth, camera intrinsics and pose project that box to a spatial target, which the fast marching method planner uses for movement. The high-level textual decision, visual grounding and physical path execution are distinct stages; an error in any of them can cause the eventual episode to fail.

Working memory retains K recent waypoint observations, with K=1 by default. A longer-lived place-object graph stores spatial anchors, connectivity, object descriptions, visit metadata and failure notes. Retrieval begins with three seed places scored by cosine relevance to the subgoal, plus 0.3 times recency and 0.3 times salience. Recency decays by a factor of 0.85 per waypoint. Expanding to adjacent places and the current neighborhood produces a context capped at five places. These choices limit retrieval size rather than guaranteeing that every instruction-relevant observation is retained.

When the agent backtracks, it annotates places on the abandoned suffix with its decision rationale. The notes are advisory: they do not prohibit future visits or directly change navigation scores, and they reappear only if the associated place is retrieved. This distinction matters when assessing the risk of negative memory. The mechanism stores the agent's explanation of a failure; it does not certify that a target was visible, absent, or incorrectly recognized.

The navigation procedure uses pretrained models without navigation-specific parameter optimization. There is consequently no navigation training loss or optimizer to reproduce, but the pretrained perception and language models, API calls and planner remain prerequisites. The graph and notes are updated during inference. The reported context-token savings measure one part of inference cost; they do not establish identical physical paths, latency or total cost across model providers.

## Novelty and closest predecessors

Decision-coupled graph writes and bounded task-conditioned retrieval integrate working/episodic/semantic/reflection views. Graph memory, recency retrieval and failure notes have predecessors; the specific combination and controlled memory ablations support the contribution. Broad 'negative failure memory' claims already overlap.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 p.6 reports R2R 61.0±1.7 SR/48.1±.2 SPL and RxR 52.7±1.5/35.7±2.6 on 100-episode subsets, three runs. Published full-split baselines and older foundation models are not matched comparisons. Controlled Table 3 p.7: full raw history 53.3 SR and 746.1 K API tokens/episode versus HAM 61.0 and 244.9 K. Table 4 removes reflection but retains backtracking:55.7 SR/36.9 SPL , supporting memory contribution independent of simply allowing return. The 65% context reduction is relative to full history; versus the cheapest K=3 history, episode tokens fall 34.8%.

## Implementation audit

ACCESS SCOPE: supplied preprint and title searches did not establish an official released repository. No memory association, rejected-stop logic or scoring implementation was inspected.


## Bottlenecks and follow-up chain

- **03-ham_vln-negative: A failed visit is not a verified absence, and advisory notes can be stale or misattributed** Conditions: Changed subgoal, occluded target, place aliasing, new environmental state Cause: Agent-authored rationale is stored without a calibrated observability/coverage certificate Evidence: §3.4 explicitly advisory failure evidence; deduction from grounding and retrieval Type: methodological. Follow-ups:  PolarMem formalizes positive/negative/uncertain memory; PSG-Nav calibrates detections, but neither supplies the same coverage-conditioned branch evidence. Status: **Partially addressed**, confidence high.

## Source/access record and remaining uncertainty

The supplied method and reported subset experiments were reviewed. No verified official implementation was inspected. Memory fidelity, full API traces and a matched large-scale evaluation remain untested.

