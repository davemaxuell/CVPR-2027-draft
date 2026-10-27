# ProFocus

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation**. Wei Xue, Mingcheng Li, Xuecheng Wu, Jingqun Tang, Dingkang Yang, Lihua Zhang CVPR 2026, identified by the main-paper watermark (2026).

Primary record: https://arxiv.org/abs/2603.05530



Supplied main file: [ProFocus.pdf](../../Paper/ProFocus.pdf). Pages actually reviewed: 1,3–7. Associated local versions are recorded below with SHA256 identifiers. Exact duplicate bytes were not reread. Distinct alternatives were checked for the relevant method/results; their complete bibliographies were not separately reread.

Associated material:

- ProFocus (alternate).pdf; reviewed pages/status: 5–8; SHA256 `3d7809ccbd120cd49027bf13ee0c02a83e131639b75dcd91d8727ec09735e9d1`

## Problem, motivation, and operating assumptions

ProFocus addresses the cost of presenting a frozen language-model navigator with entire panoramas and an expanding history of waypoints. Its inputs include an instruction, panoramic observations and graph navigation candidates. It selects image regions and historical branches that appear relevant to the current decision. This is an important operating assumption: inspecting a crop of an already captured panorama changes the model's access to existing evidence; it does not necessarily move the camera to reveal an occluded object.

## Method, training, inference, and equations

Section 3 describes two connected inference modules. First, the VLM processes overlapping views of the panorama and detects object boxes. SAM2 masks and monocular depth support object-distance estimates. Object category, box, heading and depth are converted into a textual egocentric map. An orchestrating LLM asks a focused visual question; the VLM inspects the corresponding crop and answers; the LLM decides whether the available information is sufficient. Equations 1–8 describe this interaction. A sufficiency judgment is a model-generated decision, not a measured probability that the scene has been understood correctly.

Second, branch-diverse search selects which historical paths to expose to the decision model. The LLM assigns each newly observed waypoint a semantic value in [0,1]. BD-MCTS organizes the navigation history as a tree, dropping cross-branch edges when converting the cyclic graph. It backs up the values of observed children through visited ancestors. The incremental mean update in Equations 9–11 has the form Q ← Q + (R − Q)/N, where Q is an accumulated value estimate, R is the new backed-up value and N counts updates. The reviewed method does not use simulated Monte Carlo rollouts; the name therefore does not establish the usual statistical properties of a rollout-based search algorithm.

Equation 12 ranks leaves using values along their ancestor paths and a penalty for normalized shortest-path distance, weighted by λ. The selection limits children per parent to two and retains a top-K set. The decision LLM then receives the selected path contexts and chooses the next waypoint (Equation 13). The underlying navigation graph remains available for routing; the tree is a context-selection structure, so loss of an edge there must be distinguished from physical unreachability.

The reported model pairs are DeepSeek-V3 with GLM-4.5V, or Qwen3-Max with Qwen3-VL-Max. The method uses frozen models, with no navigation fine-tuning objective or optimizer. Its costs include perception, repeated query/answer calls and historical-context processing. The reviewed material does not establish a complete reproducible API budget or latency distribution. Neither the LLM's value scores nor its sufficiency decisions have a demonstrated calibration guarantee.

## Novelty and closest predecessors

The specific contribution is the combination of query-guided crop perception and branch-diverse selection of historical context. MapGPT already uses graph-based reasoning and backtracking, while DiscussNav assigns reasoning to multiple agents. Those precedents limit a broad claim that structured reasoning or returning to previous nodes is itself new. ProFocus's perception and context-selection ablations are more relevant evidence for its particular mechanism than its use of the MCTS name.

## Results and independent verdict

PAPER EXPERIMENT: Tables 1–2 report R2R SR/SPL of 52.5/39.8 and REVERIE 40.0/24.8 for the Qwen model pair. Comparison methods were reimplemented with modern backbones and evaluated on a random subset; the exact subset size was not established in the reviewed main setup. The full R2R result in Table 3 is 52.0 SR, rather than Table 1's 52.5, so those rows should not be silently treated as identical runs.

Removing proactive perception reduces R2R SR from 52 to 48 and SPL from 39.8 to 34.4 in the reported ablation. Removing BD-MCTS reduces REVERIE SR from 40.0 to 37.78 but increases SPL from 24.83 to 27.37. That result supports a success-versus-efficiency tradeoff, not uniform dominance. The 50% SR result on the 30 longest episodes is useful small-sample evidence but cannot establish broad long-horizon reliability. REVERIE object grounding is explicitly omitted, so navigation scores are not evidence of complete remote-object grounding performance.

## Implementation audit

No verified official code URL found in reviewed paper/search; unrelated artistic ProFocus repository was excluded. No implementation inspected or executed.


## Bottlenecks and follow-up chain

- **PROFOCUS-1: Semantic value and perception sufficiency can be confidently wrong; branch pruning loses useful loops** Conditions: Ambiguous landmarks, useful cross branch route, uncertain visual crops Cause:  LLM judgments lack calibration; a cyclic adaptation drops edges; top K prunes context Evidence: Eqs 6–12, Table 4 SPL tradeoff Type: Methodological / evaluation. Follow-ups:  PSG-Nav calibrates structured goal uncertainty; HAM-VLN retrieves reflection memory; physical value-of-information still unestablished. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

Main method, tables and the corresponding alternate-version result pages were read. No official executable code was verified. Exact sampled episode identities, API budgets and loop limits were not established by the reviewed material. No benchmark was executed.

