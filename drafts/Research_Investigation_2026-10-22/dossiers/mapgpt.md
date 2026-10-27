# MapGPT

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation**. Jiaqi Chen, Bingqian Lin, Ran Xu, Zhenhua Chai, Xiaodan Liang, Kwan-Yee K. Wong ACL 2024 long papers (2024).

Primary record: https://aclanthology.org/2024.acl-long.529/



Supplied main file: [MapGPT.pdf](../../Paper/MapGPT.pdf). Pages actually reviewed: 1,3–8,12–13. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Locally choosing a neighboring viewpoint leaves zero-shot LLM agents unable to reason about unexplored branches and backtracking. Discrete MatterSim provides connectivity and navigable candidates; this is not learned free-space discovery or continuous locomotion. REVERIE evaluation covers navigation but explicitly omits object grounding.

## Method, training, inference, and equations

§3 Eq.1 generates thought T_t and categorical action from task description, instruction, action history, observations and candidates. Eq.2 adds online map M_t and previous plan P_(t−1), then produces T_t, P_t, a_t. G_t=(V_t, E_t) stores observed node IDs and simulator-provided edges; textual prompts distinguish visited, currently accessible and observed-but-inaccessible nodes. Their annotations are candidate images ( GPT-4V ) or BLIP2 / FasterRCNN descriptions ( GPT4 ). Current/visited annotations are reused from action/history; other frontier descriptions go into supplementary context. The plan is updated each step and may include multiple contingencies, but no search-optimality constraint is imposed. The action is a letter over N neighbors plus STOP. There is no navigation-training objective; pretrained caption/detection and paid GPT inference remain dependencies. REVERIE 'Surroundings' uses dataset object lists for stopping, whereas R2R does not use them (§3.1.2). Map connectivity is the DUET -style graph data structure, not the DUET learned dual-scale policy.

## Novelty and closest predecessors

An online linguistic topological graph and recurrent editable plan in a single expert prompt. Backtracking and graph memory predate it in learned VLN; contribution adapts their representation to frozen GPT4 /4 V. Analogical adds joint candidate captions; EvoNav adds retrieved cross-episode experience; HAM-VLN adds graph retrieval/compression and reflective memory.

## Results and independent verdict

Table 1 216 trajectories across 72 scenes GPT-4V 47.7 SR/38.1 SPL ; Table 2 783 instructions 11 unseen scenes 43.7/34.8 vs NavGPT 34/29. Text-only 38.8 SR improves success but 25.8 SPL is below NavGPT 29. Table 3 REVERIE 500 sample 31.6/20.3; trained HAMT/ DUET retested on same subset. These are subset benchmarks, not full R2R all-instruction leaderboard numbers. §4.2 backtracking in 49%episodes and successful correction within 80%of those are descriptive operational statistics, not causal evidence that every return helps. Extra graph/context/ API model differences confound broad comparisons. Appendix B (p.12) identifies inaccurate instruction-detail following and insufficient exploration in ambiguous scenes as failure modes. Figure 4 explicitly instructs the REVERIE agent to ignore some interactive actions and relax color adherence; this is a task-specific prompt assumption, not verified instance grounding.

## Implementation audit

INSPECTED CODE. GPT/one_stage_prompt_manager.py:41–87 assigns place IDs to simulator-provided candidates and records adjacency;124–165 serializes visited connectivity and unvisited observed nodes. Lines 216–263 build a prompt containing the instruction, trajectory, history, graph, previous plan and candidate images;265–295 extract the next textual plan. This verifies that the planning state is text over an observed topology, not a learned world-transition model. vln/gpt_agent.py:149–176 maps the selected option to a simulator action and records history;170 explicitly limits the implementation to batch size one. Parsing and stop-after heuristics affect behavior. No navigation training objective is introduced. The mechanism and inference interfaces were inspected; the entire feature/candidate preprocessing and exact sampled-evaluation list were not independently regenerated. No API or simulator execution occurred.

Repository: https://github.com/chen-judge/MapGPT. Inspected snapshot commit: `7c642f4507a8dde6703e2c61c8fd8ff3bc4bd322`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [GPT/one_stage_prompt_manager.py:41-165,216-352](https://github.com/chen-judge/MapGPT/blob/7c642f4507a8dde6703e2c61c8fd8ff3bc4bd322/GPT/one_stage_prompt_manager.py)
- [vln/gpt_agent.py:149-176](https://github.com/chen-judge/MapGPT/blob/7c642f4507a8dde6703e2c61c8fd8ff3bc4bd322/vln/gpt_agent.py)

## Bottlenecks and follow-up chain

- **MAPGPT-1: Exploration gains can lengthen paths and visual-to-text conversion loses discriminative cues** Conditions: Text-only GPT4 , several plausible branches Cause: Descriptions discard visual evidence and repeated planning encourages further exploration Evidence: Table 2 GPT4 SR 38.8/ SPL 25.8; §4.2 explanation Type: Methodological / insufficiently tested. Follow-ups:  Analogical comparisons, EvoNav visual banks, HAM-VLN selective graph memory; uncertainty of detection and fair cost controls persist. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

Main method/results, Appendix A implementation and task prompts (pp.12–13), and the released graph/prompt/action path were inspected. Remaining selected qualitative figures were not exhaustively assessed. No API benchmark was executed.

