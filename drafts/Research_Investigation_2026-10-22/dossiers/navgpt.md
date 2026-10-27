# NavGPT

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models**. Gengze Zhou, Yicong Hong, Qi Wu AAAI 2024 (2024).

Primary record: https://ojs.aaai.org/index.php/AAAI/article/download/28597/29161



Supplied main file: [NavGPT.pdf](../../Paper/NavGPT.pdf). Pages actually reviewed: 1, 3–7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Tests whether a frozen language model can execute navigation instructions without VLN training. It operates on a discrete graph with provided viewpoint IDs and candidates. Visual caption quality and historical object tracking are central challenges; 'purely LLM -based' describes its decision module, not absence of visual or simulator dependencies.

## Method, training, inference, and equations

Eq.1: (reasoning, action)= LLM (prompt(system, instruction, V FM(observation), history)). Twenty-four images per viewpoint cover 8 headings×3 elevations,45° field of view. BLIP2 ViT -G FlanT5 XL captions each; GPT-3.5 summarizes elevations to 8 direction descriptions. FastRCNN detects objects; simulator depth at bounding-box center and relative heading provide object geometry, filtering within 3 m. GPT4 ReAct-style reasoning then chooses a valid candidate ID or Stop. History stores observation/reason/action triples with GPT-3.5 summaries. No navigation loss or parameter update is used; captions, depth, IDs and language summaries form a lossy observation bottleneck. Navigation graph actions are not continuous motor commands. The action-space extension A∪R is a prompting formalism: emitting a thought does not interact with the environment or receive new evidence.

## Novelty and closest predecessors

Early explicit ReAct navigation and empirical analysis of frozen GPT4 's planning language. Existing self-monitoring, subgoal decomposition and object grounding are reproduced as prompt behaviors, not invented navigation principles. Direct successors MapGPT and NavGPT2 target graph-memory and visual-caption bottlenecks.

## Results and independent verdict

Table 1 R2R 783 trajectories/11 unseen scenes 34 SR/29 SPL versus DUET 72/60; an untrained LXMERT- DUET baseline 1 SR does not equalize pretraining capability. GPT-3.5 ablation 216 samples across 72 scenes:45° field of view 16.67 SR versus 60°10.19; object information 11.11→15.97, depth 16.67(Table 2–3). Authors explicitly acknowledge caption/summarization loss. Generated top-down trajectory and instructions are selected qualitative examples, not systematic spatial-reconstruction accuracy or demonstrated faithfulness.

## Implementation audit

INSPECTED CODE. nav_src/agent.py:300–325 serializes directional captions, object headings/distances and navigable viewpoint IDs. Lines 353–390 construct action/history prompts and call the LLM;397–406 reject nonexistent viewpoint IDs before simulator movement;421–431 summarize the resulting history. create_vln_agent:558–622 constructs action/backtrace tools and a LangChain executor with an iteration cap. This is prompted inference with no navigation-specific optimizer or training loss.

The released nav_src/env.py:264–290 evaluates final and oracle error, uses the success margin, and computes SPL as success times reference length divided by the larger reference/predicted length. Lines 294–321 aggregate the metrics. The prompt file and core loop were inspected; BLIP/object-description caches and exact historical API behavior were not regenerated. Consequently source inspection verifies interfaces and evaluator structure, not the numerical paper results. No API call, MatterSim episode or model execution was performed.

Repository: https://github.com/GengzeZhou/NavGPT. Inspected snapshot commit: `b3fc8a21b8a1f66b09dde833bcfc25767ef1962d`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [nav_src/agent.py:300-440,558-622](https://github.com/GengzeZhou/NavGPT/blob/b3fc8a21b8a1f66b09dde833bcfc25767ef1962d/nav_src/agent.py)
- [nav_src/env.py:264-321](https://github.com/GengzeZhou/NavGPT/blob/b3fc8a21b8a1f66b09dde833bcfc25767ef1962d/nav_src/env.py)

## Bottlenecks and follow-up chain

- **NAVGPT-1: Captioning and summarization omit landmarks and object continuity** Conditions: Long histories or instruction-critical small objects Cause: Serial image-to-text compression and fixed caption granularity Evidence: pp.6–7 limitations and Tables 2–3 Type: Methodological / evaluation. Follow-ups:  MapGPT uses direct GPT-4V ; NavGPT2 learns visual alignment; neither makes generated explanations necessarily causal or faithful. Status: **Partially addressed**, confidence Moderate.

## Source/access record and remaining uncertainty

The supplied main paper and targeted prompt/inference/evaluation code were inspected. The complete original caption caches and API backend cannot be reconstructed from this review; additional supplementary examples were not exhaustively reviewed. No benchmark was executed.

