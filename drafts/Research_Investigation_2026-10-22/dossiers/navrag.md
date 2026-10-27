# NavRAG

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM**. Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan. Findings of ACL 2025.

Primary paper: [NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM](https://arxiv.org/abs/2502.11142). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 9].

Associated files: NavRAG.pdf (Main PDF; 11 pages; SHA256 99f191fba4c9f1c0f4ff4ddb8ba7ca73e0232473dba0831fb96a8e691fb702a2).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Generate diverse user-demand-oriented goal instructions beyond the narrow style of trajectory speakers trained on small R2R/REVERIE corpora. NavRAG is primarily a data-generation system; its downstream DUET does not perform LLM retrieval at navigation time.

## 3. Mechanism, training, and inference

§3 pp 3–5 builds a bottom-up scene description tree from object/view to viewpoint, semantic-connectivity zone, and house. GPT-4o-mini describes 6 views per viewpoint; zones grow by connectivity plus LLM semantic membership, rather than Euclidean clustering alone. Scene context and 20 manually specified user profiles generate rough daily tasks, then top-down zone→viewpoint→view retrieval refines the instruction and selects a target. A graph planner supplies the path. Training data has 2,115,019 instructions across 861 training scenes (800 HM3D + 61 MP3D); separate 11 MP3D validation scenes are annotated for evaluation. Graph generation follows ScaleVLN. The trained navigation model uses DUET imitation/pretraining machinery, with object-box features removed in fairer downstream comparisons. No new inference loss is introduced by the data pipeline; instruction quality rests on a chain of model-generated descriptions/targets.

## 4. Novelty and evidence

User-demand simulation and hierarchical retrieval grounded in scene connectivity expand language diversity relative to route-to-text speakers. This does not certify factual correctness: the authors’ Limitations explicitly state no effective correctness evaluator and viewpoint-level, not precise object-level, target annotations.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 2 p 7 DUET trained on NavRAG achieves NavRAG 30.7 SR/25.4 SPL versus ScaleVLN-REVERIE 17.4/11.9; transfer to REVERIE 36.1/24.9 is a distinct setting. Table 3 p 8 matched no-object-box fine-tuning yields 57.3 SR/42.0 SPL versus ScaleVLN 55.9/40.6. Table 4 generator ablations change both training and validation generation processes (GraphRAG→GraphRAG versus NavRAG→NavRAG), so navigation differences do not isolate annotation correctness under a fixed test distribution. VERDICT: useful diversity/transfer evidence with explicitly unresolved semantic validation; prose task requests to manipulate objects are evaluated only as navigation.

The released shortest-path conversion imposes a goal-oriented route prior even though user-demand language is broad. Demand diversity and route-following fidelity are separate claims; this dataset is not a replacement for non-shortest-path RxR/R4R controls.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/MrZihan/NavRAG](https://github.com/MrZihan/NavRAG); inspected snapshot **7dd133e88797a6c4baacf820f3edf905740077fa**. Snapshot metadata and source are under `repositories/01/navrag`.

Inspected downstream DUET agent/environment/pretraining, conversion script, evaluator helper and training config. Full scene annotation/generation service reproducibility depends on released prompts, API behavior and data; no API generation was executed.

Important control: the released NavRAG DUET branch upweights STOP-target CE by 10 (agent.py 423–425; agent_base.py 141 unreduced CE), while retaining shared softmax normalization. Data conversion samples training starts with replacement from the nearest third of target distances. These are implementation choices that can affect stopping or task difficulty independently of hierarchical RAG. Generation chooses its API model through an external configuration, so exact regeneration also depends on that model/version and responses.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py) — teacher/rollout/navigation routines: Downstream policy is a DUET-family graph navigator.
- [VLN-DUET-NAVRAG/pretrain_src/train_rag.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/pretrain_src/train_rag.py) — training loop and data configuration: Dataset pretraining is a distinct optimization stage.
- [instruction_generator/convert_to_dataset.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/instruction_generator/convert_to_dataset.py) — instruction/path conversion: Generated annotations are converted into navigation training records.
- [VLN-DUET-NAVRAG/map_nav_src/scripts/rag_h14_envedit_mix.sh](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/scripts/rag_h14_envedit_mix.sh) — experiment configuration: Feature/data mixture choices are explicit and must be controlled.
- [instruction_generator/generate_instruction.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/instruction_generator/generate_instruction.py) — 22–70;74–150: API model is configured externally; hierarchical role→zone→viewpoint→direction→instance prompts construct demand instructions. Responses are parsed without independent semantic verification.
- [instruction_generator/convert_to_dataset.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/instruction_generator/convert_to_dataset.py) — 67–148: Uses seed 0, held-out building lists and shortest paths; samples starts with replacement from the nearest third of graph distances, 10 per training instruction versus 1 for held-out instructions.
- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py) — 394–425: Shared softmax STOP score is cached; active training multiplies STOP-target loss by 10.
- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent_base.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent_base.py) — 141: CrossEntropyLoss uses reduction=none, confirming per-example STOP weighting is active.
- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/env.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/env.py) — 314–335: Scores flattened graph paths with final/nearest target distances and SPL.
- [VLN-DUET-NAVRAG/pretrain_src/train_rag.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/pretrain_src/train_rag.py) — 180–260;349–428: Multi-task loader and optimizer; masked-token CE, region KL and action CE validation.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **navrag-B1 — Still unresolved in reviewed evidence; broader frontier audit required**. Generated descriptions, retrieval and target selection share model biases; navigation success on generated targets does not independently establish instruction truth. Evidence: §6 Limitations;Table 4 varying train/test generation. Type: data verification. Follow-ups: SRDF consistency filtering, CAST counterfactual audit. Confidence: high.
- **navrag-B2 — Partially addressed**. Targets are viewpoint-level and cannot support precise object manipulation claims. Evidence: §6 Limitations. Type: annotation granularity. Follow-ups: REVERIE object grounding. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://arxiv.org/abs/2508.13446; https://github.com/wz0919/VLN-SRDF
