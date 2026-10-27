# PolarMem

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**PolarMem: A Training-Free Polarized Latent Graph Memory for Verifiable Vision-Language Models**. Zhisheng Chen, Tingyu Wu, Zijie Zhou, Zhengwei Xie, Jinhan Li, Ziyan Weng, Liang Lin, Jingwei Song, Zikai Xiao, Yingwei Zhang arXiv 2602.00415v2, 30 May 2026; conference acceptance not verified (2026).

Primary record: https://arxiv.org/abs/2602.00415



Supplied main file: [PolarMem.pdf](../../Paper/PolarMem.pdf). Pages actually reviewed: 1,3–7. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Multimodal RAG retrieves semantically similar evidence even when it contradicts the query. PolarMem records positive/negative/uncertain concept relations. It is not VLN benchmark evidence and does not measure partial-observation navigation absence.

## Method, training, inference, and equations

§3 Eq 1 frozen VLM proposes concepts C_prop, then mean Yes probability over ensembled prompts defines s_c∈[0,1]. Anchors 0,1 augment instance scores; Otsu -style between class variance argmax givesτ*. Marginδ=κσ_within leads HAS ifs>τ+δ, NOT_HAS ifs<τ−δ, else Uncertain(Eqs 2–3). Text entity chunks align concept nodes. Graph contains visual, text, concept nodes, typed HAS/NOT_HAS edges; hybrid representation combines global visual, local patches, and serialized polarized concept embedding(Eqs 4–7). Query parses positive Q+and avoid Q−. Algorithm 1 sets logical score−1 for conflict,1 if any positive intersection,0 otherwise; lexicographic sort(logic, cosine similarity), then first K. This ranks rather than strictly removes conflicts: when fewer thanK non conflicts exist, returned context can contain conflicts. 'Entailment'checks any positive concept, not all query requirements. Frozen VLM generates from serialized retrieval context(Eqs 8–9); no backbone optimization. Otsu threshold is distribution partition, not calibrated probability or verified absence.

## Novelty and closest predecessors

Negative evidence as first-class memory edges and lexicographic logic priority. Closest ViLoMem / GraphRAG already organizes errors/relations; distinguishes positive and negative concept states explicitly. This must be cited for any proposed negative VLN memory; spatial view coverage and time validity remain unmodeled here.

## Results and independent verdict

Table 1 eight backbones/six benchmarks: Qwen 2.5 VL 7 B MRAG 60.7→70.8, Visual RAG 40.2→49.9, but MMMU 56.3→55.7; Qwen 32 B MMMU 68.4→66.5. Figure 3 Conflict@3.067→.033 and filter rate.875→.938 support reduced contradictions, not zero-conflict guarantee. Same pool/top K controls help isolate organization. Gains not universal: stronger backbone general reasoning can be harmed by hard constraints. Authors acknowledge concept proposal cannot recover un proposed answer-critical concepts.

## Implementation audit

INSPECTED CODE. src/backbone/qwen_wrapper.py:25–39 freezes Qwen and obtains the first token IDs for Yes/No;56–127 asks a single presence prompt and normalizes those two logits. src/logic/otsu_gate.py:25–46 uses Otsu when variance is sufficient, otherwise0.5, then a fixed±0.05 margin. This differs from the paper's ensemble and dispersion-scaled margin. src/database/neo4j_client.py:197–217 ranks a weighted sum of positive/negative matches, not the lexicographic ordering in the paper algorithm. The public retrieve_candidates path calls search_mixed_modality, whose image query at 303–319 scores positive-query HAS matches minus twice NOT_HAS matches; its negative-query argument is not used in that query.

These are material implementation differences: missing facts, observed absence and query negation must not be conflated when evaluating the intended method. The returned top-K list can still contain conflicting candidates. The released code is useful for inspecting the graph-memory concept, but it is not yet an exact executable specification of the paper's retrieval guarantee. Database queries and VLM outputs were read, never executed; benchmark scripts and complete dataset provenance have not been independently reproduced.

Repository: https://github.com/czs-ict/PolarMem. Inspected snapshot commit: `4d0de0befe3571ddeaccafd55ac19aa764ef9b0d`. Retrieval status: Source snapshot retrieved; file inspection tracked separately. Project code was not executed.

- [src/backbone/qwen_wrapper.py:25-39,56-127](https://github.com/czs-ict/PolarMem/blob/4d0de0befe3571ddeaccafd55ac19aa764ef9b0d/src/backbone/qwen_wrapper.py)
- [src/logic/otsu_gate.py:1-48](https://github.com/czs-ict/PolarMem/blob/4d0de0befe3571ddeaccafd55ac19aa764ef9b0d/src/logic/otsu_gate.py)
- [src/pipeline/retriever.py:110-139](https://github.com/czs-ict/PolarMem/blob/4d0de0befe3571ddeaccafd55ac19aa764ef9b0d/src/pipeline/retriever.py)
- [src/database/neo4j_client.py:120-146,197-240,284-362](https://github.com/czs-ict/PolarMem/blob/4d0de0befe3571ddeaccafd55ac19aa764ef9b0d/src/database/neo4j_client.py)

## Bottlenecks and follow-up chain

- **POLAR-1: False negative memories can suppress valid evidence; ranking does not ensure conflict exclusion** Conditions: Occluded/partial images, missing candidate concepts, fewer thanK compatible memories Cause:  VLM score partition conflates uncertainty with absence; hard constraints rely on fallible perception Evidence: Eqs 1–3, Algorithm 1,§3.3 coverage boundary and Table 1 degradations Type: Methodological / evaluation. Follow-ups:  Navigation would need view coverage/temporal validity/abstention; HAM-VLN reflection is advisory, ContextNav has Unknown state but different task. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

Core method, principal retrieval results and pinned graph-memory implementation were inspected. Full index construction, all dataset splits and end-to-end retrieval costs were not reproduced. It supplies no VLN performance evidence.

