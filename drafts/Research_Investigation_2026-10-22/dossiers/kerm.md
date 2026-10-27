# KERM

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**KERM: Knowledge Enhanced Reasoning for Vision-and-Language Navigation**. Xiangyang Li, Zihan Wang, Jiahao Yang, Yaowei Wang, Shuqiang Jiang. CVPR 2023.

Primary paper: [KERM: Knowledge Enhanced Reasoning for Vision-and-Language Navigation](https://openaccess.thecvf.com/content/CVPR2023/html/Li_KERM_Knowledge_Enhanced_Reasoning_for_Vision-and-Language_Navigation_CVPR_2023_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: KERM.pdf (Main PDF; 10 pages; SHA256 5ef50fe5477832f10dc3699682f3aaa8c1410abe9618021621c0be3c946d321c).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Augment DUET view/history features with relational descriptions not explicit in holistic visual embeddings. Discrete panoramic graph navigation, with object boxes/detector required for grounding tasks.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3 pp 3–5): normalize Visual Genome attribute-object and subject-predicate-object phrases into 630k unique facts. Crop each panorama view into 5 regions; retrieve top 5 facts per region by CLIP cosine similarity, producing 25 fact tokens/view. Instruction-token dot products followed by row max score relevance; purify facts, visual regions and history. Two-layer cross-modal fact attention enhances view/history features, instruction-CLS-guided attention pools subregions, FFN combines them, then DUET predicts global action and object. Train MLM, masked-view classification, SAP and optional OG; finetune with pseudo-expert IL. R2R pretraining 200k and finetuning 20k iterations on 4 RTX3090s; REVERIE 100k/20k. Retrieval uses external Visual Genome/CLIP priors, not verified facts guaranteed true in the current scene.

## 4. Novelty and evidence

Knowledge retrieval plus instruction-conditioned purification and separate fact interactions, atop DUET. Table 4 isolates modules; Table 5 compares relation facts with 5 detected object labels embedded through the same text encoder.

## 5. Results and independent verdict

PAPER EXPERIMENT: Table 2 p 7 R2R test KERM 69.73 SR/59.25 SPL versus reproduced DUET 69.25/58.68: modest gain. REVERIE Table 1 test KERM 52.43 SR/39.21 SPL versus DUET 52.51/36.06: efficiency improves while SR slightly drops. Table 5 val-unseen facts 50.44 SR/35.38 SPL versus object labels 49.73/33.95. Verdict: relational feature utility is supported, but broad claims of universally better navigation should respect dataset and metric differences.

Table 6, PDF p 8, varies region granularity: REVERIE unseen SPL is 34.46 for one crop, 35.38 for five and 34.87 for nine. More retrieved regions are therefore not monotonically better. The selected qualitative attention example establishes relevance to an example, not factual correctness of every retrieved phrase.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/XiangyangLi20/KERM](https://github.com/XiangyangLi20/KERM); inspected snapshot **b51504334bebb2805b4608b7774bac076d85432a**. Snapshot metadata and source are under `repositories/01/kerm`.

INSPECTED CODE confirms knowledge/crop/instruction/history inputs to panorama encoding and inherited graph/local masking. Retrieval and knowledge-base parsing scripts exist; their downloaded snapshots preserve the preprocessing pathway. No timing, factuality or full reproduction executed.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [knowledge_nav_src/models/vilmodel.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/models/vilmodel.py) — forward_panorama_per_step1465–1467;forward_navigation_per_step1747;masking1823–1833: knowledge-conditioned DUET pipeline
- [preprocess/retrieve_facts.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/preprocess/retrieve_facts.py) — FeaturesDB22–28: HDF5 feature loading for retrieval
- [preprocess/get_knowledge_base.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/preprocess/get_knowledge_base.py) — VisualGenomeCaptions15–21: Visual Genome annotation parsing
- [knowledge_nav_src/r2r/agent.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/r2r/agent.py) — 426–442;515: Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/pretrain_src/data/dataset.py) — 419–512: Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [knowledge_nav_src/r2r/eval_utils.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/r2r/eval_utils.py) — 6–25: Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [knowledge_nav_src/scripts/run_r2r.sh](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/scripts/run_r2r.sh) — 19–68: Pinned script selects dynamic fusion and external trained checkpoint; training command is commented out in this file.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **kerm-B1 — Partially addressed**. CLIP top-k retrieval supplies plausible but unverified region facts. Instruction-aware filtering measures relevance, not truth; wrong relationships can be amplified. Evidence: AUTHOR §3.2–3.3; DEDUCTION. Type: methodological. Follow-ups: OVER-NAV, NavRAG, UAGM. Confidence: medium.
- **kerm-B2 — Uncertain because evidence is insufficient**. Five crops×five facts×36 views increases representation/preprocessing load; no matched-latency baseline in inspected main tables isolates utility per compute. Evidence: AUTHOR representation counts; DEDUCTION. Type: insufficiently tested. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: medium.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: 
