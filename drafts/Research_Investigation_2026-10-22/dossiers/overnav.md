# OVER-NAV

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**OVER-NAV: Elevating Iterative Vision-and-Language Navigation with Open-Vocabulary Detection and StructurEd Representation**. Ganlong Zhao, Guanbin Li, Weikai Chen, Yizhou Yu. CVPR 2024.

Primary paper: [OVER-NAV: Elevating Iterative Vision-and-Language Navigation with Open-Vocabulary Detection and StructurEd Representation](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_OVER-NAV_Elevating_Iterative_Vision-and-Language_Navigation_with_Open-Vocabulary_Detection_and_StructurEd_CVPR_2024_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: OVER-NAV.pdf (Main PDF; 11 pages; SHA256 485b4d4af5d787036d2897dc16895c845c7c39223fca2b56971672a8abf32b4d).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Persistent tour memory should retain instruction-relevant objects beyond the 13 fixed categories of IVLN’s semantic map and avoid overwhelming the navigator with raw historical observations.

## 3. Mechanism, training, and inference

§3 pp 3–6: GPT-3.5 extracts landmark/attribute phrases; OWL-ViT-L/14 detects them in panoramas at visited viewpoints. An Omnigraph stores viewpoint connectivity, keyword labels, confidence and headings across episodes. For discrete HAMT, collect neighbors within 3 graph steps, keep highest-confidence box per keyword/viewpoint, select modal distance/direction for duplicates, and append keyword-text embeddings plus direction/distance embeddings to instruction tokens. For continuous MAP-CMA, register viewpoints when farther than 1m from existing ones, redetect stored panoramas within 0.25m, collect neighbors within 7m and attend from instruction to keyword-position features. Eq 1 adds projected keyword CLS and angle embeddings plus distance embedding, all in the navigator’s text dimension. Train the underlying policies with IVLN hyperparameters; one 3090 runs navigation and another detector. Inference has no additional per-scene weight training, but navigation models are trained and memories persist.

## 4. Novelty and evidence

Open-vocabulary semantic memory and structured egocentric retrieval extend IVLN. This is not a frozen VLM directly controlling navigation: the LLM preprocesses language, detector supplies memory, and trained HAMT/MAP-CMA predicts actions.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 IR2R unseen t-nDTW 50→55 and SR 56→60 over HAMT. Table 2 continuous t-nDTW 47→50 with SR 35 unchanged, illustrating path-fidelity improvement without endpoint gain. Table 3 p 7 full open vocabulary 55 versus fixed-category 52; Type III 56 uses a second evaluation pass without clearing memory, explicitly privileged repeated exposure. Table 4 retrieval radius 3 best, larger neighborhoods hurt. VERDICT: credible structured-memory gain, but duplicate-keyword modal aggregation discards instance identity and scene changes are not tested.

Section 5.1, PDF p 8, explicitly explains that Type III memory contains the exact episode keywords in advance after a previous evaluation pass. Its advantage is therefore transductive repeated exposure, not the ordinary first-pass condition. The map visualization is a selected tour and does not quantify stale-memory behavior under changed objects.

## 6. Implementation and reproducibility

Official executable implementation was not located in the bounded paper/project search. Underlying HAMT and IVLN-CE were inspected separately; their availability does not count as inspection of OVER-NAV modifications. The locally supplied main paper was reviewed; a distinct supplementary prompt file was not included in this assignment inventory, and no claim of reviewing one is made.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

**Execution status:** No model source inspection is claimed beyond the explicit scope above. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **overnav-B1 — Partially addressed**. Collapsing same-keyword detections to modal heading/distance can merge distinct instances and lose attribute-grounding distinctions. Evidence: §3.2 filtering algorithm. Type: methodological. Follow-ups: GaussianMap semantic grouping, HSGM instance maps. Confidence: high.
- **overnav-B2 — Uncertain because evidence is insufficient**. Persistent detections can become stale; static-tour gains do not validate changed-room/object scenarios. Evidence: IVLN setup and update rules. Type: environment shift. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://arxiv.org/abs/2403.17334; https://github.com/Teacher-Tom/HSGM_public
