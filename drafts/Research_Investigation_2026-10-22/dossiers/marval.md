# MARVAL

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**A New Path: Scaling Vision-and-Language Navigation with Synthetic Instructions and Imitation Learning**. Aishwarya Kamath, Peter Anderson, Su Wang, Jing Yu Koh, Alexander Ku, Austin Waters, Yinfei Yang, Jason Baldridge, Zarana Parekh. CVPR 2023.

Primary paper: [A New Path: Scaling Vision-and-Language Navigation with Synthetic Instructions and Imitation Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Kamath_A_New_Path_Scaling_Vision-and-Language_Navigation_With_Synthetic_Instructions_and_CVPR_2023_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: MARVAL.pdf (Main PDF; 11 pages; SHA256 1a96c7fe8a0fdd0eee4feadd1ce1e99fdb66dabefe6106134d473d88ce173c80); MARVAL (slides).pdf (Conference slides; 30 pages; SHA256 5f6ab52e30525ebcd26e026ef60b6e4c25a1194b706ee81a1e0ce630bd54d643).

Main-paper analysis is supplemented by the explicitly listed associated-file review below. Exact SHA256 duplicates were deduplicated and not counted as additional evidence. MARVAL (slides).pdf: pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]; All slide text inspected; slides 22–23 also rendered and visually inspected for failure analysis and DAgger expert rules.

## 2. Problem and operating assumptions

Test whether higher-quality, larger synthetic navigation datasets plus imitation learning can replace complicated RL pipelines. The main challenge is scaling both environments and grounded instructions, while retaining the language style of the evaluation dataset.

## 3. Mechanism, training, and inference

§3–4 pp 3–6: a randomly initialized mT5-style encoder concatenates instruction, pooled panoramic/action history, current 36 views and candidates with learned modality/relative-position biases. Fixed 640D MURAL-large features inherit large external image–text pretraining. Pretraining combines masked language, 20-bin progress classification, candidate-constrained action CE and 37-class direction/STOP CE; constrained and unconstrained logits are equally mixed. Fine-tuning updates only word embeddings, using behavioral cloning or one effective DAgger iteration. Experts follow the next reference action when on path, shortest-to-goal when the reference is shortest, and shortest-back-to-reference otherwise. Marky generates 3.2M multilingual instructions on 491 Gibson scenes; graph construction combines learned reachability, geodesic/Euclidean ratio and MST connectivity. SE3DS creates 200 perturbed panorama variants per Matterport scene. Model pretraining reaches 5.14M iterations, batch 128 (>650M state examples). Pre-exploration additionally self-trains on evaluation environments and is a separate privileged regime.

## 4. Novelty and evidence

Large multilingual Marky-Gibson data and a deliberately simple, pure-imitation Transformer demonstrate the importance of instruction quality, environment diversity and distribution correction. It is outside DUET’s dual-branch architecture, although it shares panoramic graph tasks and history pooling.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 2 p 7 replacing low-quality Speaker data with Marky yields large gains; mixing Speaker back in can hurt, so more data is not uniformly better. Table 3 RxR test SR 60.7/nDTW 66.8; pre-exploration 61.8/68.6 must be separated. Table 4 R2R test 62 SR/58 SPL is not SOTA. Authors attribute weaker R2R transfer to Marky’s RxR language style and acknowledge seen gains exceed unseen gains. VERDICT: strong evidence for data quality/domain matching, but graph F1 of .70 and synthetic panorama/instruction errors remain. Longer training/model scale vary with datasets in several ablations, so purely causal attribution to one data source is limited.

Conference slide 22 contrasts a greater fraction of error-free MARVAL trajectories with lower eventual success than humans and motivates recovery from deviations. The visual plot is descriptive and does not establish a calibrated per-error recovery probability. Slide 23 restates the DAgger rule: follow the reference when on it, otherwise take a shortest route back to it; this already couples recovery labels to route adherence. The Marky data link points to the official RxR data repository, not a complete executable MARVAL implementation.

## 6. Implementation and reproducibility

A verified official executable MARVAL repository was not located in the bounded primary-source search. Marky, MURAL, SE3DS and data-generation dependencies are separate artifacts; their existence does not establish full MARVAL reproducibility. The paper provides explicit expert-label rules and objectives.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

**Execution status:** No model source inspection is claimed beyond the explicit scope above. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **marval-B1 — Partially addressed**. Synthetic-language style can help one benchmark while transferring poorly to another; low-quality additional data may reduce performance. Evidence: Tables 2–4 and §6. Type: data distribution. Follow-ups: ScaleVLN, SRDF, NavRAG. Confidence: high.
- **marval-B2 — Partially addressed**. Automatic graph generation and novel-view rendering add synthetic transition/observation errors; larger dataset claims also consume substantial compute. Evidence: §4 graph F1=.70;5.14M iterations. Type: data/compute. Follow-ups: ScaleVLN graph refinement. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/wz0919/ScaleVLN; https://github.com/MrZihan/NavRAG
