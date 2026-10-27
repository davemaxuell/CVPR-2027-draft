# ScaleVLN

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Scaling Data Generation in Vision-and-Language Navigation**. Zun Wang, Jialu Li, Yicong Hong, Yi Wang, Qi Wu, Mohit Bansal, Stephen Gould, Hao Tan, Yu Qiao. ICCV 2023.

Primary paper: [Scaling Data Generation in Vision-and-Language Navigation](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Scaling_Data_Generation_in_Vision-and-Language_Navigation_ICCV_2023_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: ScaleVLN.pdf (Main PDF; 12 pages; SHA256 4b3616d4881aa332eb527228d486f09f91cc9a37e65bd7df6af7586d25a1be25); ScaleVLN (alternate 1).pdf (Alternate / duplicate PDF; 12 pages; SHA256 4b3616d4881aa332eb527228d486f09f91cc9a37e65bd7df6af7586d25a1be25); ScaleVLN (alternate 2).pdf (Alternate / duplicate PDF; 12 pages; SHA256 4b3616d4881aa332eb527228d486f09f91cc9a37e65bd7df6af7586d25a1be25); ScaleVLN (supplement).pdf (Supplement; 6 pages; SHA256 ca89c239b0312fb9561d781b14e6970563c2aaf7ccfac77163d7d37293f781e4); ScaleVLN (supplement copy).pdf (Supplement; 6 pages; SHA256 ca89c239b0312fb9561d781b14e6970563c2aaf7ccfac77163d7d37293f781e4).

Main-paper analysis is supplemented by the explicitly listed associated-file review below. Exact SHA256 duplicates were deduplicated and not counted as additional evidence. ScaleVLN (supplement).pdf: pages [1, 2, 3, 4, 5, 6]; Training/data details, Tables 1–6 and graph/image figure captions inspected; visual graph and image quality figures do not quantify repair reliability. Supplement copy is exact SHA duplicate.

## 2. Problem and operating assumptions

Insufficient environment diversity and trajectory-language supervision for generalization. ScaleVLN constructs training data; DUET+ScaleVLN is a trained DUET configuration, not a new independent architecture. Uses HM3D/Gibson simulator meshes, navigability oracles, rendered RGB and an existing instruction speaker.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3 pp 3–4): sample navigable positions at minimum 0.4m separation, agglomeratively cluster at 1m, initially connect nearby points≤5m with degree cap 5, refine connectivity/traversability, render panoramas, repair mesh artifacts using a Co-Mod GAN trained on rendered/camera pairs, enumerate shortest paths with 3–5 intermediate nodes, then use EnvDrop speaker. Result 4,941,710 pairs from 800 HM3D plus 491 Gibson scenes. Training: CLIP-B/16 features; DUET for R2R/CVDN/REVERIE and HAMT plus waypoint predictor for R2R-CE. MLM/SAP pretraining is favored over also using MRM; DAgger finetuning mixes human and selected augmented data. R2R reported 20k pretraining iterations, batch 256, learning rate 5e-5, about 72 GPU-hours; finetuning peak about 48 GPU-hours (§4.1 p 5). Inference is the original navigator; no online generator required.

## 4. Novelty and evidence

Data construction quality and a component-level study of graph quality, environment count, image restoration, instruction generator and train-stage mixing. Tables 1–7 isolate these choices more usefully than the strongest leaderboard variant, which changes visual backbone and adds EnvEdit.

## 5. Results and independent verdict

PAPER EXPERIMENT: Table 9 p 8 distinguishes base DUET+ScaleVLN test 77 SR/68 SPL from starred EnvEdit+CLIP-H/14 variant 80/70; original DUET 69/59. Table 2 p 6 holds 700k samples while varying scenes; unseen SPL rises 64.59→65.32→66.00 for 200→400→800 scenes. Table 5 compares EnvDrop SR 76.59/SPL 67.74 against GPT-2 SR 75.82/SPL 66.08. Table 7 shows MLM+SAP 76.59 SR/67.74 SPL versus adding MRM 75.82/66.62. Verdict: compelling evidence for data/scene quality, without establishing a universal scaling law or language independence. Its suggestion to investigate stopping comes from remaining OSR–SR, not a causal stop-mechanism intervention.

Supplement Table 5 shows improved R4R route fidelity: nDTW 51.21→52.77 and SDTW 30.00→32.73 after scaling, alongside SR 43.97→45.60. This is counterevidence to a blanket claim that scaling necessarily weakens instruction following. Supplement Table 4 lists Seq2Seq SR 37.34→47.51, a 10.17 percentage-point difference; nearby prose says 12.2 points, an arithmetic inconsistency. REVERIE supplementary Table 6 shows higher navigation SR does not guarantee higher grounding: validation RGS 35.76 is below AutoVLN 36.58. These comparisons do not causally measure language reliance.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/wz0919/ScaleVLN](https://github.com/wz0919/ScaleVLN); inspected snapshot **3bdff0da2216066cade2e7923c4ad0b5c36d6592**. Snapshot metadata and source are under `repositories/01/scalevln`.

Pinned tree and source show a released REVERIE DUET training path, augmented-scene environment loader, and HM3D pretraining script. This establishes reuse of the DUET implementation family; it does not by itself demonstrate that all image repair, graph construction and R2R configurations can be reconstructed. The inspected source subset does not establish full end-to-end pipeline release. No model execution.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [VLN-DUET/map_nav_src/r2r/agent.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/map_nav_src/r2r/agent.py) — 407–424;469–503: Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [VLN-DUET/pretrain_src/data/dataset.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/pretrain_src/data/dataset.py) — 382–472: Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [VLN-DUET/map_nav_src/r2r/eval_utils.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/map_nav_src/r2r/eval_utils.py) — 6–25: Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [VLN-DUET/pretrain_src/run_r2r_h14.sh](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/pretrain_src/run_r2r_h14.sh) — 2–14: Two-GPU launcher selects CLIP-H14 model and HM3D+MP3D+Gibson pretraining configuration.
- [VLN-DUET-RVR/map_nav_src/reverie/agent_obj.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET-RVR/map_nav_src/reverie/agent_obj.py) — rollout and training loss: REVERIE-specific navigation/object grounding branch was inspected separately from R2R.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **scalevln-B1 — Partially addressed**. Simple synthetic speakers constrain lexical/semantic diversity; scene scaling alone does not certify correct instructions. SRDF improves generated-language diversity and navigation-consistency filtering but introduces verifier selection bias and much greater compute. Evidence: PAPER EXPERIMENT Table 5; AUTHOR limitation p 6. Type: methodological. Follow-ups: SRDF, NavRAG, RoomTour3D. Confidence: high.
- **scalevln-B2 — Partially addressed**. Best results combine data, larger visual encoder and EnvEdit; architecture-controlled comparisons must separate starred and unstarred Table 9 rows. Evidence: PAPER EXPERIMENT Table 9 footnote. Type: insufficiently tested. Follow-ups: SRDF controlled InternViT comparison. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/wz0919/VLN-SRDF; https://github.com/MrZihan/NavRAG
