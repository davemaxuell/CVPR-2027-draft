# GridMM

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**GridMM: Grid Memory Map for Vision-and-Language Navigation**. Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, Shuqiang Jiang. ICCV 2023.

Primary paper: [GridMM: Grid Memory Map for Vision-and-Language Navigation](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_GridMM_Grid_Memory_Map_for_Vision-and-Language_Navigation_ICCV_2023_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 9].

Associated files: GridMM.pdf (Main PDF; 12 pages; SHA256 b51df7b494c06ad3f7cf57e3c8d2933da10186aecf3b7578235b38d348733147).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Retain global spatial relationships across previously visited observations without collapsing history to recurrence or restricting geometric reasoning to a local metric window. Uses RGB-D, pose and panoramic observations; CE uses pretrained waypoint proposals.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3 pp 4–6): store CLIP-B/32 grid features with absolute projected coordinates. At every action rotate/translate all history to the current egocentric frame. Set square extent to twice the maximum absolute coordinate and divide into fixed N×N cells. Within each cell, score feature-token relevance, max over instruction words, softmax over features, and weighted-pool into a 768-D cell token (Eqs 6–10). Combine map and trajectory memories, then cross-attend observation/history queries to instruction/history/map features. DUET-style dynamic fusion scores local views and global waypoints. Pretrain MLM/MVM/SAP plus historical-environment reasoning (HER), which predicts action without current panorama; DAgger uses graph shortest-route pseudo-expert. Inference retains all historical feature/coordinate pairs, so fixed map tokens do not mean fixed raw-history memory.

## 4. Novelty and evidence

Compared with BEVBert, global rather than local metric coverage and instruction-conditioned per-cell pooling; compared with DUET, explicit continuous geometry and history-only auxiliary prediction. HER is not a learned future simulator.

## 5. Results and independent verdict

PAPER EXPERIMENT: Table 2 p 7 R2R test 73 SR/62 SPL. CE Table 4 p 8 contrasts shared-waypoint DUET 42/36 test with GridMM 46/39. Table 5 CE map ablation 49.05 SR/40.99 SPL versus no-map 45.19/37.82. Table 7 increases N from 8 to 14 to 20 and SPL 39.49→40.99→42.52, exposing a resolution/compute tradeoff. Main discrete/CE scores should not be merged: camera 90-degree versus panoramic and waypoint availability differ explicitly in Table 4 footnote.

Section 4.5/p 9 reports the map footprint growing from roughly 10m to 20m and maximum per-cell feature counts exceeding 600 late in episodes. A fixed cell count therefore reduces physical spatial resolution as explored extent grows, while feature aggregation grows denser. The authors explicitly leave multi-floor handling open. These are more concrete constraints than a generic statement that mapping is expensive.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/MrZihan/GridMM](https://github.com/MrZihan/GridMM); inspected snapshot **713139f0be045f339bd041a7caefe10b5c4fc6eb**. Snapshot metadata and source are under `repositories/01/gridmm`.

INSPECTED CODE: CE GridMap stores geometry and constructs maps in getGlobalMap, and model forward_navigation_per_step consumes grid features/indexes alongside graph/observation embeddings. The CE implementation adds local/global logits on candidate space; do not assume exact discrete global-action semantics from this branch alone.

agent.py 364–368 contains commented-out STOP reweighting code, but the active loss at 370 is ordinary candidate-action criterion. Commented experiments are not evidence of an executed or evaluated algorithm.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [VLN_CE/vlnce_baselines/models/Policy_ViewSelection_GridMap.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/VLN_CE/vlnce_baselines/models/Policy_ViewSelection_GridMap.py) — get_rel_position1263;get_gridmap_pos_fts1321;getGlobalMap1377: depth-driven map construction
- [VLN_CE/vlnce_baselines/models/gridmap/vilmodel.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/VLN_CE/vlnce_baselines/models/gridmap/vilmodel.py) — forward_navigation_per_step1419–1595: map-conditioned action path and local/global masked logits
- [map_nav_src/r2r/agent.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/map_nav_src/r2r/agent.py) — 330–370;401–448: Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/pretrain_src/data/dataset.py) — 636–751: Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [map_nav_src/r2r/eval_utils.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/map_nav_src/r2r/eval_utils.py) — 6–25: Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [map_nav_src/scripts/run_r2r.sh](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/map_nav_src/scripts/run_r2r.sh) — 19–67: Default script uses four GPUs, batch 4 and15 action steps; test invocation and training command/checkpoint selection must be distinguished.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **gridmm-B1 — Partially addressed**. Fixed cell count with expanding physical extent merges increasingly distant objects into cells; raw history grows with steps. Long-horizon gains must be evaluated against both spatial resolution and cumulative memory. Evidence: AUTHOR Eqs 2–4; Table 7; DEDUCTION. Type: methodological. Follow-ups: VER, 3D Gaussian Map. Confidence: high.
- **gridmm-B2 — Uncertain because evidence is insufficient**. Instruction-max relevance pooling can suppress evidence for negation/order when salient nouns dominate; no counterfactual grounding experiment in inspected ablations establishes compositional fidelity. Evidence: DEDUCTION Eqs 6–9. Type: insufficiently tested. Follow-ups: NavNuances. Confidence: medium.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: 
