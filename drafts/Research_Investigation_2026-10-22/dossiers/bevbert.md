# BEVBert

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**BEVBert: Multimodal Map Pre-training for Language-guided Navigation**. Dong An, Yuankai Qi, Yangguang Li, Yan Huang, Liang Wang, Tieniu Tan, Jing Shao. ICCV 2023.

Primary paper: [BEVBert: Multimodal Map Pre-training for Language-guided Navigation](https://arxiv.org/pdf/2212.04385). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17].

Associated files: BEVBert.pdf (Main PDF; 17 pages; SHA256 c09af750b387aa2b8289a7c410e1b87822f055df2929b6cd396878324449b4b0).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Panorama-only pretraining duplicates observations and makes local spatial relations implicit. BEVBert combines a global topology with a fine local ground-plane representation, assuming depth and pose or estimated depth.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3): CLIP patch features projected using depth/pose into a 21×21 metric grid of 0.5m cells, centered on the current agent; aggregate nearby visited nodes within a topology-order neighbourhood. Keep global visited/current/ghost graph nodes with panorama features. DUET-style graph-aware attention handles topology; a second cross-modal transformer handles cell features with polar location and navigability embeddings. Hybrid single-action prediction maps local cells to global node scores and gates local/global outputs (Eqs 3–5). MLM predicts masked words; masked semantic imagination (MSI) predicts 40-category multilabel semantics in masked cells with binary cross entropy (Eq 6). MSI is a training auxiliary, not an online future-state rollout. Alternate teacher/student forcing after pretraining. Pretrain 100k batch 64 then finetune 40k batch 16, 4 A100s, respectively about 10/20 hours. CE variant adds waypoint prediction and topology construction.

Appendix B, pp 13–15, samples MLM:HSAP:MSI at 5:5:1 for route-following tasks; REVERIE omits MSI and uses MLM:HSAP:MRC:OG at 1:1:1:1. Fine-tuning minimizes λL_teacher + L_student, with optional object grounding; λ=.2 for R2R/CE/REVERIE but .8 for RxR, whose non-shortest paths use nDTW-oriented pseudo-expert labels. Transformer text-branch attention is cut during fine-tuning to control expanding rollout memory. The estimated-depth ablation trains RedNet on training houses and retrains BEVBert with those estimates; it is not a zero-cost replacement at inference.

## 4. Novelty and evidence

Joint local metric/global topological pretraining, spatial embeddings and MSI. It explicitly inherits DUET GASA; hybrid representation differs from simply appending raw depth. Table 5 compares topology, metric and hybrid maps under sensed/estimated depth.

## 5. Results and independent verdict

PAPER EXPERIMENT: R2R Table 1 p 6 test 73 SR/62 SPL versus DUET 69/59. REVERIE test 52.81 SR/22.09 RGSPL is nearly DUET 52.51/22.06 despite larger validation gains. Table 5 p 7 hybrid estimated-depth 74.67 SR/63.63 SPL versus sensed 74.88/63.60 suggests exact depth is not essential on that test. Fig 4 spatial/numerical-token slices degrade with increasing complexity even for BEVBert. Verdict: improves spatial representation, while complexity-conditioned language understanding remains partial; leaderboard comparisons must control CLIP, augmentation and sensor interfaces.

Appendix E, pp 15–17, explicitly records early deviations followed by excessive unsuccessful backtracking, loss of instruction-state tracking, ordinal-room ambiguity, and stopping at a kitchen counter after skipping a required turn. These failures persist despite a metric map and masked semantic imagination. Table 7, p 8, gives short-term-transformer FLOPs 4.5G for 11×11 cells, 15.2G for 21×21, and 32.7G for 31×31, showing that map refinement incurs a measurable cost.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/MarSaKi/VLN-BEVBert](https://github.com/MarSaKi/VLN-BEVBert); inspected snapshot **ee40002e9eb75ca6d587e8746990e91b0463f5ca**. Snapshot metadata and source are under `repositories/01/bevbert`.

INSPECTED CODE confirms depth-to-world projection, local BEV embeddings and cross-entropy fused action losses. Inspected CE pretraining source hard-codes bev_dim=11 (vilmodel.py 715), whereas paper general/discrete setting describes 21; this is a variant/config difference requiring explicit reconciliation, not automatically a bug.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [bevbert_ce/pretrain/pretrain_src/model/bev_utils.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/bevbert_ce/pretrain/pretrain_src/model/bev_utils.py) — pixel_to_world_mapping200;PointCloud.forward349;project_bev382: depth/pose projection and BEV pooling
- [bevbert_ce/pretrain/pretrain_src/model/vilmodel.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/bevbert_ce/pretrain/pretrain_src/model/vilmodel.py) — LocalBEVEncoder581–609;GlocalTextPathCMT712–730: cell embedding and CE grid setting
- [bevbert_ce/pretrain/pretrain_src/model/pretrain_cmt.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/bevbert_ce/pretrain/pretrain_src/model/pretrain_cmt.py) — 347–350: global/local/fused action cross-entropy losses
- [map_nav_src/r2r/agent.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/map_nav_src/r2r/agent.py) — 539–555;585–633: Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/pretrain_src/data/dataset.py) — 397–440;471–578: Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [map_nav_src/r2r/eval_utils.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/map_nav_src/r2r/eval_utils.py) — 6–25: Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **bevbert-B1 — Shifted into another cost or limitation**. Fixed local metric extent limits detailed memory of distant regions; GridMM keeps global history but dynamic rescaling sacrifices resolution as coverage expands. Evidence: AUTHOR §3.1; GridMM §3.2. Type: methodological. Follow-ups: GridMM, VER, GA-VLN. Confidence: high.
- **bevbert-B2 — Partially addressed**. MSI fills masked semantic cells without proving action-conditioned future prediction or calibrated uncertainty; unseen-region hallucination remains insufficiently tested. Evidence: AUTHOR Eq 6; DEDUCTION. Type: insufficiently tested. Follow-ups: HNR, UAGM. Confidence: medium.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: 
