# DUET

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Think Global, Act Local: Dual-Scale Graph Transformer for Vision-and-Language Navigation**. Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, Ivan Laptev. CVPR 2022.

Primary paper: [Think Global, Act Local: Dual-Scale Graph Transformer for Vision-and-Language Navigation](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Think_Global_Act_Local_Dual-Scale_Graph_Transformer_for_Vision-and-Language_Navigation_CVPR_2022_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 2, 3, 4, 5, 6, 7, 8].

Associated files: DUET.pdf (Main PDF; 11 pages; SHA256 baac593072d1a155a28062330224bddfe34838f118533622100155dd02269b15); DUET (alternate 1).pdf (Alternate / duplicate PDF; 11 pages; SHA256 baac593072d1a155a28062330224bddfe34838f118533622100155dd02269b15); DUET (arXiv).pdf (Alternate / duplicate PDF; 15 pages; SHA256 c88f98b2378f5250d82fc3b44c3e195f5c38be59a91dcfb18f3f47db3ee8685b); DUET (CVPR copy).pdf (Alternate / duplicate PDF; 11 pages; SHA256 baac593072d1a155a28062330224bddfe34838f118533622100155dd02269b15); DUET (extended).pdf (Alternate / duplicate PDF; 15 pages; SHA256 c88f98b2378f5250d82fc3b44c3e195f5c38be59a91dcfb18f3f47db3ee8685b).

Main-paper analysis is supplemented by the explicitly listed associated-file review below. Exact SHA256 duplicates were deduplicated and not counted as additional evidence. DUET (arXiv).pdf: pages [11, 12, 13, 14, 15]; Distinct extended appendix: objectives, data, hyperparameters, backtracking statistics and failure analysis; text inspected. Extended.pdf is exact SHA duplicate.

## 2. Problem and operating assumptions

Discrete graph navigation from language, RGB panoramas and simulator-provided pose/neighbour connectivity; REVERIE/SOON also require final object grounding. The full graph is hidden at inference, but each observation provides navigable neighbours and coordinates. This is a materially privileged sensing/action interface compared with monocular continuous robot control (§3, p 3).

## 3. Mechanism, training, and inference

AUTHOR METHOD: encode panorama views/objects, average them for visited-node features, and accumulate partial views for unseen frontier nodes. A growing graph records visited and navigable locations. The 768-D text encoder feeds a coarse graph transformer and a fine local cross-modal transformer. Graph attention adds a learned scalar transform of pairwise path distance to QKᵀ/√d (Eqs 3–4); this biases topology without a hard connectivity attention mask. Local neighbour logits are mapped to global actions, pooling visited-neighbour scores into a backtrack score (Eq 6). A sigmoid gate combines coarse/fine logits (Eqs 7–8). Visited nodes are normally masked as selected exploration destinations, yet shortest routes to frontier goals may traverse them.

TRAINING: MLM, masked-region classification, single-action prediction and object grounding where available initialize the transformer. Cross-entropy behavioural cloning is supplemented by pseudo-interactive demonstrations from full training graphs (Eqs 9–12), reducing exposure mismatch. The expert selects useful graph destinations; released R2R code also supports an nDTW expert and teacher-forced sequence targets. REVERIE setup uses 100k pretraining iterations, batch 32, two P100s; 20k finetuning iterations, batch 8, one P100 (§4.3 p 6). INFERENCE: grow map, score actions, route through known graph with Floyd shortest paths, cache stop scores and choose final location. Quadratic graph attention/path-distance storage grows with explored nodes; no fixed-memory guarantee is established.

Extended appendix A.1, PDF pp 11–12, supplies the pretraining targets: masked language modeling masks 15% of tokens and minimizes −log p(w_i | W_without_i, trajectory), with a vocabulary classifier on the averaged coarse/fine text features. Masked region classification zeros 15% of final-view features and minimizes KL(P_teacher || P_model) over ImageNet view or Visual Genome object classes. These soft labels supervise semantic reconstruction, not correctness of instruction completion. Appendix B, p 13, gives R2R pretraining 200k steps, batch 64, then 20k fine-tuning steps, batch 8, selecting by validation-unseen SPL; dataset-specific schedules differ.

## 4. Novelty and evidence

Relative to HAMT, adds global frontier selection and explicit topological relations; relative to prior graph navigators, combines graph-aware transformer reasoning with fine local object grounding. Table 1 isolates local/coarse/fused policies, Table 2 graph awareness, Table 3 training losses; a full leaderboard gain alone does not isolate architecture.

## 5. Results and independent verdict

PAPER EXPERIMENT: R2R Table 7 p 8 gives DUET 72 SR/60 SPL on val-unseen and 69/59 test-unseen, versus HAMT-e2e 66/61 and 65/60. Higher SR therefore does not imply more efficient successful paths. REVERIE Table 6 gives val-unseen 46.98 SR, 33.73 SPL, 32.15 RGS. Table 4 p 7 finds synthetic instructions help pretraining but adding them during PID finetuning does not improve this configuration. Verdict: strong explicit exploration and grounding architecture, with an acknowledged backtracking/length tradeoff. OSR–SR measures endpoint selection loss conditional on visited trajectory; it does not identify softmax competition as the unique or primary cause.

Extended appendix C.2–C.4, PDF pp 13–14, explicitly reports backtracking in 48.6% of REVERIE unseen trajectories and 23.2% of R2R unseen trajectories, directly contradicting a no-return interpretation of visited-node masking. REVERIE unseen outcomes are 29.82% wrong room, 23.20% correct room but wrong location, and 46.98% correct location; object grounding conditional on the correct location is 68.43%. These are dataset-specific descriptive categories, not evidence that one softmax denominator causes all stopping failures. Figure 8 shows different outcomes for paraphrases describing the same path. Appendix Table 9 also shows a tradeoff between SR and grounding-weighted SPL as behavioral-cloning weight changes, rather than a uniformly optimal loss weight.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/cshizhe/VLN-DUET](https://github.com/cshizhe/VLN-DUET); inspected snapshot **93e8b233164bc079a6db48b8a0a78d123ec8de41**. Snapshot metadata and source are under `repositories/01/duet`.

INSPECTED CODE: global visited-node masks coexist with shortest-path backtracking. A material paper/code distinction is that the paper specifically describes final return on forced timeout, while this pinned R2R rollout applies maximum cached stop-probability return whenever an episode ends. Sampled training forces stopping at the exact ground-truth endpoint, whereas testing uses predicted stop. These are diagnosis candidates, not demonstrated bugs or demonstrated performance causes. Reproduction needs licensed MP3D assets, precomputed features and checkpoints; imports include legacy NumPy aliases, so current Python compatibility is not established.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [map_nav_src/models/vilmodel.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/models/vilmodel.py) — forward_navigation_per_step, lines 750–817: dual branches, learned fusion, masks and shared backtrack contribution
- [map_nav_src/r2r/agent.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/agent.py) — lines 239–285, 295–303, 394–402, 452–478: teacher policy, shortest-path action execution, stop cache, train/test stop difference, terminal return
- [map_nav_src/models/graph_utils.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/models/graph_utils.py) — FloydGraph.path lines 76–92: routes recursively include intermediary visited nodes
- [pretrain_src/model/pretrain_cmt.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/pretrain_src/model/pretrain_cmt.py) — forward_mlm, forward_mrc, forward_sap: Task heads implement token NLL, soft-label region KL and action classification; these proxy objectives differ from rollout fine-tuning.
- [pretrain_src/data/dataset.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/pretrain_src/data/dataset.py) — 376–466: R2R STOP labels equal index0 at the exact final path node; otherwise labels target the next reference node, with unavailable actions ignored; constructs truncated instruction and graph/panorama inputs.
- [map_nav_src/scripts/run_r2r.sh](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/scripts/run_r2r.sh) — 19–67: Released script fixes DAgger, dynamic fusion, SPL expert, 15 action steps, 200 instruction tokens, AdamW and behavioral-cloning weight.2; checkpoints/data remain external.
- [map_nav_src/r2r/eval_utils.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/eval_utils.py) — 6–25: DTW uses graph shortest distances; nDTW=exp(−DTW/(3×reference_length)); SDTW multiplies endpoint success.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **duet-B1 — Still unresolved in reviewed evidence**. Stop scores are normalized over a changing action set and then compared across time; cached values also come from different histories. Such values need not be temporally calibrated. Test this with fixed logits/candidate perturbations and held-out final-node selection, not an OSR gap alone. Evidence: INSPECTED CODE agent.py 394–402,472–478; DEDUCTION. Type: methodological. Follow-ups: ScaleVLN, SRDF. Confidence: high for code, medium for practical impact.
- **duet-B2 — Partially addressed**. Discrete routing and exact pose/neighbour access omit continuous obstacle avoidance and localization error. BEVBert/ETPNav add continuous control but inherit waypoint/sensor assumptions. Evidence: AUTHOR TASK §3 p 3; PAPER EXPERIMENT Table 7. Type: methodological. Follow-ups: BEVBert, ScaleVLN R2R-CE, ETPNav. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/wz0919/ScaleVLN; https://github.com/wz0919/VLN-SRDF; https://github.com/MarSaKi/ETPNav

## Coordinator reconciliation: isolated execution and recovery integration

The execution statement above describes assignment 01's paper/source review. The coordinator subsequently executed only the reviewed `FloydGraph` class on a four-node synthetic graph, and assignment 10 executed an AST-extracted `_teacher_action_r4r` with stipulated histories. These are [isolated component checks](../experiments/README.md), not checkpoint or benchmark reproduction.

The independent integration audit confirms three different contracts: selectable global destinations can be masked if visited; teacher/sample rollout terminates at the reference endpoint (`agent.py:453–455`); and termination can append travel to the highest cached STOP node (`471–478`). A proposed semantic recovery task must control the candidate interface, termination and cached return together for every comparator. Intermediate route nodes do not automatically provide new visual observations: `make_equiv_action` records the route but resets the simulator at its destination (`287–303`). These findings establish integration requirements, not errors in the original endpoint-defined benchmark. See [assignment 10](../evaluations/10_feasibility.md) for pinned sources and the [source-check record](../experiments/10/source_check_results.json) for exact transformations/stubs.
