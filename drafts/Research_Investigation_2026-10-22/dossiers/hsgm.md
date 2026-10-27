# HSGM

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation**. Kailing Li, Tianwen Qian, Lijin Yang, Yuqian Fu, Jingyu Gong, Xiaoling Wang, Liang He. CVPR 2026.

Primary paper: [Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Bridging_the_2D-3D_Gap_A_Hierarchical_Semantic-Geometric_Map_for_Vision_CVPR_2026_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: HSGM.pdf (Main PDF; 10 pages; SHA256 4cc9e491c3ff74f985a472ffc88330d64fd29f477fc8a4af2b7d21ea6b0c347d).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Training-free task adaptation using a pretrained VLM to choose semantically meaningful, geometrically reachable waypoints. Inputs include front/left/right RGB-D and pose; GPT-5 and YOLO-E remain pretrained components, so training-free does not mean no learned perception or no API cost.

## 3. Mechanism, training, and inference

§3 pp 3–6: backproject RGB-D into a geometric point cloud, segment open-vocabulary instances with YOLO-E, and merge consistent 3D instances. Build dense global and sparse local waypoint graphs after cylinder-occupancy, distance, obstacle-clearance and reachability checks; stairs use surface-normal/tilted-plane filtering. Rasterize geometry, object categories, path and completed-subtask locations into BEV, and annotate current image with waypoint IDs. A VLM decomposes instruction into ordered subtasks with pending/in-progress/done states. Each subtask has at most 3 motion directives and a stated termination condition. The VLM selects a waypoint/turn/STOP; A* over the observed graph minimizes sum of edge Euclidean lengths and converts the path into rotations/forward motions. The paper specifies two consecutive STOP predictions for final completion; the released code retains the first STOP flag across intervening motion, as detailed below. A timeout returns to the subtask start for another attempt. No VLN training objective is optimized; runtime updates maps and prompt history.

## 4. Novelty and evidence

Geometric waypoint validation, rasterized hierarchical maps, structured subtask management and decoupled A* control are integrated into a zero-shot VLM pipeline. Double-confirmed stopping, semantic progress bookkeeping, and timeout backtracking are explicit prior art for recovery proposals.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 full R2R-CE val-unseen 47.9 SR/32.8 SPL; RxR-CE 41.8/25.1/54.9 nDTW on 500 random English episodes. Ablations use a separate 300-episode R2R subset: full 51 SR, without subtasks 42.1, without structured CoT 34.0. Table 4 reports backtracking triggered 18.3%/19.0%, recovered-episode success 30.8%/26.8%; this is conditional success, not causal recovery gain against a matched no-backtracking control. The A* ablation uses straight-line movement, so its gain does not isolate semantics from collision handling. VERDICT: strong practical baseline with expensive proprietary inference and privileged sensors; matched VLM/backbone/episode comparisons are essential.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/Teacher-Tom/HSGM_public](https://github.com/Teacher-Tom/HSGM_public); inspected snapshot **e2df6e70df869af57d77a32ba658fba04e6814eb**. Snapshot metadata and source are under `repositories/01/hsgm`.

Official repository pinned and source downloaded for high-level agent, map construction, path planning, experiment runner and configuration. Detailed function evidence is recorded in the implementation appendix; no API calls or simulator episodes were executed.

CODE FINDING: src/agent/agent.py 1082–1094 records first_stop and then second_stop; first_stop is initialized at 1070 and is not reset after movement anywhere in this pinned file. Thus the release does not enforce consecutivity if movement intervenes between the two STOP predictions. The 20-step no-movement guard 1692–1701 can force second_stop independently. Rollback 1742–1816 plans a path to the subtask origin rather than erasing executed travel. run_experiments.py 445–529 computes success and SPL from the final position without an explicit STOP predicate; the loop 1030–1084 saves at second_stop or the step budget, and on budget termination overwrites finish_status but retains distance-based goal_reached/SPL. This is a protocol-comparability caveat, not proof that published numbers used this exact snapshot or are incorrect.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [src/agent/agent.py](https://github.com/Teacher-Tom/HSGM_public/blob/e2df6e70df869af57d77a32ba658fba04e6814eb/src/agent/agent.py) — 804–979;1082–1094;1646–1884: Instruction subtask state, persistent first/second STOP flags, no-movement guard, replanning and timeout rollback.
- [src/run_experiments.py](https://github.com/Teacher-Tom/HSGM_public/blob/e2df6e70df869af57d77a32ba658fba04e6814eb/src/run_experiments.py) — 445–529;979–1120: Final-distance success/SPL and Euclidean fast-DTW; stopping/saving occurs on second_stop or maximum steps.
- [config/vlnce_test.yaml](https://github.com/Teacher-Tom/HSGM_public/blob/e2df6e70df869af57d77a32ba658fba04e6814eb/config/vlnce_test.yaml) — camera and simulator configuration: Sensor FOV135 degrees, pitch−30 degrees, height 1.2m, maximum 200 steps and radius 0.1m are part of the released operating conditions.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **hsgm-B1 — Uncertain because evidence is insufficient**. Repeated self-reported completion is not independent verification because the same model can repeat a correlated error. The paper proposes consecutive STOP confirmation, while the inspected release does not reset the first STOP flag after motion. Evidence: Paper §3.4; INSPECTED CODE src/agent/agent.py 1070–1094; deduction about correlated errors. Type: verification. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: medium.
- **hsgm-B2 — Uncertain because evidence is insufficient**. Geometric safety holds only for the observed map and sensor/model assumptions; mapping misses and pose drift can invalidate reachability. Evidence: §3.3 cylinder/edge checks. Type: sensor/control. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.
- **hsgm-B3 — Shifted into another cost or limitation**. Runtime API cost and sampled RxR evaluation limit reproducibility/comparability. Evidence: §4.1 GPT-5 and 500/300-episode subsets. Type: compute/protocol. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/Teacher-Tom/HSGM_public
