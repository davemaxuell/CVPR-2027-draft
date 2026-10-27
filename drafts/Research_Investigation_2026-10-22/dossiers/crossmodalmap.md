# CM2

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Cross-Modal Map Learning for Vision and Language Navigation**. Georgios Georgakis, Karl Schmeckpeper, Karan Wanchoo, Soham Dan, Eleni Miltsakaki, Dan Roth, Kostas Daniilidis. CVPR 2022.

Primary paper: [Cross-Modal Map Learning for Vision and Language Navigation](https://openaccess.thecvf.com/content/CVPR2022/papers/Georgakis_Cross-Modal_Map_Learning_for_Vision_and_Language_Navigation_CVPR_2022_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: CM2.pdf (Main PDF; 11 pages; SHA256 db209cd240bb0a303d6a96dbdc6c96e6c0f67d3bdda68f8d92da66fbbe7b7929).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

CM² addresses VLN-CE with a 90-degree RGB-D view, ground-truth pose, noiseless 0.25m forward/15-degree turns and STOP. It predicts a language-conditioned local semantic map and a complete waypoint sequence, then delegates physical control to DD-PPO. Ground-truth pose and pretrained controller experience are material resource assumptions.

## 3. Mechanism, training, and inference

§3 pp 3–5: project depth into a 192×192 occupancy grid with 5 cm cells; separately segment RGB. BERT produces M×128 text features, while a truncated ResNet 18 encodes the map into N×128 tokens. Cross-attention H=softmax((YWq)(XWk)ᵀ/√128)XWv uses map queries and text keys/values. Two stacked U-Nets predict occupancy then 27-category semantics, including unseen regions, trained with pixel CE against simulator semantic maps. A second cross-modal network predicts k waypoint heatmaps P̂∈R^(k×u×v) and traversed-waypoint labels, conditioned on the start heatmap. Lwp=Σi bi||P̂i−Pi||² plus waypoint-coverage CE, where bi masks waypoints outside the local map. Total loss combines map and waypoint losses in one backward pass. At inference choose the waypoint after the one closest to current pose; DD-PPO reaches it. Stop when close to the final predicted waypoint and heatmap confidence exceeds a threshold. This is language-conditioned map hallucination plus replanning, not an action-conditioned recurrent future-state simulator.

## 4. Novelty and evidence

Language informs both unobserved map completion and path prediction. Table 3 isolates map attention, supporting better map/waypoint accuracy. Unlike pre-exploration topological planners, CM² builds predictions online, but it assumes true pose and privileged map-label supervision during training.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 val-unseen SR 34.3/SPL 27.6, versus LAW 35/31; ground-truth local maps give 37/30.6. Table 3 p 7 map attention improves IoU 21.2→28.3 and correct-waypoint 71.1→76.5. Table 4 changes stop threshold 1.5→0.5m: val-seen SR 49.5→52.8 while SPL 45.1→41.8, demonstrating an explicit stopping/efficiency tradeoff. VERDICT: credible representation evidence, but stronger sample-efficiency claims should count pretrained DD-PPO and segmentation training; oracle-map results are not deployable performance.

Section 4.2, PDF p 8, doubles the local map from 192 to 384 cells per side in the ground-truth-map experiment and reports gains of 5.6 SR points and 4.9 SPL points. This supports a concrete spatial-coverage bottleneck, although the oracle-map ablation does not establish the same gain for a learned hallucinated map.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/ggeorgak11/CM2](https://github.com/ggeorgak11/CM2); inspected snapshot **4730cfdf49c468d6632fff03ecc9cee9955e7dee**. Snapshot metadata and source are under `repositories/01/crossmodalmap`.

Inspected map/path model losses, map projection utilities, trainer, evaluator/controller interface, dataset labels and options. The released coverage head uses two-class cross entropy, clarifying the compact paper notation.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [models/vln_goal_predictor.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/models/vln_goal_predictor.py) — coverage_loss89–102; position_loss106–126: Two-class coverage CE and visibility-masked heatmap MSE.
- [models/semantic_grid.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/models/semantic_grid.py) — spatial-transform and map update methods: Map construction uses pose-aligned spatial transformations.
- [datasets/util/map_utils.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/datasets/util/map_utils.py) — projection and heatmap helpers: Training labels and observations use explicit metric coordinates.
- [trainer_vln.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/trainer_vln.py) — training loss aggregation: Waypoint and map objectives are optimized together.
- [trainer_vln.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/trainer_vln.py) — 24–75: Offline dataset, Adam optimization, and summed waypoint-position plus coverage losses.
- [tester.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/tester.py) — 260–298: Stops by predicted goal distance or time limit; local-policy STOP is replaced by random motion, and final metric helper is called with stop_signal=True. This protocol must be replicated before comparisons.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **crossmodalmap-B1 — Partially addressed**. Language-conditioned hallucination may invent unseen semantics and local 9.6m coverage excludes distant goals. Evidence: §3.4 and local map dimensions. Type: methodological. Follow-ups: WS-MGMap, BEVBert, HSGM. Confidence: high.
- **crossmodalmap-B2 — Uncertain because evidence is insufficient**. True-pose projection and fixed pretrained controller hide localization/control robustness costs. Evidence: §3.2 ground-truth pose assumption. Type: operating assumption. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/PeihaoChen/WS-MGMap; https://github.com/Teacher-Tom/HSGM_public
