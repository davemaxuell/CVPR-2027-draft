# WS-MGMap

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Weakly-Supervised Multi-Granularity Map Learning for Vision-and-Language Navigation**. Peihao Chen, Dongyu Ji, Kunyang Lin, Runhao Zeng, Thomas H. Li, Mingkui Tan, Chuang Gan. NeurIPS 2022.

Primary paper: [Weakly-Supervised Multi-Granularity Map Learning for Vision-and-Language Navigation](https://arxiv.org/pdf/2210.07506). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18].

Associated files: WS-MGMap.pdf (Main PDF; 18 pages; SHA256 941b5c78675646c1b77e98d7f2b97408dc254daf5394ad9b381b125f430ae54b).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

WS-MGMap addresses loss of object attributes in category-only maps and weak correspondence between map regions and instruction objects in continuous navigation. It assumes RGB-D, metric projection and simulator-provided training geometry/semantics.

## 3. Mechanism, training, and inference

§3 pp 4–6: project 64-channel latent U-Net segmentation features into a 100×100 grid with 12 cm cells. A hallucination module predicts 27-category semantic occupancy; fuse fine-grained and semantic features by convolutions. A GRU encodes RGB/depth/map history, attends to BiLSTM instruction features, then computes a spatial distribution P̂ over map cells. Weak target P=softmax(P′), P′k=(dmax−dk)/(dmax−dmin), is based on distance dk to the demonstration path, not annotated object locations. Optimize KL(P||P̂). The attended map, text and first GRU state feed a second GRU; linear heads regress a 2D waypoint and scalar progress. A LAW-style expert chooses a point 3m along the shortest route to the nearest reference waypoint; DD-PPO executes low-level actions and STOP fires when progress>.8. L=Lsemantic+10Llocalization+10Lprogress+10Lwaypoint, with squared regression errors; teacher-forcing on 150k augmented examples is followed by DAgger. Frozen segmentation/controller and 2 V100 GPUs over 3 days are reported. Waypoints refresh every 3 steps.

Appendix A–B, pp 14–15, clarifies that fine-grained, semantic and fused maps actually have different tensor resolutions despite the shared m×m notation in the main paper: fine map 64×100×100, semantic map 27×48×48, fused map 256×24×24. Adam uses 2.5e−4; teacher forcing runs 30 epochs, followed by 10 DAgger collections of 5000 trajectories and 4 epochs per collection, with oracle-action probability 0.5^n. The released trainer regresses bounded 2D waypoints using MSE, not categorical primitive-action CE.

## 4. Novelty and evidence

Fuses latent attribute features with hallucinated category maps and path-derived weak localization. The weak target encourages path proximity; calling it verified object grounding would exceed its supervision. Eq 3 describes cosine similarity but prints an unnormalized projected dot product; the inspected attention implementation uses dot-product-style attention, so normalization assumptions should be checked before reusing the equation.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 3 p 8 no map SR 26.5/SPL 21.4; fine-grained 31.6/28.2; semantic 33.3/28.2; fused 38.9/34.3. Appendix Table 7 p 17 no DAgger 24.6 SR versus 38.9 with DAgger: training distribution correction is a major contributor. VERDICT: strong complementary-feature evidence, but weak localization success does not prove correct object identity; comparison must retain data/DAgger/sensor controls.

Appendix C, pp 15–16, scores localization by IoU between the top 10% mass regions of the predicted and weak target distributions:36.7% full, 32.3% without fine-grained map, 6.9% without localization loss. These targets remain weak localization labels, not independent precise boxes. Removing hallucination drops SR 38.9→37.3 in Table 6. Appendix H/p 18 reports attributes in 51% of instructions and ambiguity in 33% of their trajectories, motivating attribute-aware map features.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/PeihaoChen/WS-MGMap](https://github.com/PeihaoChen/WS-MGMap); inspected snapshot **e6af3c40c19010ecf07ea1d62c18c8196e37150d**. Snapshot metadata and source are under `repositories/01/weakmap`.

Inspected projected RGB-feature mapping, semantic/fine-grained fusion, dual recurrent policy, DAgger training and configuration. The source shows map attention and bounded waypoint-regression output; the trainer uses waypoint MSE rather than a primitive-action classifier loss.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [vlnce_baselines/models/mg_map_policy.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/models/mg_map_policy.py) — forward180–251: Projects RGB features, predicts semantic map, concatenates map features, computes text-conditioned map attention, and updates the second recurrent state.
- [vlnce_baselines/common/rgb_mapping.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/common/rgb_mapping.py) — mapping update methods: Builds pose-aligned projected latent-feature memory.
- [vlnce_baselines/dagger_trainer.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/dagger_trainer.py) — collection/training paths: DAgger is a substantive training stage rather than an inference-only map addition.
- [vlnce_baselines/config/CMA_AUG_DA_TUNE.yaml](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/config/CMA_AUG_DA_TUNE.yaml) — training configuration: Defines the augmented/DAgger setup needed for matched reproduction.
- [vlnce_baselines/dagger_trainer.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/dagger_trainer.py) — 505–536: Applies tanh to2D waypoint predictions and weighted MSE to target waypoints, then adds auxiliary map/localization losses and backpropagates.
- [habitat_extensions/measures.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/habitat_extensions/measures.py) — 320–323: Success requires task.is_stop_called and target distance below threshold.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **weakmap-B1 — Partially addressed**. Path proximity is an imperfect proxy for the exact object referred to; salient objects can be off the path. Evidence: KL target §3.3. Type: supervision. Follow-ups: OVER-NAV open-vocabulary detections, REVERIE explicit grounding. Confidence: high.
- **weakmap-B2 — Partially addressed**. Height-collapsed fixed-resolution maps and exact-pose projection lose 3D distinctions and inherit localization noise sensitivity. Evidence: §3.2 map construction. Type: methodological. Follow-ups: VER, GaussianMap, GA-VLN. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/DefaultRui/VLN-VER; https://arxiv.org/abs/2403.17334
