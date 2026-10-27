# IEVE

Instance-aware Exploration-Verification-Exploitation for Instance ImageGoal Navigation

Xiaohan Lei; Min Wang; Wengang Zhou; Li Li; Houqiang Li. CVPR 2024. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [IEVE.pdf](../../Paper/IEVE.pdf). Pages reviewed: ['3–8'].

## Problem and operating conditions

Find the exact object instance shown in a goal image, using RGB-D and pose in an unknown environment. This is Instance ImageGoal Navigation, not language-conditioned route following. Camera height and field of view may differ between goal and current observation.

## Mechanism, training, and inference

A classifier identifies the central goal-image object category. RGB-D detections are projected to point clouds, voxelized, pooled, and merged into a 2+N-channel global semantic map. DISK descriptors and LightGlue matching yield instance match count omega. Distance-dependent low/high thresholds select exploration, verification, or exploitation: weak evidence explores; ambiguous evidence approaches for a better view; sufficient matching commits to the candidate; a close poor match rejects it. A learned exploration policy emits a waypoint every ten local steps, with area-coverage and target rewards. FMM generates local motion; low-level actions include continuous linear/angular velocity, camera pitch, and stop, bounded by 35 cm translation and 60 degree rotation per frame. Training uses HM3D scenes and a constructed instance re-identification dataset. Evaluation success requires proximity within one metre and oracle visibility, with 1,000 validation episodes.

## Novelty and closest predecessors

The explicit approach-to-verify stage adjusts evidence acquisition to distance-dependent instance matching quality. It separates category exploration from instance commitment instead of treating a fixed matching threshold as sufficient at all viewpoints. It does not supply a conformal or probabilistic correctness guarantee.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 Mod-IIN SR/SPL 0.561/0.233 versus Swin/Mask IEVE 0.684/0.241 and InternImage version 0.702/0.252. Table 2 controlled EVE versus exploration/exploitation improves 0.590/0.221→0.666/0.235 with the same Mask classifier/perception and 0.620/0.216→0.684/0.241 with Swin classification. Oracle classification/segmentation reaches 0.850/0.399, leaving a nonzero planning gap. Classification changes interact with the segmentation backbone, so the full gain cannot be attributed solely to verification. Re-identification experiments show fixed-threshold true positives degrade with greater distance, supporting the motivation to approach before rejecting.

SUPPLEMENT CHECK: Appendix pp. 1–2 factorizes uncertainty under specified distance/match-count assumptions, without a general history-dependent guarantee. Table 2 normalizes TP+FN and TN+FP separately, while the printed algebra also sums all four as one; those conventions should not be conflated. The supplement describes a linear switching boundary from 60 matches at 2 m to 100 at 4 m, whereas the inspected active source path uses piecewise thresholds around 3 m. Page 4 collision handling marks the obstructed front cell untraversable and previously visited cells traversable. These scoped heuristics do not establish safe verification under localization errors.

## Implementation and reproducibility

INSPECTED CODE: agents/instance_exp.py loads DISK/LightGlue and implements the hand-crafted instance_discriminator switch. At 629–660, >100 matches confirms a target; at goal distance<150 map units, >60 matches confirms while lower counts reject; farther candidates become temporary verification goals. An alternate instance_discriminator_auto path uses a learned switch, so its behavior must not be conflated with the hand-crafted path. main.py constructs the exploration PPO agent, loads policy weights and records success/SPL/distance at evaluation. The controller implements velocity actions; README explicitly notes the release was updated to match the paper’s velocity-control setting. No weights or Habitat evaluation were executed. Image-instance matching and threshold selection do not imply instruction-level error diagnosis.

Inspected release reference: [https://github.com/XiaohanLei/IEVE at 398414ab78cda1b7a77a63f9cae0c30ddd1a82fa](https://github.com/XiaohanLei/IEVE/tree/398414ab78cda1b7a77a63f9cae0c30ddd1a82fa). File-level inspection is restricted to the paths listed below.

- README.md
- agents/instance_exp.py:73–74,426–486,592–661
- main.py:246–278,396–402

## Specific bottlenecks

**ieve-perception: Distance-aware verification reduces rejection errors but remains dependent on viewpoint-sensitive instance features.**

Conditions: Textureless objects, camera mismatch, occlusion, repeated similar instances, or inaccurate category segmentation. Cause: Match counts are heuristic evidence and both the map and verification depend on visual perception. Evidence: PAPER EXPERIMENT: Tables 1–2 oracle perception gap and re-identification distance analysis.

Type: methodological. Status: Partially addressed. Confidence: High within the specified evidence scope. Follow-ups: ['AIUTA']

## Sources and limitations

- [Supplied main paper](https://openaccess.thecvf.com/content/CVPR2024/html/Lei_Instance-aware_Exploration-Verification-Exploitation_for_Instance_ImageGoal_Navigation_CVPR_2024_paper.html)
- [Official supplementary or complete author version; reviewed scope listed in associated material](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Lei_Instance-aware_Exploration-Verification-Exploitation_for_CVPR_2024_supplemental.zip)
- Four-page official supplement reviewed; accompanying video not reviewed. Piecewise released switching differs from the supplementary linear boundary; no measured impact or robot reproduction.
- Associated material: {"file": "evidence/04/supplements/ieve_supp.txt", "pages_read": "1–4", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
