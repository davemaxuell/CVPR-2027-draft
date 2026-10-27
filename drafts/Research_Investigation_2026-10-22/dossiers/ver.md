# VER

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Volumetric Environment Representation for Vision-Language Navigation**. Rui Liu, Wenguan Wang, Yi Yang. CVPR 2024.

Primary paper: [Volumetric Environment Representation for Vision-Language Navigation](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Volumetric_Environment_Representation_for_Vision-Language_Navigation_CVPR_2024_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: VER.pdf (Main PDF; 12 pages; SHA256 b78a102204ba001dbb8d75d1bb016ef290bfb1fa8e9a6032708eec92f62a0593).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Preserve vertical geometry and scene semantics lost by panorama and top-down summaries. Assumes calibrated multi-view images and Manhattan-style room layouts; trains with derived 3D occupancy, layout and object annotations.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3 pp 3–6): learned 3D queries project into 2D feature maps through deformable cross-view attention; cascade 3D deconvolution/refinement grows 15×15×4 initial queries to fine volume. Occupancy focal loss, layout L1/IoU and 3D detection matching losses jointly pretrain geometry (weights 2,.25,.25). Freeze this encoder for navigation. Height-sliced shared transformers combine instruction with voxel tokens, predict a 3D state distribution, average over height, integrate around candidate nodes and normalize to local actions. Pooled candidate pillars form global episodic graph nodes; mix graph action scores with local/backtrack scores. Navigation uses heatmap focal supervision locally, CE globally, MLM/SAP/OG pretraining 100k batch 64 and DAgger 20k batch 8. Reported training 8×24GB RTX4090. Fine labels cover 120×120×35 cells at 0.1m. Predicting volume state is action scoring, not a learned action-conditioned environment rollout.

## 4. Novelty and evidence

Coarse-to-fine volumetric representation and 3D task supervision, compared with 2D metric maps, plus local volume/global graph decision fusion. Extra 3D supervision is part of the contribution and a confounder when asserting architecture-only gains.

## 5. Results and independent verdict

PAPER EXPERIMENT: Table 1 p 6 test R2R 76 SR/66 SPL versus BEVBert 73/62 and DUET 69/59. Additional dense-label generation uses existing MP3D LiDAR labels, so “no additional human annotation” does not mean no additional privileged supervision. Verdict: stronger 3D representation, but freezing a scene encoder and assuming Manhattan geometry restricts adaptation to changing/non-Manhattan scenes. Height grouping reduces full volumetric attention cost while limiting direct cross-height interactions.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/DefaultRui/VLN-VER](https://github.com/DefaultRui/VLN-VER); inspected snapshot **d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a**. Snapshot metadata and source are under `repositories/01/ver`.

The pinned release contains actual volumetric perception training/inference code, but a complete navigation-policy rollout and evaluator were not located in its tree. Inspected the detector, voxel-query head, occupancy/detection losses, dataset evaluator, configuration and README. The README instructs external HAMT ViT feature extraction and downloading MP3D annotations; docs/prepare_dataset.md is inherited nuScenes guidance rather than a complete MP3D pipeline. Reproducing the paper’s navigation results therefore still requires missing integration details. No code was executed.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [projects/mmdet3d_plugin/bevformer/detectors/voxelformer.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/mmdet3d_plugin/bevformer/detectors/voxelformer.py) — 231–326: Loads six horizontal-view precomputed ViT patch feature tensors, ground-truth geometry/occupancy and invokes volumetric losses; HDF5 drops CLS to retain196×768 patch features.
- [projects/mmdet3d_plugin/bevformer/dense_heads/voxelformer_occupancy_head.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/mmdet3d_plugin/bevformer/dense_heads/voxelformer_occupancy_head.py) — 282–350;903–991;1505–1540: Learned voxel queries and positional embeddings; matched detection classification/L1 plus occupancy loss; focal-loss inference adds a0.25 empty-space threshold before argmax.
- [projects/configs/verformer/vocc.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/configs/verformer/vocc.py) — 9–20;55–207;209 onward: 15×15×4 initial volume, 768 channels, 17 object and16 occupancy classes; focal-loss and L1 weights; external feature/annotation paths remain placeholders.
- [projects/mmdet3d_plugin/datasets/mp3docc_dataset.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/mmdet3d_plugin/datasets/mp3docc_dataset.py) — 304–345;485–579: Indoor box evaluation and occupancy IoU using sparse labels/optional invalid-voxel masks; these are perception metrics, not VLN SR/SPL.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **ver-B1 — Shifted into another cost or limitation**. Volumetric resolution, labeling and feature storage cost scale with XYZ; height grouping reduces attention coupling but does not make dense voxels free. Gaussian maps target compact/sparse geometry but add reconstruction costs. Evidence: AUTHOR §3.1–3.5; DEDUCTION. Type: methodological. Follow-ups: 3D Gaussian Map, UAGM. Confidence: high.
- **ver-B2 — Still unresolved in reviewed evidence**. Manhattan layout and static frozen geometry supervision are untested under pose drift, movable objects and non-Manhattan scenes in the inspected navigation table. Evidence: AUTHOR assumptions §3; DEDUCTION. Type: insufficiently tested. Follow-ups: GA-VLN sensor noise. Confidence: medium.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: 
