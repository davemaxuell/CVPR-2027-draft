# 3D Gaussian Map

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation**. Jianzhe Gao, Rui Liu, Wenguan Wang. ICCV 2025.

Primary paper: [3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_3D_Gaussian_Map_with_Open-Set_Semantic_Grouping_for_Vision-Language_Navigation_ICCV_2025_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: 3D Gaussian Map.pdf (Main PDF; 11 pages; SHA256 55d7e8e3d9b7d765429fb98f1e6d8d31d90aba80ee1fc9e95a92542ffcabdc31).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Represent both geometry and open-set semantics for discrete VLN. Requires panoramic RGB-D, camera intrinsics/poses and graph candidate nodes; 3D representation alone does not turn this into continuous control.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3 pp 3–5): back-project depth pixels to pseudo-lidar points, initialize anisotropic Gaussians with position, covariance, opacity and color, then refine through differentiable RGB/depth rendering. SAM2 masks identify coherent regions; CLIP produces 512-D region embeddings, mean-pooled to one scalar per region. A rendered scalar semantic attribute on each Gaussian is optimized to match these targets. Action features concatenate 3-D position, 3-D color and 1-D semantic code into 7-D vectors. Scene-wide, view-restricted and instance-restricted pooling feed language-conditioned transformer heads; map their scores to candidate waypoints and combine with traditional 2D action scores and topological memory. Navigation pretrains MLM/SAP/OG, then DAgger; map optimization uses RGB L1/SSIM, depth L1 and semantic L1. Reported 100k pretrain iterations batch 64, 25k finetuning batch 8; inference constructs map at each waypoint. Claimed 0.07s map construction is author-reported and should not be conflated with total SAM2/CLIP/navigation latency.

## 4. Novelty and evidence

Explicit Gaussian geometry plus semantic grouping and scene/view/instance action heads, relative to VER voxels and BEVBert top-down cells. It reuses a DUET-referenced topological/decision framework rather than establishing a wholly independent architecture (§3.5).

## 5. Results and independent verdict

PAPER EXPERIMENT: Table 2 p 6 reports R2R test 75 SR/65 SPL, versus BEVBert 73/62; REVERIE Table 1 val-unseen 53.59 SR/26.74 RGSPL and test 52.93/25.76. Test navigation SR is below some contemporary methods even when grounding improves. The evidence supports task utility, not retention of arbitrary CLIP semantics. DEDUCTION: mean-pooling a 512-D embedding is non-injective, so distinct vectors can share the same semantic scalar. Eq 12 sums three normalized distributions but labels the result [0, 1]; absent normalization it need not be a probability distribution. This notation/implementation ambiguity cannot be resolved without code.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/Gaozzzz/3D-Gaussian-Map-VLN](https://github.com/Gaozzzz/3D-Gaussian-Map-VLN); inspected snapshot **3a42711d7449280d4eda89d257e5891947a1ced2**. Snapshot metadata and source are under `repositories/01/gaussianmap`.

Official pinned repository contains README only; no released mapping, semantic compression, loss or action-fusion code was available to verify equations or latency accounting.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

**Execution status:** No model source inspection is claimed beyond the explicit scope above. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **gaussianmap-B1 — Uncertain because evidence is insufficient**. Open-set scalar semantic compression irreversibly discards distinctions; test with pairs of regions with equal/near-equal feature mean but different navigation relevance. UAGM changes semantic representation/uncertainty, but released implementation is also unavailable. Evidence: AUTHOR Eq 7–8 p 4; DEDUCTION. Type: methodological. Follow-ups: UAGM. Confidence: high for information loss, unknown behavioral impact.
- **gaussianmap-B2 — Still unresolved in reviewed evidence**. RGB-D/pose assumptions and segmentation/rendering overhead remain; no observed closed-loop physical robot robustness established by discrete benchmark gains. Evidence: AUTHOR task §3 and runtime §3.5. Type: insufficiently tested. Follow-ups: GA-VLN. Confidence: medium.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/Gaozzzz/Uncertainty-Aware-VLN
