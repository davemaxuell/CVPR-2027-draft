# UAGM

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Uncertainty-Aware Gaussian Map for Vision-Language Navigation**. Jianzhe Gao, Rui Liu, Yuxuan Xu, Tongtong Cao, Yingxue Zhang, Zhanguang Zhang, Sida Peng, Yi Yang, Wenguan Wang. ICLR 2026 (PDF header).

Primary paper: [Uncertainty-Aware Gaussian Map for Vision-Language Navigation](https://arxiv.org/abs/2605.26503v1). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23].

Associated files: UAGM.pdf (Main PDF; 23 pages; SHA256 a4a90b10da489a4fd6b88eedcc73ceeae413732ce55d511b5e25494de73dff3c).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Use spatially localized uncertainty to guide RGB-D graph navigation under ambiguous geometry, semantics and appearance. Requires differentiable Gaussian reconstruction and known camera transforms; does not directly estimate calibrated action-error probabilities.

## 3. Mechanism, training, and inference

AUTHOR METHOD (§3 pp 4–6): Semantic Gaussian Map initializes position 3, scale 3, quaternion 4, opacity 1, color 3 and semantic 3 attributes. Prune small/transparent Gaussians. Learn variational position/scale perturbations with an ELBO under Gaussian position and scale-relative uniform priors; aggregate posterior standard deviations for geometric uncertainty. Repeat for semantic offsets. Appearance score is logdet of each Gaussian block of an image-reconstruction Fisher/Gauss–Newton approximation (Eq 9). Concatenate three uncertainty scalars with 17 original attributes into 20-D Gaussian vectors, project to 768 dimensions, aggregate and fuse with instruction through transformer action scores/topological memory. Pretrain MLM/SAP/OG for 100k iterations batch 64, Adam 1e-4; DAgger 25k batch 8 learning rate 1e-5. Inference reconstructs map and uncertainty before predicting action.

Appendix B.1–B.2, PDF p 17, explicitly retains DUET’s 2D branch alongside the added 3D value branch. The 768-dimensional visual and text representations produce candidate distributions; pretraining uses masked-token NLL and expert-action NLL, adding target-object NLL for REVERIE. DAgger supplies shortest-path pseudo-expert labels. Appendix C.4 specifies a per-Gaussian block-diagonal, first-order-gradient outer-product approximation for Fisher information. A finite, well-conditioned log determinant still requires an explicit regularization/rank convention, which the unavailable code prevents checking.

## 4. Novelty and evidence

Extends Gaussian scene representations with separate geometry/semantic perturbation variability and appearance sensitivity. Table 4 compares DUET, geometry-only, uncertainty-only and joint representations; Table 6 ablates uncertainty components. This is uncertainty-informed representation learning, not a proven Bayesian navigation policy.

## 5. Results and independent verdict

PAPER EXPERIMENT: R2R Table 2 p 8 gives test 76 SR/66 SPL, equal to VER in these rounded metrics; val-unseen 78/66. Table 4 p 9 lists REVERIE full-model SR 53.37, whereas headline Table 1 p 7 lists 56.37 with the same 37.65 RGS/27.01 RGSPL; preserve this internal discrepancy rather than choosing one silently. Table 5 pruning increases FPS from 11.2 to 15.5 at 42k primitives, but aggressive 35k harms SR. DEDUCTION: logdet Fisher can be negative or −∞ and is a sensitivity/precision quantity; interpreting it as nonnegative uncertainty-ellipsoid volume requires qualifications, regularization and possibly inverse information. Eq 8 approximation also requires small residuals. No calibration/error-prediction experiment in inspected tables establishes that higher score means larger decision error.

Appendix Table 8, PDF p 18, reports per-step time DUET 1.31s versus full UAGM 3.48s and MobileSAM UAGM 2.63s. Nearby text says the full method adds 1.35s, while the table difference is 2.17s; report table values separately. Table 11, p 19, labels five-run mean±standard deviation, confidence intervals and paired t-test p-values. The very small reported p-values warrant checking the statistical unit and raw paired runs before relying on them; no misconduct inference is made. Appendix E.2’s similar-looking renderings do not mathematically bound the neglected Hessian residual. E.3/Table 13 reports small gains when uncertainty maps are supplied to DUET, BEVBert and VER, which supports transferable signals but not calibrated probability estimates. E.4/Figure 7 gives concrete premature-stop examples: seeing a sofa before reaching it, and halting outside a room with a hidden target. E.4/Table 14 uses 20 selected scenes, so its 10/10 to 2/10 complex-scene result is not a representative benchmark estimate.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/Gaozzzz/Uncertainty-Aware-VLN](https://github.com/Gaozzzz/Uncertainty-Aware-VLN); inspected snapshot **43748013cc34376b68b159cca6226535ed84ccd0**. Snapshot metadata and source are under `repositories/01/uagm`.

Pinned official tree has only README; cannot inspect damping, singular-block handling, semantic embedding reduction, uncertainty gradient estimation or exact benchmark settings. No code execution.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

**Execution status:** No model source inspection is claimed beyond the explicit scope above. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **uagm-B1 — Still unresolved in reviewed evidence**. Calibration and numerical interpretation of appearance uncertainty are unverified: Fisher sensitivity is not posterior covariance without extra assumptions. A scalar uncertainty controller should not treat these features as probabilities. Evidence: AUTHOR Eqs 8–9 p 5; DEDUCTION. Type: methodological. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high for mathematical caveat.
- **uagm-B2 — Uncertain because evidence is insufficient**. Reconstruction uncertainty can reflect texture/surface sensitivity rather than task-relevant ambiguity; test equal-compute selective sensing or wrong-action prediction before claiming active information gathering. Evidence: DEDUCTION from unsupervised rendering scores vs task policy; Table 6 only aggregate navigation. Type: insufficiently tested. Follow-ups: active perception / world-model evaluators. Confidence: medium.
- **uagm-B3 — Still unresolved in this method; broader literature audit required**. Uncertainty is attached to current observations but does not select evidence-gathering actions; visible-yet-not-reached and hidden-target cases cause premature termination. Evidence: AUTHOR ACKNOWLEDGMENT: Appendix E.4/Figure 7, PDF p 21; F.1, p 23. Type: methodological. Follow-ups: HSGM, Active Visual Information Gathering, world-model methods in assignment 02. Confidence: high.

## 8. Access limitations and source record

No implementation released in pinned tree. Table 1 versus Table 4/6 REVERIE SR discrepancy remains unresolved. Figures show qualitative cases but do not establish physical collision avoidance on the discrete benchmark.

Version provenance: the local 23-page PDF is byte-identical to arXiv 2605.26503v 1 (26 May 2026); SHA256 and the exact downloaded URL are in evidence/01/uagm_version_provenance.json. Appendix F.1 explicitly lists active perception and predictive future views as future work. Noise robustness in Table 12 does not establish epistemic/aleatoric identifiability, nor physical navigation safety. The Table 11 test-unit question remains unresolved pending independent mathematical review.

Related primary sources: 

## Independent mathematical reporting check

Assignment 08 visually checked Table 11 on PDF page 19 and independently tested its stated five-run paired-t interpretation, including ±0.005 rounding of every printed mean and SD. If the reported SDs describe the same five paired observations, the difference SD satisfies `s_difference >= abs(s_ours - s_baseline)` by the covariance bound. With four degrees of freedom, all six printed p-values fall below even the minimum possible one-sided p-value under those rounded constraints. For DUET SR, the smallest possible two-sided p is approximately 4.13e-9 versus the printed 3.42e-12; for DUET SPL it is approximately 1.16e-8 versus 6.60e-15.

This is a reporting inconsistency **under that explicit statistical interpretation**. A different unit of analysis, SD definition or statistical protocol needs clarification. It does not establish misconduct, invalidate the reported mean gains, or reproduce navigation performance. The [independent logic review](../evaluations/08_logic.md) gives the derivation and alternatives; [executed bounds and assumptions](../experiments/08/logic_results.json) preserve all six rows. The earlier statement that this check was pending is superseded by this independently attributed result. Broader uncertainty and sensing follow-ups are assessed in [the final follow-up matrix](../followups/Bottleneck_Matrix.csv), rather than inferred from this consistency check.
