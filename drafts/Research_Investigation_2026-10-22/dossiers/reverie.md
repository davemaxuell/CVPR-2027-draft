# REVERIE

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments**. Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, Anton van den Hengel. CVPR 2020.

Primary paper: [REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_REVERIE_Remote_Embodied_Visual_Referring_Expression_in_Real_Indoor_Environments_CVPR_2020_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 11].

Associated files: REVERIE.pdf (Main PDF; 16 pages; SHA256 6047a52873cb927c34dfe3e456f2764ca64413b0ae3887993ba07b9cbd89fa48).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

REVERIE requires navigating to and identifying a remote object described by a high-level command, rather than merely stopping near a location. Object recognition and stopping are coupled: correct navigation alone is insufficient.

## 3. Mechanism, training, and inference

Section 3, PDF pp 4–5: 21,702 instructions refer to 4,140 target objects in 489 categories. Target boxes are projected from 3D object annotations into simulator views and retained within 3m. The agent receives panoramic RGB and navigable viewpoints and may output one target box or object ID. Selecting the correct supplied candidate, or predicting a box with IoU at least 0.5, counts as correct grounding. Multiple viewpoints may see the same object. The baseline combines FAST-short navigation with MAttNet subject, location and relation matching. For each view, the top three matching objects supply averaged visual features and BiLSTM-encoded category labels, concatenated with panorama features. Navigation uses action cross entropy and progress squared error. The pointer uses margin ranking: L=Σ[δ+S(o, r−)−S(o, r)]+ +[δ+S(o−, r)−S(o, r)]+, where positive object/expression pairs must outscore mismatched pairs by margin δ. Printed Eq 8 has a negative squared-error term and uses logit notation inside a logarithm; the released loss must disambiguate these rather than implementing the literal expression. Section 5.1 describes training the pointer first, then using its object information for navigator training, qualifying the compact joint-loss description.

## 4. Novelty and evidence

A remote referring-expression benchmark and interactive navigator–pointer baseline couple navigation with object grounding. The task goes beyond goal proximity. However, supplied annotated object boxes are privileged perception proposals compared with detected boxes in a physical deployment.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 3, p 8 reports test joint success 11.28% versus 7.07% for FAST plus a pointer and 77.84% for humans; navigation-only SR is 19.88%. Table 4 places the model at the true goal: MAttNet achieves 48.98% grounding accuracy versus 90.76% for humans, showing that object selection remains difficult even after navigation succeeds. The weak language-only baseline demonstrates a need for visual evidence, but does not prove every model uses every semantic relation. VERDICT: report RGS/RGSPL alongside SR/SPL, and distinguish ground-truth boxes from detected boxes. Goal-only datasets cannot replace this evaluation.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/YuankaiQi/REVERIE](https://github.com/YuankaiQi/REVERIE); inspected snapshot **207e90cdde48acf2f2f3fc912528966a062fc432**. Snapshot metadata and source are under `repositories/01/reverie`.

The pinned official release includes the navigation model, trainer and eval_release.py. modelFast.py and follower.py were retrieved for the loss, stopping and grounding audit. Reproduction additionally requires the dataset, simulator and trained perception components.

Additional loss inspection: follower.py 798 uses positive 0.5×cross-entropy +0.5×MSE for the progress head. eval_release.py 56–109 separates endpoint visibility success, any-visited-view oracle success, target-ID grounding and graph path length. Its target-ID correctness branch does not itself additionally gate on visibility; practical object-candidate construction is therefore part of the evaluation contract.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [tasks/REVERIE/follower.py](https://github.com/YuankaiQi/REVERIE/blob/207e90cdde48acf2f2f3fc912528966a062fc432/tasks/REVERIE/follower.py) — 304–305;685–712;798: Positive MSE and action CE, including an explicit equally weighted combined loss.
- [tasks/REVERIE/eval_release.py](https://github.com/YuankaiQi/REVERIE/blob/207e90cdde48acf2f2f3fc912528966a062fc432/tasks/REVERIE/eval_release.py) — 56–109: Grounding target ID match; endpoint/anywhere target visibility for navigation and oracle success; graph-distance trajectory length.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **reverie-B1 — Partially addressed**. Navigation success can hide incorrect object identity, while annotation-provided proposals simplify perception. Evidence: Section 3 metric definitions and Table 4 grounding at the true goal. Type: evaluation/perception. Follow-ups: DUET object branch, GaussianMap, NavRAG navigation without object boxes. Confidence: high.
- **reverie-B2 — Partially addressed**. Goal-directed instructions may omit route evidence, and multiple objects or viewpoints create ambiguity. Endpoint confidence is not object confidence. Evidence: Dataset definition and measured pointer errors. Type: methodological. Follow-ups: KERM knowledge, NaviLLM multitask grounding. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/cshizhe/VLN-DUET; https://github.com/LaVi-Lab/NaviLLM
