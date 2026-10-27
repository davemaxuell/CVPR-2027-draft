# Assignment 01 — Foundations, DUET, data scaling, maps, and datasets

**Status: complete within the documented evidence and access scope.** Research cutoff: **2026-10-22**. All 33 assigned entries have a substantive dossier, verified bibliography, explicit inspected-page record, implementation/access assessment, and bottleneck/follow-up record. The full investigation's later-literature audit and proposal evaluation remain the coordinating agent's other assignments.

## Deliverables and coverage

- [Structured paper records](01_results.json): 33 entries, including full titles, authors, publication year/status, primary source links, code commits, and evidence labels.
- Individual reports: `../dossiers/<paper_id>.md` for every assigned ID listed below.
- [Implementation appendix](../evidence/01/implementation_appendix.md): phase-by-phase coverage for models, losses, training, inference, preprocessing, evaluators, and configuration; file/function locators and explicit gaps.
- [Pinned source inspection ledger](../evidence/01/source_inspection_ledger.json): 132 file/function records across 26 papers with actual source inspection. There are 29 pinned primary repository snapshots plus a separately pinned PREVALENT_R2R application snapshot.
- Seven distinct associated-file review records cover DUET's extended version, HAMT's supplement and extended version, ScaleVLN's supplement, NaviLLM's supplement, MARVAL's slides, and RoomTour3D's slides. Exact-SHA duplicates were identified without treating them as independent evidence. Relevant internal appendices were also read, including UAGM, SRDF, BEVBert, WS-MGMap, Self-Monitoring, RxR, and REVERIE.
- Search/source logs and exact UAGM-version provenance are under `../evidence/01/`. Public source snapshots are under `../repositories/01/`.

No fetched project code, checkpoint, simulator, training run, or benchmark was executed by this assignment. Source inspection, mathematical deductions, author-reported experiments, and proposed tests remain distinct. The coordinator's separate DUET routing toy test is not counted as benchmark validation here.

## Findings that materially constrain the proposed research

1. **DUET can backtrack and return to visited nodes.** Its exploration-destination mask coexists with shortest-path routing through visited viewpoints and final return to a cached visited stopping candidate. The extended appendix explicitly reports backtracking. The pinned R2R release performs final maximum-stop-score return whenever an episode ends; the paper specifically discusses forced-timeout return. Sampled training also forces exact-ground-truth-endpoint stopping. These distinctions merit controlled investigation, but do not establish a stopping bug or identify its performance impact.

2. **Stopping and goal reranking already have substantial prior art.** CMTP-BCE in Topological Planning uses a separate binary STOP loss. FAST separates frontier expansion from final goal candidates and trains a reranker combining path, progress, and speaker scores. Regretful and Self-Monitoring already implement progress-guided rollback. A new contribution must specify what these mechanisms lack.

3. **An OSR–SR gap does not identify a unique cause.** It does not by itself prove softmax competition, reduced language reliance, or an inability to return. ScaleVLN and SRDF improve several route-fidelity metrics on R4R/RxR. These results oppose an unqualified claim that larger synthetic datasets necessarily weaken language following, while still leaving semantic-verification questions open.

4. **Compatibility is not correctness.** RCM Figure 6 gives a successful trajectory with matching reward 0.53 and a wrong-stop failure with reward 0.54. SRDF's SPL/nDTW filters establish navigator consistency under specified metrics, not language entailment. NavRAG explicitly lacks a correctness evaluator; diverse generated language and shortest-path target conversion do not certify route semantics. AirBERT's noun/direction corruption tests are relevant counterfactual-discrimination prior art.

5. **Released implementation details affect baseline selection.** NaviLLM's released navigation branch is an LLM-conditioned graph classifier using DUET-derived infrastructure, alongside separate generation tasks. NavRAG actively weights STOP-target CE by ten; GridMM contains only commented-out STOP reweighting. WS-MGMap regresses bounded waypoints with MSE. These are different action/loss interfaces and should not be grouped solely as “VLM” versus “CE.”

6. **HSGM requires protocol reconciliation.** The paper describes consecutive STOP confirmation, while the inspected release preserves its first STOP flag across motion and can force completion after no movement. Its runner computes distance-based success/SPL without an explicit STOP predicate, including at a step budget. This is a comparability caveat, not evidence that the published results were computed incorrectly.

7. **Map uncertainty and representation labels need mathematical care.** Gaussian Map's 512-dimensional-to-scalar semantic reduction is non-injective. UAGM's Fisher log determinant is a sensitivity/precision quantity and requires numerical/rank conventions before being interpreted as uncertainty volume. GOAT's learned attention and residual feature paths do not establish front-door/back-door causal identification. These are bounded deductions, not measured benchmark failures.

8. **Physical and data costs frequently move outside the reported metric.** AirBERT's high-SR search result, Speaker-Follower's physically logged search, pre-exploration in Topological Planning, target-environment adaptation in RCM, oracle corrections in IVLN, and repeated-exposure Type III OVER-NAV are separate protocols. Training-free HSGM still uses learned perception and proprietary inference. AirBERT's “one-shot” experiment means fine-tuning on one house, not no training.

## Specific evidence gaps

At the pinned snapshots, original SRDF, Gaussian Map, and UAGM repositories were README-only. Later HLR/SkillNav specialist code may enable SRDF-policy experiments, but does not establish release of the original complete flywheel. Official executable implementations for RCM, MARVAL, OVER-NAV, and Topological Planning were not located in the bounded primary-source search; this is not a claim that no implementation exists. VER releases volumetric perception code, but the complete VLN rollout/evaluation integration was not located. PREVALENT's application code is pinned separately from its top-level repository.

UAGM's full local PDF is byte-identical to arXiv `2605.26503v1`. Its Table 11 statistical-unit issue is an unresolved reporting question requiring the underlying analysis unit/raw paired runs, not an allegation. Exact provenance and table text are preserved for independent evaluation. Reported timing and cross-table value discrepancies are retained explicitly rather than silently harmonized.

Current follow-up searches surfaced RSI-Nav, CAST, CoRe, and DRIVE-Nav as leads. Where only primary abstracts/records were inspected, they remain leads rather than full-paper findings. Assignments 05–06 and the novelty evaluator must determine which mechanisms actually overlap and which bottlenecks they resolve.

## Assigned inventory

`gaussianmap`, `airbert`, `bevbert`, `crossmodalmap`, `duet`, `gavln`, `goat`, `gridmm`, `hamt`, `hsgm`, `ivln`, `kerm`, `marval`, `navillm`, `navrag`, `overnav`, `prevalent`, `r2r`, `rcm`, `regretful`, `reverie`, `roomtour3d`, `rxr`, `scalevln`, `selfmonitoring`, `speakerfollower`, `srdf`, `tacticalrewind`, `topologicalplanning`, `uagm`, `ver`, `vlnbert`, `weakmap`.

All 33 are complete at the stated scope. Bibliographic abbreviations and merged prose were corrected in both dossiers and structured records. No empty author list, missing primary bibliography URL, malformed cited commit, or missing dossier remains in the final validation.
