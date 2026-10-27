# Assignment 05 — completed

**Cutoff: 22 October 2026.** Completed the follow-up investigation of instruction reliance, generated-data selection, off-path supervision, grounding, stopping and recovery. Relevant final assignment 04 source deltas and assignment 06's ReInsVLN evidence were incorporated before closure.

- **24 scoped primary-source reviews**, including **21 outside-corpus papers**; all PDFs and extracted text available in `followups/sources`. Actual pages, versions, authors, primary links and limits are in `evidence/05/sources.json`.
- **37 bottleneck-resolution rows** connect **46 original paper IDs**; **48 relevant dossiers** were read. Direct PDF reads are distinguished from team-dossier evidence.
- **Six own pinned repository snapshots**, plus direct NavNuances and ULN source audits from 04's snapshots. No project training, inference, checkpoint or benchmark execution.
- **Two executed finite formula checks** and **one AST/file-existence audit**. Their scope is arithmetic/static evidence only. Artifact consistency checks pass.

Main deliverables: [follow-up review](../followups/05_review.md), [JSON matrix](../followups/05_matrix.json), [CSV matrix](../followups/05_matrix.csv), [source reviews](../evidence/05/source_reviews.md), [code audit](../evidence/05/code_audit.md), [final integration](../evidence/05/final_integration.md), [search log](../evidence/05/search_log.json), [validation](../evidence/05/validation_results.json).

## Findings for proposal development

1. **Recovery must distinguish location from unfinished obligations.** FGR2R, M-TRACK and landmark supervision already establish explicit progress mechanisms and relevant limitations. A new teacher must allow valid alternate continuations and survive paired-history tests; merely adding milestones is weak novelty.
2. **Pair validity and navigator competence are different selection targets.** Compatibility scoring, APS, CCC, SRDF and RSI-Nav are strong predecessors. Independently audit valid-hard and invalid-easy examples before training another selector.
3. **Verification and evidence acquisition already have strong recent implementations.** CLOSER, SeekVLN and VLN-NF substantially narrow generic stopping/active-sensing proposals. Target detectability, identity attribution and complete acquisition cost remain measurable questions, not certified novel mechanisms.
4. **Language sensitivity is insufficient without correctness.** NavNuances and behavioral diagnostics already supply paired controls; CAG/CAST/RSS occupy conditional guidance and same-observation alternatives. Test both valid contrasts and paraphrase invariance at equal information and compute.
5. **Endpoint rescoring is existing prior art.** DUET/FAST backtracking and MG-VLN ranking require any new score-calibration proposal to demonstrate independent headroom with fixed visited pools and physical return costs.

Final source limitations materially affect implementation plans: FGR2R and Landmark-RxR omit navigator implementations; M-TRACK is incomplete; NAP labels its release old; ULN's released uncertainty input differs from its paper and parallel exploration traces require cost replay; slow4fast's released reflection uses placeholders and lacks the documented full integration. These limit reproduction, not the papers' prior-art significance or their privately generated results.

No proposal is novelty-certified, no performance gain is invented, and unavailable code/supplements are not counted as reviewed. Assignments 07–10 should use the linked sources and precise controls to develop, challenge and rank the final ideas.
