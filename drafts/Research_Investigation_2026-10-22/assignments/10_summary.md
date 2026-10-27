# Assignment 10 complete

Independently reviewed implementation feasibility and empirical validity for **P1–P4**. All four require revisions before method claims; P1/P2 remain the best low-cost diagnostic starting points. P3 training should wait for representation/learnability evidence, while P4 initially supports a release-level negative-write audit.

Deliverables:

- [Full evaluation](../evaluations/10_feasibility.md)
- [Structured decisions and findings](../evaluations/10_findings.json)
- [24-file pinned inspection manifest](../evidence/10/source_manifest.json)
- [Executed source-function tests](../experiments/10/source_check_results.json) and [reproducible script](../experiments/10/source_checks.py)
- [Independent asset/annotation check](../evidence/10/asset_check_results.json)
- [Validation](../evidence/10/validation.json)

Critical revisions: P1 must change/control DUET training termination and cached endpoint return, not only teacher labels. P2 must preserve route-family splits, source-versus-verification trajectory distinction, human witness effort and update/replay budgets. P3's selected NavQ predictor is frozen/detached during navigation, so its new head needs explicit pretraining and a validated ordered-route task port. P4's released color-histogram hard veto differs from paper EEC; feature/retrieval/gating controls and no-write/quarantine are mandatory.

Executed actual AST source checks: SkillNav mixed batch uses the last selected navigation head for every item and changes under permutation; PSG-Nav can write/reject a stipulated-correct synthetic crop following supplied control failure, even with positive evidence, and the actual rejection caller clears the goal/marks a spatial region; DUET endpoint teacher returns STOP for both illustrative histories. No project module, checkpoint, simulator or navigation benchmark was executed. These checks establish conditional code behavior, not natural error prevalence or invalid published results.

Remote NavQ checkpoint listings are accessible, but checkpoint contents and compatibility were not tested. Local annotation hashes and scene disjointness independently match the inventory; raw annotations need tokenizer encodings. Physical 8 GB GPU hardware exists, while the installed torch build is CPU-only and the simulator/assets remain unestablished.
