# Assignment ledger

Concurrency ceiling: coordinator + three workers. Worker tasks are actual native delegations, not simulated personas. Native workers inherit the runtime model; no external model-provider calls are authorized or made. Coordinator owns substantive Assignment 8.

Worker names are relative to `/root/` (for example `a1_baseline` is `/root/a1_baseline`). All ten investigation assignments are complete; unexecuted CE experiments remain separate.

| Assignment | Worker | Scope | Dependencies | Status | Deliverable / evidence | Open issue |
|---|---|---|---|---|---|---|
| 1 | a1_baseline | Problem and baseline audit | Original sources | Complete | assignments/01_problem_baselines.md | Current-policy headroom unmeasured |
| 2 | a2_formulation | Faithful mathematical formulation and independent candidates | Original sources | Complete | assignments/02_formulation.md; 02_independent_candidates.md | Model and utility assumptions |
| 3 | a3_adversarial | Independent adversarial reconstruction, then challenge A2 | Original first; A2 later | Complete | assignments/03_adversarial.md | Stochastic/aliasing risks need real outcomes |
| 4 | a4_novelty | Direct novelty and current literature | Original sources | Complete | assignments/04_direct_novelty.md | Recent preprint overlap |
| 5 | a5_crossfield | Cross-field equivalence and independent candidates | Original sources; saved candidates before peers | Complete | assignments/05_crossfield.md | Semantic/outcome mismatch and adaptation unmeasured |
| 6 | a6_feasibility | Code, resources, availability and independent simplification | All original10; candidates before peers | Complete | assignments/06_feasibility.md | GPU/simulator/checkpoint prerequisites |
| 7 | a7_experiments | Falsification, estimands, controls and resource gates | Material A1–A6; later v2 protocol corrections | Complete | assignments/07_experiments.md; experiments/README.md | All CE stages unexecuted; simpler feature-objective preference retained |
| 8 | /root | Creative exploration, mathematical pruning, synthesis and reconciliation | Material A1–A7; independent9/10 rechecks | Complete | assignments/08_synthesis.md; proposal_v3.md; shared registers; Research_Report.md | Two strong contributions not secured |
| 9 | a9_correctness | Independent final correctness | Fresh non-designer; initial assessment before peer verdicts; v1/v2/v3 inspected | Complete | assignments/09_correctness.md; initial assessment; evidence/09_* | Fixed-critic exploitation and semantic attribution require empirical evidence |
| 10 | a10_novelty_viability | Independent final fresh-search novelty/viability | Fresh distinct non-designer; own initial assessment; v1/v2/v3 recheck | Complete | assignments/10_novelty_viability.md; initial assessment | Two contribution targets unestablished; Q_ref remains policy-relative |

## Inspected evidence and independence

| Assignment | Evidence actually inspected / executed |
|---|---|
| 1 | Full original design; DREAMWALKER/HNR papers/supplement; StreamVLN/SimpleVLN/FutureNav/ReInsVLN primary papers and pinned releases; evidence/a1 provenance/snapshots |
| 2 | Original design/scorer/labels; primary mathematical precedents; own500-tree/7500-history/79-margin checks; independent candidates saved first |
| 3 | All original10; reconstruction before A2; A2 underlying source/results; primary policy-improvement/CVAML; own2150-history/1296-margin/gradient checks |
| 4 |15 primary full-text snapshots and4 pinned implementations; VISTAv2/SeekVLN/World4Scorer/MCR closest sections and source/query ledger |
| 5 | VAML/IterVAML/VaGraM/CVAML/value-equivalence/decision-focused/rollout/distillation sources; independent centered/Laplacian derivation and candidates |
| 6 | All original10/source hashes; pinned actor/DINO/NWM code;13 copied scorer tests; CPU profile; scoped hardware/interpreter/assets; candidates before peers |
| 7 | Original design/code; completed1–6 and underlying execution logs; pinned VLN-CE metrics/StreamVLN config; primary World4Scorer; no redundant toy test |
| 8 | Original10, all audits and saved independent candidates; direct primary source rechecks; mathematical rejection of projected-gradient complexity; versioned synthesis/reconciliation/package audit |
| 9 | Original source; primary CVAML/World4Scorer/MCR/DINO/GUI/RIWANav; A2/A3 scripts; A6/A7; own6561 bound/gradient checks; exact reviewed hashes |
| 10 | Original/current proposal; fresh primary searches including GUI decision-value/RIWANav/PiJEPA/GWM; closest World4Scorer/MCR and A6/A7; exact reviewed hashes |

Exactly ten substantive work packages were performed: nine actual native worker assignments plus coordinator8. Initial schedule1/2/4 was backfilled by3/5/6, then7 after material findings. Fresh9/10 began after v1 freeze and received material v2/v3 revisions. Corrections stayed within existing assignments. Completed workers were not counted as concurrently running; maximum coordinator plus three workers was respected. An optional A4 follow-up attempt was rejected by the platform's active-thread limit; no additional worker ran or work was claimed from it.

Final versions and review dispositions are in [reconciliation](review_reconciliation.md). Supplied-file ownership is in [material coverage](material_coverage.md). Completion of review assignments does not mean blocked CE experiments were executed. Native-worker procedural independence is not cross-model verification.
