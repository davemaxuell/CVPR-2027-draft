# Claim register

Original specification frozen by hashes in material_inventory.json. The first table preserves intake claims; the final table records dispositions after the audits. These IDs identify original claims and are distinct from proposal contribution targets C1/C2.

| ID | Precise claim | Necessary assumptions | Supporting / opposing evidence at intake | Intermediate change and outcome | Falsifier / next evidence |
|---|---|---|---|---|---|
| C1 | Current small CE policies make consequential action-selection errors that one-step reranking can correct | Candidate includes better action; continuation competent; root distribution relevant | User hypothesis; prior oracle analyses are leads, not evidence on these policies | Better first choice → higher actual continuation utility | Measured candidate oracle gain negligible; collect all four branches at matched roots |
| C2 | Predicted future features improve ranking over identical real context and action alone | Predictor carries useful inductive bias; scorer learns to use it; generalizes to new scenes | No CE test; future is generated from existing observations, not additional sensed information | Better decision representation → lower action regret | Matched no-future model equals or beats full model |
| C3 | Proposed recurrent memory and instruction conditioning preserve route progress | Labels distinguish relevant histories; memory capacity and gradients suffice | Structural memory checks only; policy q already includes history | Resolve identical-looking but different-progress states → route adherence | Context-only scorer equal; legal order-sensitive task counterexamples |
| C4 | Real-branch Q_ref utility labels support deployed repeated reranking | State restoration exact; labels comparable; coverage and policy mismatch controlled | Labels mathematically definable; no CE collection | Learn useful action contrasts → better closed-loop policy | Offline regret improves while episode outcomes worsen |
| C5 | Task-aware world-model training can outperform ordinary prediction | Precise objective; grounded constraints; scorer cannot be exploited | Explicitly only a future extension in original; strong VAML precedents | Reduce decision-relevant prediction errors → lower regret | Lower training critic loss without better real-outcome ranking |
| C6 | Added system is feasible within 8 GB | Actual backbone/context, simulator and predictor fit or can be scheduled | Scorer counts only; no GPU profile | Affordable deployment | Full-pipeline memory/latency fails target |
| C7 | There are 2–3 distinct CVPR contributions | Direct/cross-field prior work does not subsume claims; meaningful evidence | Original explicitly says novelty pending | One coherent scientific advance | Generic world-model + critic + memory already established; no distinct second result |

For every claim, distinguish published experiments, code inspected, tests executed here, deductions and hypotheses. A positive claim status requires its own evidence rather than team agreement.

## Final disposition and next evidence

| Original claim | Supporting / opposing evidence from review | Current status | Next decisive evidence |
|---|---|---|---|
| C1 Candidate opportunity | DREAMWALKER/HNR show adjacent intervention headroom, but protocols/policies differ; one-sample maxima can exaggerate opportunity | Broad problem valid; specific H1 headroom unverified | A7 E0/E1 deterministic or cross-fit selected-action lift on ordinary roots |
| C2 Predictive usefulness | Finite-capacity/prior explanation coherent; generated futures add no new sensed evidence; GUI direct-ranking precedent strengthens control requirement | Hypothesis, no CE support | E2/E3 full-input direct critic, real/predicted/copy/tiny predictor and cost controls |
| C3 Progress memory | Synthetic memory invariance passes; available policy context already has history; unobservable states cannot be repaired by memory capacity alone | Structural contract tested, semantic ability unknown | Legal stage-sensitive tasks and explicit-memory versus same-policy-history control |
| C4 Repeated Q_ref improvement | Conditional full-history policy-improvement deduction and independent arithmetic checks; compressed state/occupancy error unresolved | Mathematically coherent under stated assumptions | Held-out calibration and offline-to-closed-loop effect; targeted occupancy refresh if needed |
| C5 Task-aware dynamics | Operational centered objective and gradient contract specified in v2; VAML/MCR/World4Scorer overlap and critic-exploitation counterexamples | Conditional unimplemented extension | Native/simple weighted/feature-contrast/closest-objective rivals with independent branch outcomes |
| C6 8 GB feasibility | 2.87–3.17M scorer, 13 copied CPU checks pass; native feature-grid mismatch and missing integrated environment; full cost unmeasured | Scorer feasible; full claim unproven | Real checkpoint adapter, CE replay, full-device/latency profile |
| C7 Two–three contributions | Two complementary possible targets; direct and cross-field mechanisms substantially anticipated; no project-specific result | **Requested novelty target not met** | Consequential outcome/proxy finding with independently justified semantic role, and nonroutine minimal remedy |

All original claims retain explicit falsifiers. Rejected overclaims are not substituted for the user's careful original qualifications. V2's two contribution targets remain hypotheses: a diagnostic method alone is not a novel finding, and outcome/geometry disagreement alone is not semantic understanding.
