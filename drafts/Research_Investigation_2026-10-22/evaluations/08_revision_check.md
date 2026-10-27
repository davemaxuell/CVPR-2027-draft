# Assignment 08 focused revision check

**Decision: bounded acceptance of P1, P2 and P4 as conditional diagnostic studies. No residual material logical blocker was found in the reviewed revision.** P3 is correctly excluded as a new-method recommendation. This is a continuation of assignment 08, not an additional assignment.

I compared the revised Markdown and JSON with [the original logic review](08_logic.md), its [structured findings](08_findings.json), and the [revision record](../proposals/Revision_Record.md). I also checked the [P3 rejection appendix](../proposals/rejected_ideas.md). Exact SHA256 hashes and the machine-readable decisions are in [08_revision_check.json](08_revision_check.json). All ten substantive A–J fields in each JSON proposal occur verbatim in the Markdown; the JSON contains exactly P1, P2 and P4.

## P1: material objections resolved in the formal contract

The acceptance predicate `C(v,q)` now explicitly includes the **current** required terminal position as well as completed obligations. Thus a history state recording that the goal was once visited cannot license STOP after leaving it. STOP enters an absorbing success or failure terminal, rather than a zero-cost self-loop. The zero-slack Bellman argument is restricted to finite, static, known teacher dynamics with positive movement costs; it is not transferred to the learned policy.

For implementation clarity, spell out `c(STOP)=0`, `V*(success terminal)=0` and `V*(failure terminal)=infinity`. The terminal values already follow from minimum accepting-completion cost; the explicit zero-cost STOP convention completes the travel-cost definition. This is a sufficient wording clarification, not a residual mechanism blocker. It must preserve absorbing termination rather than reintroducing a zero-cost self-loop.

The primary semantic audit now evaluates complete teacher continuations, including STOP, irreversible rejection or a separately declared shared budget. It does not call a recoverable, inefficient first action a semantic violation. `Y` denotes efficient actions and `F_B` denotes budget-feasible completions. Unreachable and unresolved states remain in the reporting denominator. These revisions accurately incorporate the original J→G versus J→G→STOP counterexample and its budget distinction; no new numerical test was needed.

The initial interface is fixed graph neighbors plus STOP for every comparison. The node-landmark grammar avoids unsupported orientation/control claims. Native DUET integration explicitly requires the evolving map/menu state or a frozen-menu subproblem, common endpoint/exhaustion handling, and a shared cached-return convention. Actual path-node bookkeeping is correctly distinguished from observed intermediate images. These are consistent with the previously inspected DUET teacher and termination source at commit `93e8b233164bc079a6db48b8a0a78d123ec8de41`, `map_nav_src/r2r/agent.py:239–285,404–478`.

The action/state ambiguity masks, transferable state encoding, set-loss limitations and comparison with label-only training are also explicit. **Remaining limits are empirical prerequisites, not hidden guarantees:** reliable natural annotations, consequential invalid teacher continuations, a deployable state signal and an incremental result beyond the strong teachers must still be demonstrated. The complete-continuation audit is the appropriate next gate.

## P2: witness and selection logic corrected

The witness unit is now a **complete evidence bundle**. Its OR-loss applies only across alternative complete bundles; it no longer treats separately required pieces as interchangeable witnesses. The alternative group formulation uses separate normalized distributions and a joint relation encoder. Missing-element and order-swap tests are mandatory, and the text correctly avoids claiming that an attention loss or saved explanation certifies evidence use.

The fixed-count selector uses disjoint exhaustive strata with feasible quotas. The revision acknowledges that hard quotas may select low-quality examples; a quality floor with abstention/infeasibility is a separate protocol and cannot silently preserve an equal-count claim. This matches the elementary per-stratum ranking argument in the original review.

Unresolved status is correctly described as an annotation/evidence state. Masked labels identify the labelled subset without additional assumptions; the score is not presented as a calibrated population truth probability. Probability sampling before enrichment, route-family grouping, label-availability reporting and annotation-cost controls address the stated selection/missingness objections. **Remaining limits:** complete witnesses may be unavailable or unreliable, the scorer may fail the missing-piece tests, and selection quality may not improve navigation. Those are accurately retained as falsification conditions.

## P4: identification and gate limits preserved

The revision does not infer a wrong semantic category merely from failed arrival. It states that indistinguishable causes cannot be separated by adding heads, that masked BCE fits labelled subsets without missingness/support assumptions, and that simulator interventions need not match natural failures. The invariance loss is limited to evidence- and label-preserving controller interventions and is acknowledged to admit constant predictions.

The entire correlated `q_D/W` gate must be measured for precision, coverage and downstream memory effects. The text explicitly says marginal calibration can fail after witness conditioning; it does not multiply independent confidences or infer gate precision from a threshold. Predicted, witness-only and oracle-witness variants remain distinct, with scene-disjoint threshold development and no shift guarantee. This accurately addresses the earlier calibrated-score/uncalibrated-subgroup counterexample.

The proposed study is also bounded to the inspected release where appropriate. A conditional write/read pathway is not natural incidence or a failure of paper-faithful EEC. No-write, quarantine, simple controller rules, representation/retrieval controls, bank reset/exposure and feedback matching remain necessary. Offline replay is explicitly separated from closed-loop navigation SR. **Remaining limits:** identifying evidence may be unavailable, suppression may be rare, simple quarantine may suffice, or retained stale beliefs may offset safer writes. All remain stated kill criteria.

## P3: rejection accurately retains only a diagnostic

P3 is absent from the retained-proposal JSON and clearly rejected as an original-method proposal. The appendix preserves the corrected `AB0/B0A/0AB` versus `A0B/0BA/BA0` example, including matching timewise and ever-event marginals, and does not claim an actual NavQ collision or learnability result. Its event-monitor discussion uses the prior monitor state to preserve strict order. The expected-discounted-sum wording, probability-coherence limits and continuation-policy conditioning are also accurate. I did not independently repeat assignment 09's novelty search in this focused check.

## Acceptance scope

The earlier material objections are either corrected in the formulation or retained as explicit empirical limits. The revised artifacts can support the coordinator's final recommendation **to run the diagnostic gates**. They do not establish three original algorithms, meaningful natural error rates, successful training, benchmark gains, convergence, or calibrated deployment risk.

No new literature search, downloaded project execution, model training, annotation collection or navigation benchmark was performed for this recheck. The only new executable check was document/JSON consistency and hashing. Earlier primary-source inspections and isolated mathematical checks remain attributed to their original records. Future substantive changes to the hashed formulations need a new scoped review; editorial status changes alone do not alter this logical decision.
