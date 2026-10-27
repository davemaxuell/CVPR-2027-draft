# Candidate register

Assignment 8 record. Candidates are research hypotheses, not established novel contributions. Independent A2/A5/A6 records were saved before cross-reading. Their final comparison and operational shortlist are in [A8](assignments/08_synthesis.md); the retained conditional candidate is [v3](proposal_v3.md).

| Candidate | Intervention family and causal rationale | Main alternative / failure | Preliminary treatment |
|---|---|---|---|
| K1 Original predicted-view scorer | Outcome-supervised evaluation of four one-step futures with actual memory | Context/action-only critic can explain gain; direct architectural prior | Essential baseline, no generic novelty claim |
| K2 Outcome-gap dynamics objective (A2-A) | Train existing predictor to preserve measured within-root utility differences through a qualified frozen scorer, with native loss anchoring | VAML/MCR overlap; critic exploitation; sample loss variance; detached current interface | Shortlist only after future channel proves useful |
| K3 Utility-weighted observed feature contrasts (A2-B) | Focus prediction on real action-dependent feature differences; no learned critic gradient | Weighted MSE/contrast loss overlap; feature changes may be irrelevant; no language-specific bottleneck evidence | Simpler rival to K2, not an added module |
| K4 Occupancy-matched root aggregation (A2-C) | Label histories visited by reranker with the same frozen continuation target | DAgger/rollout precedent, extra simulator cost, never cures unobservability | Coverage safeguard/control, not a novelty slot |
| K5 Root-relative critic supervision (A2-D) | Predict advantages rather than spending capacity on common offsets | Standard ranking/value learning; may replace all WM benefit | Strong no-future control |
| K6 Training-only future supervision / distillation (coordinator) | Use actual future supervision to shape a small critic, then omit future generation at inference | FutureNav/WorldMAP/World4Scorer overlap; loses online adaptability | Potential efficiency pivot, not original deployment mechanism |
| K7 Active evidence acquisition (coordinator) | Choose an actual turn/seek action when information is absent, rather than trust imagined missing evidence | SeekVLN direct prior; extra observations/controller protocol; scope grows | Explicit substantial pivot; not default rescue |
| K8 Grounded task-gradient projection (coordinator) | Constrain value-driven prediction updates to low-rank directions spanned by observed within-root action consequences; preserve native prediction objective | Span may omit task-relevant directions; train-only projection can cause mismatch; projected backward rule need not optimize a valid objective | Speculative shortlist for mathematical screening, no claim of safety or novelty |

An all-pairs squared gap loss alone is not a second mechanism: with residuals e_a=s_a-U_a, sum over a<b of (e_a-e_b)^2 equals K times the centered residual sum of squares. Relabeling this as both 'relative value learning' and 'ranking preservation' does not create two contributions.

Final selection must compare K2/K3/K8 against direct outcome learning and ordinary prediction, not combine them by default. Native projection/adapter choices, auxiliary heads, memory slots and candidate enumeration are implementation choices unless independent evidence establishes a substantive advance.

## Final selection

Retain **K2 only conditionally**, with K1 as original reference and K3/K5/K6 as substantive simpler rivals. K8 was mathematically screened and discarded: a projected nonlinear gradient need not be conservative; the valid scalar alternative has unsupported span/semantic assumptions. K4 is a coverage safeguard, K7 a substantial excluded pivot. Additions from A5/A6—adaptive branch repeats, longer discriminating horizon, residual/tiny predictor, compute gate and branch tapes/staging—are recorded as K9–K13 in A8. Repeats/tapes/staging are collection/engineering; longer horizon/gate are deferred; tiny prediction is a control.

A7 prefers K3 first because it avoids optimizing extrapolated critic values. The coordinator selects K2 to test measured-value alignment directly but requires K3 even absent visible exploitation. If K3 or simpler per-action weighting matches K2, replace/drop K2, rather than combining them as extra contributions. V2 adds no new module and only one conditional loss weight. Independent final review narrowed the putative scientific finding from 'semantic reversal' to outcome/proxy mismatch until valid semantic controls support the stronger interpretation. **No candidate currently establishes two novel CVPR contributions.**
