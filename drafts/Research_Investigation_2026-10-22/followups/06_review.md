# Follow-up audit: prediction, memory, uncertainty and adaptation

Investigation cutoff: **22 October 2026**. Assignment 06 completed a targeted follow-up audit of 15 papers' substantive full-text sections, plus two explicitly limited records: ReliableNav's publisher abstract and selected official code, and selected HTML sections of the Sensors safety-gating study. Supplement coverage varies and is stated per source. Seven executable official implementations were inspected at pinned commits; one additional release was README-only. Five small original mathematical checks were executed. No navigation benchmark, fetched project, model checkpoint, training run or robot experiment was executed.

The detailed record is divided into [17 source cards](../evidence/06/source_cards.md), a [machine-readable bibliography](../evidence/06/sources.json), [implementation/check appendix](../evidence/06/implementation_and_checks.md), [search scope](../evidence/06/search_log.json), and the [20-row bottleneck matrix](06_matrix.json) ([CSV](06_matrix.csv)). The source cards hold compact source-specific descriptions and precise locators; this report develops comparative deductions rather than repeating them.

## What the follow-up evidence changes

The broad ingredients are already crowded: action-conditioned imagination, observed forecast residuals, uncertainty-based horizon selection, source-labelled imagined maps, persistent future-guided retrieval, signed negative memories, confidence calibration, evidence revisits and uncertainty-triggered replanning all have relevant predecessors. A credible contribution must identify a narrower failure mechanism and demonstrate that its intervention changes decisions under matched information and compute.

The strongest common gap is the distance between **a useful internal score and a useful decision**. Prediction fidelity, action likelihood, scalar calibration, set coverage, retrieval compatibility and physical success are different targets. Improving one does not establish improvement in the next. The reviewed literature gives concrete reasons to evaluate those links, but does not prove they universally fail. Matrix W01-W04, U01-U02 and P01-P02 specify the settings and evidence boundaries.

Several tempting premises are incorrect. SC²-WM already has delayed real-observation residual adaptation. WMNav can reset identified goal cells despite its ordinary minimum merge. Pathdreamer/SGM and newer imagination systems distinguish observed and generated material. DUET permits backtracking and cached-stop recovery. NavFoM has an oldest-frame fallback. EvoNav uses its own rollout and terminal metrics rather than reference trajectories at inference. These corrections are preserved in the original dossiers and must constrain assignment 07.

The investigation also exposed release-level differences. A paper's architecture diagram is insufficient to specify an executable baseline: the released memory pathway, objective reduction, rollout gradient, retrieval score or default controller can differ. The implementation appendix records these discrepancies without asserting which unpublished branch produced a table.

## Follow-up map

| Original line | Strongest reviewed continuation or challenge | What remains to distinguish |
|---|---|---|
| Pathdreamer, HNR, NavMorph, MonoDream, SC²-WM | UA-NWM; PiJEPA; Adaptive Rollout; NEUBAY | Task-relevant action ranking, forecast horizon value and deployment feedback versus generic prediction quality |
| SGM, Learning to Map, WMNav, SALI | ImagineNav++; Schrödinger's Navigator; WorldMAP | Source provenance, absent-object evidence, actual execution and the price of extra sensing/planning |
| HAMT, GridMM, IVLN, GSA-VLN | Memoir; DART-VLN; JanusVLN; ReInsVLN; CMMR-VLN | Relevant recall versus extra context, extra inference and stronger cross-episode information |
| KnowNo, Explore until Confident, CA-Nav, AIUTA | CPNAV; VLA Calibration; ReliableNav; safety gating | Coverage/calibration versus conditional decision benefit and physical execution |
| FSTTA, FeedTTA, ATENA, SAME, M3E | Original implementation audits plus calibration/feedback follow-ups | Proxy quality, supervision access, selection effects and domain retention |
| Graph VLN to continuous/physical control | HumanoidVLN and reviewed physical deployments | Controller, morphology, localization, collision and latency contracts |

These arrows indicate mechanism relationships, not claims that later work uses the same architecture or conclusively resolves the predecessor's bottleneck. Follow-up status is assigned per matrix row, not per entire paper. Two rows resolve narrow source-level ambiguities; they do not certify that the broader research problem is solved.

## Comparison rules that change conclusions

Use a protocol vector for every baseline: sensor access; localization/map access; candidate generator; instruction/goal access; low-level controller; episode memory/reset; feedback; training data; calibration data; parameter updates; inference budget; and success criterion. A result is comparable only on the dimensions relevant to its claim.

For example, offline future-image goals, generated trajectory supervision and embodied route following answer different questions. A predicted camera motion need not be an executable robot motion. Persistent tours with oracle correction are not equivalent to autonomous repeated deployment. A frozen high-level VLM may still depend on trained waypoint or low-level policies. Supervised learning on a new domain is not label-free test-time adaptation. These distinctions explain why the audit does not create a cross-paper numerical ranking from incompatible success rates.

Memory cost must include raw frames, retrieved frames, feature tensors, graph nodes, database entries and model queries. Counting only the currently attended tokens can hide unbounded raw history. Conversely, a recent-frame cache does not prove all old information is absent if a separate sampler can recover old frames. Useful testing manipulates the needed evidence while holding the complete budget fixed.

For uncertainty, specify the event and population before naming a score calibrated. Episode success, local action correctness, collision, object visibility, answer correctness and admissible-path retention are not interchangeable. A calibration set also needs a stated statistical unit: steps from an episode are dependent; policy changes alter the states visited; an intervention changes the outcome distribution. Error bars should resample the appropriate episode or scene cluster, with a separate domain-shift test.

## Prioritized measurable opportunities

The following are **research gaps to test**, not novelty-certified proposals. Assignment 07 must develop mechanisms and assignments 08-10 must independently challenge them.

### 1. Determine when a forecast changes the right decision

Define a state/history h, instruction x, candidate action sequence a and forecast score s(h,x,a). The target is not merely low reconstruction error: it is a useful ordering of candidate consequences under the task's route and execution constraints. A candidate study should distinguish model error from irrelevant visual variation and from task-relevant ambiguity. The strongest reviewed controls include structured error scoring, its released optional contrastive/parallel losses, real-observation correction and policy-guided candidate generation (W01-W03).

**Cheapest decisive test:** freeze one backbone and candidate set, collect independent ground-truth outcomes in matched simulator branches, and measure ranking regret and harmful confidence before introducing a new planner. Compare simple cosine error, scalar variance, calibrated score, the existing structured scorer, and its already-released loss options. Include horizon-schedule replay and fixed-budget search. If a proposed score cannot improve ranking at equal candidates and observations, an expensive end-to-end planner is premature.

**Success criterion:** a prespecified reduction in action-selection regret or failure at equal execution and sensing budget on held-out scenes; aggregate ECE alone is insufficient. **Confidence:** high that generic uncertainty/forecast ideas already exist; medium that this precise navigation slice remains inadequately tested. It may become an evaluation contribution unless a new causal mechanism survives novelty review.

### 2. Validate when memory may be contradicted or retracted

A failed attempt can mean occlusion, localization error, wrong instance, unreachable route, instruction misunderstanding or true absence. These causes imply different memory updates. The relevant gap is whether a memory update carries enough spatial, temporal and visibility evidence to invalidate a prior claim without destroying useful uncertain hypotheses. Positive/negative/unknown memory, source tags and observed-value overrides are already prior art (W04-W05, M03).

**Cheapest decisive test:** controlled paired scenes or recorded views with one factor changed: object moved, object merely occluded, viewpoint wrong, same-class instance confused, or route blocked. Feed identical observations to a no-update baseline, recency decay, tri-state/source-tag baseline, and the proposed update. Measure false deletion, stale-fact persistence, recovery time and downstream wrong-goal selection. A new module that merely marks unseen objects absent should fail this test immediately.

**Success criterion:** fewer harmful deletions and fewer stale-fact errors jointly, followed by matched navigation benefit. **Confidence:** medium; bounded search does not establish universal novelty. PolarMem, Context-Nav, CLOSER and assignment 05's goal-verification work are mandatory overlap checks. This is more defensible than claiming hallucination is solved by keeping two maps.

### 3. Preserve instruction-critical relations under a complete memory budget

The failure slice is a current action that depends on an earlier ordinal, spatial relation or disambiguating view, with visually similar alternatives. More history, future-guided retrieval, fixed reminders and token compression are established. The remaining question is whether a task-specific retention/retrieval mechanism restores the necessary evidence rather than simply paying for more queries or re-grounding the starting state (M01-M02, M05).

**Cheapest decisive test:** at fixed recorded checkpoints, vary an earlier relation while preserving current pixels and distractor identity. Compare original history, budgeted retrieval, instruction-only reminder, neutral reminder, random matched frames and oracle relevant evidence. Separate errors present at the initial state from errors caused by a later loss of history. Measure paired action changes and preservation of previously correct decisions, not just recovered failures.

**Success criterion:** improvement on validated evidence-dependent cases at the same total stored bytes and vision/model calls, without relying on reference-path suffixes at deployment. **Confidence:** medium. Memoir is a particularly strong predecessor and its oracle retrieval gap does not by itself identify a new mechanism.

### 4. Choose interventions by benefit rather than confidence alone

A trigger can be well calibrated yet waste interventions on irrecoverable states, or interrupt correct behavior. The scientific target is a decision among continue, acquire information, update/replan and stop/help, with explicit costs and observable state. The generic three-way gate and progress-conditioned confidence idea already exist (U02, A02), so novelty would require a defensible mechanism for estimating incremental intervention value.

**Cheapest decisive test:** matched-state intervention branches, including a random or fixed-frequency control with the same intervention count. Compare confidence ranking with actual benefit conditional on intervention type, then test on a held-out policy or scene distribution. Use observed progress; hindsight normalized episode length is a diagnostic, not an online feature. First establish that intervention outcomes are identifiable from the available data and that the cheapest control does not explain the gain.

**Success criterion:** a better success/cost frontier with valid early warning and preserved calibration on the evaluated distribution. **Confidence:** lower novelty confidence than the other opportunities; active sensing, policy steering and assignment 05's SeekVLN are strong collisions. Reject a generic confidence threshold proposal.

## Physical execution and adaptation are validation axes

The audit does not recommend treating every deployment gap as a separate architectural proposal. Fix one high-level policy and cross controlled sensor noise, localization error, tracker quality, body morphology and latency. Report falls/collisions, intervention time, travel distance and completed instructions separately. The HumanoidVLN and physical source cards provide motivating evidence, but their limited pilots do not establish universal transfer.

For adaptation, first preserve the exact feedback contract. GSA-VLN and IVLN primarily change persistent inputs; SAME jointly trains on mixed datasets; M3E uses labelled domain-incremental supervision; FSTTA, FeedTTA, ATENA and SC²-WM use different online objectives and signals. Assignment 04's paper analyses were incorporated. This assignment did not independently inspect source details that were still pending when its review closed. **Coordinator update:** assignment 04 subsequently completed its scoped source audits; see its [final implementation record](../assignments/04_repository_audit.md) for SAME's batch-axis router, M³E's release blockers and slow4fast's placeholder/integration limits. Those release findings constrain reproduction and do not overturn the reported paper results.

A useful feedback experiment corrupts terminal labels, changes episode order and compares resetting versus persistence. It distinguishes self-confirmation from actual external evidence. Claimed expected-gradient or stronger-gradient guarantees must survive the algebraic checks before being used to justify robustness. They do not substitute for measuring adaptation performance.

## What is established, and what is not

The executed checks establish five narrow properties: a printed confidence interpretation has a counterexample; two residual reductions differ; a constructed subspace can hide a task-relevant difference; equal relative memory priors cancel; and path retention does not alone imply finite-budget success. None measures benchmark impact. The corresponding sources and formulas are in the implementation appendix so independent evaluation can reproduce the checks.

The final ranking of ideas should therefore depend on cheap discriminating experiments and the novelty audit, not on invented gains. The strongest immediate investment is a matched-state evidence and action-ranking harness that can invalidate a mechanism before training. It supports all four gap definitions while keeping the evaluation target explicit. Any resulting method still needs an independent full navigation evaluation; this assignment provides evidence and scoped opportunities, not completed empirical validation.
