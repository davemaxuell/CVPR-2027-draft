# Instruction, supervision, grounding and recovery audit

Cutoff: **22 October 2026**. Assignment 05 investigates follow-up chains rather than repeating the original-paper dossiers. Its primary-source catalog contains 24 scoped reviews, including 21 papers outside the supplied corpus and three independently reread supplied papers. Six repository snapshots in its own directory and two original-paper snapshots from assignment 04 were inspected. No navigation benchmark, model checkpoint or fetched project script was executed.

The complete 37-row [bottleneck matrix](05_matrix.json) is also available as [CSV](05_matrix.csv). Precise bibliographic records, actual pages read, repository commits and access limits are in [sources.json](../evidence/05/sources.json); readable individual reviews are in [source_reviews.md](../evidence/05/source_reviews.md). The [original-paper scope record](../evidence/05/original_scope.json) covers 48 relevant dossiers and distinguishes direct PDF reading from another assignment's documented inspection. [Implementation notes](../evidence/05/code_audit.md) separate data, training, inference and evaluation paths.

## What survives the follow-up audit

The useful research opportunities are narrower than “add recovery,” “make the model use text,” or “verify before stopping.” Those mechanisms already have substantial predecessors. The strongest remaining questions concern **what a verification signal actually establishes**, **how recovery supervision represents unfinished instructions**, and **whether extra evidence pays for its acquisition cost**.

| Chain | What is already established | What the reviewed evidence does not establish |
|---|---|---|
| Speaker-Follower → RCM → compatibility scoring/APS/CCC → SRDF/RSI-Nav | Generation, learned matching, adaptive sampling and recursive filtering | Independent semantic validity of every retained pair, or preservation of valid difficult pairs |
| Self-Monitoring/Regretful → FGR2R/BabyWalk/Landmark-RxR/M-TRACK → CorrectNav/AwareVLN/Progress-Think/SeekVLN | Progress tracking, subtask gates, error-state training and evidence seeking | Correct obligation state after an off-path history, especially with repeated landmarks or permissible detours |
| DUET/FAST → MG-VLN | Backtracking and final visited-endpoint selection | That a new stopping head or extra return option is itself novel, or that endpoint rescoring has no physical cost |
| GELA/VLN-Trans/NavNuances → behavioral diagnostics/CAST/CAG/RSS | Grounding supervision, controlled language contrasts and stronger conditional guidance | A causal law that data scaling weakens language use, or that action sensitivity implies correct semantics |
| Context-Nav/PolarMem/HAM-VLN → CLOSER/DRIVE-Nav/VLN-NF | Uncertainty states, active verification and coverage-aware NOT-FOUND | A general calibrated certificate of target absence under incomplete views, identity errors and detector misses |

Each statement is bounded by the reviewed methods/settings; this is not a claim that the entire literature has been exhausted.

## Premises that must be corrected before proposing a method

DUET already permits return through its graph and selects a cached STOP endpoint. FAST already ranks visited goals. Topological Planning already trains a separate binary stopping classifier; NavRAG's released DUET loss weights STOP targets. The research problem must therefore identify a failure of existing targets or score comparisons, not assume these mechanisms are missing. See the original dossiers and matrix rows S01–S02.

VLN-Trans introduces latent translator representations, not an actual instruction rewrite at inference. GELA is supervised grounding pretraining. SkillNav's full R2R SR is already matched by its Temporal DUET control; routing and expert-input text must be separated. ScaleVLN and RSI-Nav contain fidelity improvements, so “more data makes navigation language-blind” remains an untested hypothesis. These corrections are essential controls for any language-focused proposal.

Success-based filtering is useful evidence of model utility, but utility and label validity are different variables. RSI-Nav itself acknowledges this distinction in Appendix H; its implementation also separates the successful verification rollout from the source-path actions subsequently supervised. A proposed data method must outperform independent compatibility and adaptive-sampling baselines, not merely rename recursive filtering. See D01–D09 and the pinned RSI-Nav audit.

## Prioritized gaps and the cheapest useful tests

### 1. Recovery supervision that distinguishes location from unfinished obligations

**Problem.** Two histories can end at the same pose while satisfying different parts of an instruction. A nearest-reference or furthest-visited anchor can then assign an incorrect remaining suffix. This is a mechanism-level deduction from the inspected teachers, not a measured prevalence claim.

**Strongest challenges.** FGR2R explicitly anticipated reversible/multi-step shifting; M-TRACK already checks grounded milestones and acknowledges history-dependent procedural commands. AwareVLN and Progress-Think supply richer progress signals, while SeekVLN actively gathers evidence. A new contribution must go beyond another progress variable or milestone checklist.

**Decisive first test.** Build a small independently annotated set of naturally occurring deviations plus controlled paired histories, keeping the final pose, instruction and visible evidence matched where appropriate. Annotate acceptable next actions and completed obligations, allowing multiple valid routes. Compare the existing geometric teacher, a reversible chunk pointer, a milestone gate, and the proposed state representation. Stop if the proposed distinction is rare, labels are unreliable, or current teachers already agree with the acceptable-action sets. Do not force history invariance when history changes task meaning.

**Success criterion.** Lower incorrect recovery-label rate, followed by better held-out recovery at equal training samples and interaction cost. Report ordinary-task regressions and annotation agreement. Confidence is high that the distinction is real; frequency and practical effect remain untested.

### 2. Separate semantic pair validity from navigator competence during data selection

**Problem.** A failed rollout can mean a valid difficult instruction or an invalid pair. A successful rollout can ignore an erroneous relation. Model-relative acceptance conflates these cases.

**Strongest challenges.** Learned compatibility ranking, APS, CCC, SRDF and RSI-Nav already cover much of selection and self-improvement. Grounded generation and human wayfinding audits also precede current self-training. Adding a second model is not sufficient evidence of independent truth.

**Decisive first test.** Draw one fixed candidate pool and independently annotate a balanced validity × difficulty sample, with reviewers blinded to navigator success. Measure false acceptance and false rejection for all selectors. Then train equal-count subsets with fixed compute and common initial weights. Start with annotation and selector diagnostics; do not pay for large-scale training before a measurable separation exists.

**Success criterion.** Retain more valid difficult pairs while reducing invalid easy pairs, then improve language-conditioned decisions on independently authored evaluation instructions. Purely better self-generated-target SR is insufficient. Evidence strength is high for the confounding mechanism; generalization benefit is untested.

### 3. Target-specific evidence decisions under a real sensing budget

**Problem.** Looking at many objects or accumulating several uncertain frames does not ensure that the queried instance would have been detected if present. Identity errors can also contaminate accumulated evidence.

**Strongest challenges.** CLOSER already performs active verification; SeekVLN already learns from paired intervention continuations; VLN-NF already evaluates coverage-aware NOT-FOUND. Active Visual Information Gathering already includes an exploration-cost penalty. These substantially narrow the opportunity. KnowNo/Explore until Confident supply calibration precedents with assumptions that must be preserved, as audited by assignment 04.

**Decisive first test.** On paired feasible/infeasible goals, stratify occlusion, repeated-instance identity, observed coverage and target detectability. Match total rotations, travel, model calls and tokens across trigger policies. Report false FOUND, false NOT-FOUND, deferred decisions and all-episode cost. Include fixed-rate/random sensing, original verifier and oracle-benefit ceilings. Charge unsuccessful episodes as well as successes.

**Success criterion.** A better risk–cost frontier on held-out scenes, not simply higher coverage or more queries. Any risk guarantee requires explicit distribution/policy assumptions and an accounting of forced fallbacks. Practical value is plausible; novelty confidence is lower after the recent collisions.

### 4. Correct language control under both valid contrasts and paraphrases

**Problem.** A policy should change its action for a meaning-changing instruction and preserve it for a meaning-preserving reformulation. Neither behavior alone establishes grounded control.

**Strongest challenges.** NavNuances and behavioral intervention work already provide paired/atomic diagnostics; CAG/CAST occupy same-scene alternative instructions; RSS occupies stronger conditional guidance. GSA-VLN already evaluates style rewriting, and VLN-Trans is a distinct latent mechanism.

**Decisive first test.** Extend existing diagnostics only where they lack composed off-path or repeated-instance conditions. Use correct-action sets and trajectory-level clustering, with matched data and forward-pass budgets. Compare original text, valid paraphrases, changed goals/relations, and any added spatial captions as separate conditions. Reject the idea if sensitivity increases without correctness or ordinary-task robustness.

**Success criterion.** Joint contrast accuracy and paraphrase invariance with no hidden information advantage. This is a strong evaluation requirement; a standalone method/benchmark contribution still needs independent novelty review.

### 5. Cross-time endpoint score comparability

**Problem.** Scores cached under different candidate sets and histories need not be calibrated for a final ranking. This is a plausible issue, not yet an established source of DUET failures.

**Strongest challenge.** MG-VLN already learns an instruction-conditioned visited-endpoint ranker. Merely recomputing STOP or adding BCE is weak novelty.

**Decisive first test.** Freeze visited trajectories, endpoint labels and the return planner. Compare existing cached scores, explicit endpoint reranking and calibrated alternatives, charging the actual path to the selected endpoint. Reject the direction if the visited pool lacks good endpoints or existing reranking already removes the error.

**Success criterion.** Better endpoint selection at unchanged exploration, followed by end-to-end improvement after return cost. This is lower priority until an offline error audit establishes headroom.

## Executed checks and limits

Two self-contained finite arithmetic checks ran successfully: arithmetic probability extrapolation differs from log-density guidance, and raw future-logit ranking can change under state-wise additive offsets that preserve local action probabilities. [Setup and results](../evidence/05/finite_check_results.json) state the exact numbers. They do not show a CAG implementation bug, a measured ULN failure rate, or benchmark gains.

Several frontier papers lacked inspectable official implementations. Their paper experiments remain author-reported, with full-text scope recorded. Existing source files were inspected, not installed or executed. Data/checkpoint completeness and actual resource requirements still need assignment 10's feasibility audit. Assignment 09 must independently challenge the narrowed gaps, including alternative terminology outside VLN.

Final original-source integration narrows practical baselines. FGR2R releases its data/chunker but not its navigator; the inspected M-TRACK release lacks a complete executable paper pipeline. NAP identifies its release as an old version and leaves final settings/CE code pending. ULN's released uncertainty network differs from Eq. 6 and its parallel exploration trace requires explicit physical-cost replay. The slow4fast release contains mock reflection and placeholder embeddings without the paper's full integration. These affect reproduction plans, not the prior-art status of the papers or the validity of their privately produced results. See the [integration record](../evidence/05/final_integration.md), [ULN audit](../evidence/05/uln_code_audit.md), and [NavNuances audit](../evidence/05/navnuances_code_audit.md).
