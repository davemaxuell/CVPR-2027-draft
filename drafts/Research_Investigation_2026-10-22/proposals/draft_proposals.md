# Research proposals for independent challenge

**Assignment 07 synthesis complete; draft prepared for independent review. Cutoff: 22 October 2026.** Final evidence from assignments 01–06 is incorporated. This is not a novelty certification or an empirical validation: independent evaluations 08–10 remain pending, and the recommendations below can be narrowed or rejected.

The most defensible starting point is a **supervision audit before new model training**. Several existing methods improve aggregate success while the supervision target, memory update or future representation still answers a weaker question than the instruction requires. The proposed experiments first test whether that distinction occurs often enough to matter. A mathematical counterexample establishes possibility; it does not establish prevalence or an exploitable benchmark gap.

Four candidates use different interventions: recovery labels, data selection, future prediction targets, and persistent memory updates. Their provisional priority is **P1, P2, P4, P3**. P3 is deliberately marked speculative because its representational premise is clear but practical relevance and novelty are weaker. No expected numerical benchmark gains are asserted.

| ID | Simple name | Main intervention | First question that can kill it |
|---|---|---|---|
| P1 | Recovery labels | Replace geometric-only recovery labels with history-conditioned sets of efficient acceptable actions. | Do independently annotated deviations actually disagree with existing teachers, after legitimate route alternatives are allowed? |
| P2 | Validity and difficulty | Select grounded instruction–route pairs using independent validity evidence while preserving valid difficult examples. | Does validity information improve selection beyond compatibility, success filtering and ordinary data-quality baselines? |
| P3 | Joint future events | Predict ordered event co-occurrence on one continuation, rather than relying only on additive expected future semantics. | Is there useful headroom after controlling representation size, supervision, candidate actions and a direct future-question model? |
| P4 | Failure-aware memory | Separate incorrect goal-identity evidence from control/arrival failure before writing negative semantic memory. | Do erroneous negative writes cause measurable downstream harm, beyond what freezing or quarantining all failed-episode writes fixes? |

All four share the [experimental control rules](../appendices/Experimental_Control_Rules.md), [baseline map](../appendices/Baseline_Map.md), [premise audit](../appendices/Previous_Idea_Claim_Audit.md), and source-specific dossier limits. The complete 132-entry supplied-material role map is in [coverage review](../evidence/07/coverage_review.md). New preparation sources and their deliberately limited review scope are in [additional source cards](../evidence/07/additional_sources.md).

## P1 — Recovery labels

### Hypothesis and target

**Falsifiable hypothesis.** For route instructions whose remaining requirements depend on the observed action history, training on a history-conditioned set of efficient acceptable recovery actions will reduce incorrect recovery supervision and improve held-out recovery completion, compared with geometric rejoining, an nDTW teacher, and a reversible chunk tracker using the same trajectories, labels, architecture and interaction budget. It need not improve simple shortest-path goal instructions.

The target is **teacher aliasing**, not a claim that all navigators ignore history. CorrectNav projects a deviation to a reference segment and supervises its suffix; BudVLN chooses the furthest visited reference anchor. Those rules can disagree with unfinished procedural requirements. FGR2R, M-TRACK, Landmark-RxR, AwareVLN and Progress-Think already represent progress or milestones, so another scalar progress head is insufficient. [CorrectNav method and bottleneck](../dossiers/correctnav.md), [FGR2R](../dossiers/fgr2r.md), [M-TRACK](../dossiers/mtrack.md), matrix **05-R01/R03/R04/R06/R08**.

The coordinator's SPL geometry check **ROOT-C03** motivates measuring semantics separately; it is not evidence that any instruction was violated. A new original finite graph check shows that “visit A, then reach the goal” can require different next actions at the same junction before and after visiting A. This is an elementary history-dependence result, not a new theorem or a measured failure of DUET. [Check setup/results](../evidence/07/mechanism_check_results.json).

### Mechanism

Create a small, auditable instruction grammar for the initial study: ordered landmark visits, explicitly permitted alternatives, and a limited set of procedural events supported by annotations. Each event has a witness in the observed trajectory, not merely a nearby reference waypoint. Human annotators validate the event definitions and distinguish unresolved grounding from a completed event. Exclude instructions whose semantics cannot be reliably annotated; report the excluded share. Do not impose exact route replay on instructions that permit alternatives: the original R2R annotation protocol prioritizes reaching the destination, and M-TRACK explicitly notes that procedural commands are rare in existing datasets. Those facts make natural-prevalence measurement essential.

During training, use the full training graph and these event labels to construct a recovery oracle on the product of physical state and instruction state. The teacher can return several equally efficient acceptable actions. Two histories ending at the same pose may receive different labels. A reference route remains one example of satisfying the instruction, rather than the only accepted string of viewpoints.

At deployment, a learned history encoder estimates the instruction state from **available observations and executed actions**. Its output conditions the existing graph navigator. No reference suffix, full unseen graph, event ground truth or future image enters inference. Start with a soft state representation, not a hard action ban: uncertain grounding must not permanently forbid the correct branch. Any benefit from oracle parsing or oracle state is reported separately from deployable behavior.

```text
Training:
  validated instruction events + annotated history + training graph
      -> product-graph teacher -> acceptable action set and state labels
  observed history + instruction -> state estimator -> graph policy
      -> supervised losses

Inference:
  observed history + instruction -> estimated instruction state
      -> ordinary observed graph/candidates -> action -> real observation
```

### Formulation

Let the navigation graph be G=(V,E), with positive edge costs. An instruction has a finite automaton A_I=(Q,q0,delta,F). Q is an instruction-state set, F its accepting states, and delta consumes a grounded event vector. The formal language and product graph are **established tools**, not the claimed novelty. A macro action a may traverse several edges; delta must consume all actual intermediate events, not only the destination.

For an annotated history h_t, let q_t be its verified automaton state and v_t the current node. Let V*(v,q) be the minimum remaining travel cost to an accepting state under the annotation's semantics. For a candidate macro action with cost c(v,a), endpoint v', and updated state q', define

\[
Y_t=\{a: c(v_t,a)+V^*(v',q')\le V^*(v_t,q_t)+\epsilon\}.
\]

Use epsilon=0 initially, allowing exact cost ties. This is a set of **efficient acceptable actions**, not every semantically valid detour. Unreachable or unresolved cases receive no invented expert label. STOP is allowed only when the annotated completion condition holds. The shortest-path oracle is valid only for this specified graph, cost and event model.

Let b_phi(q|h_t,I) be the learned categorical state distribution, with |Q| entries, and e(q) a learned d-dimensional state embedding. Supply sum_q b_phi(q)e(q) to the policy. For unambiguous labelled states:

\[
L_{state}=-\log b_\phi(q_t|h_t,I),\qquad
L_{action}=-\log\sum_{a\in Y_t}\pi_\theta(a|h_t,I,b_\phi).
\]

Use L=L_action+lambda L_state plus the same ordinary-task loss as each control. The set loss avoids punishing a correct tied action. It does **not** guarantee that every member receives useful mass, that b is calibrated, or that the resulting policy obeys the instruction. For ambiguous labels, either use an independently adjudicated acceptable state set with a clearly labelled partial-label loss or mask that state loss; do not silently call ambiguity a negative. Keep a separate ambiguous evaluation stratum.

### Closest prior art and narrow novelty claim

This is not the first use of automata, history-conditioned task states, grounded milestones, alternative actions, or interactive imitation learning. In particular, [Ground-Compose-Reinforce](https://arxiv.org/abs/2507.10741v2), §§3–6, already learns grounding and uses reward-machine states. [HAMT](../dossiers/hamt.md) already introduces R2R-Back, which requires reaching the original destination before returning to the start; its extended PDF p 16, §§B.1–B.2, explicitly prevents motionless success and uses the full reference route length for SPL. Table 7/p 9 compares models retrained on this history-dependent task. Thus neither history-dependent task evaluation nor a return requirement is a new contribution here. FGR2R explicitly mentions reversible progress as future work. M-TRACK explicitly acknowledges procedural history and alternative task orders. GC-VLN/HSGM/CA-Nav and IDEAL-VLN add constraints or re-anchoring.

The candidate contribution is narrower: **an independently validated recovery-label protocol and a matched teacher intervention on photorealistic VLN deviations**, with same-pose/different-history pairs and set-valued labels. A new policy module earns a method claim only if it adds value over the corrected teacher alone and over a straightforward reward-machine baseline. If prior work already supplies this combination and evidence, retain only a replication/diagnostic contribution or reject P1.

### Minimal implementation and cost

Start from [DUET, commit 93e8b233164bc079a6db48b8a0a78d123ec8de41](https://github.com/cshizhe/VLN-DUET/tree/93e8b233164bc079a6db48b8a0a78d123ec8de41). The inspected `_teacher_action_r4r` in `map_nav_src/r2r/agent.py:239–285` selects an exact reference step, nDTW-based destination, or goal-distance destination; `agent.py:404–423` constructs and applies the action targets. Add a separate oracle module and auditable annotation loader, then replace this label call for the controlled experiment. Add the state token in `map_nav_src/models/vilmodel.py` only after the teacher diagnostic passes. `graph_utils.py` already routes through visited nodes; no novelty is claimed for returning.

Some recovery tasks require choosing a visited destination directly. If the candidate set is expanded, expand it identically for **all recovery controls**, retain actual routing costs, and report the native DUET interface separately. Otherwise an action-space change can masquerade as better labels. Dynamic programming costs scale with |V||Q|; arbitrary automata may grow exponentially with instruction structure. Begin with short verified structures and record label-generation time and annotation time. No inference LLM is required in the first DUET experiment, but event-state estimation still adds parameters and compute.

The final release audits constrain baseline implementation: FGR2R exposes data and a chunker but not the described navigation/shift predictor, and the inspected M-TRACK snapshot has an unwired simplified checker, a missing launcher target and an AST parse failure. Use their papers as prior art; describe any reconstructed baseline as a paper-derived reimplementation and publish its validation. These incomplete releases do not establish that the published methods fail. The first experiment remains based on the inspected DUET teacher, not on those missing components. See [FGR2R](../dossiers/fgr2r.md), [M-TRACK](../dossiers/mtrack.md) and the [follow-up audit](../followups/05_review.md).

### Experiments, falsification and risks

1. **Cheapest gate: teacher audit.** Sample natural deviations from frozen baseline rollouts and separately construct controlled paired histories. A proposed pilot is 200 natural deviation checkpoints plus 100 controlled pairs, with two independent annotators and adjudication; these are planned counts, not collected data. Label the remaining obligations and acceptable next actions while blinded to the teacher. Match current pose, visible scene and instruction where the design requires it, while preserving the genuinely different history. Report agreement, excluded/ambiguous cases, natural incidence, and teacher error by deviation type. Synthetic pairs must not be used to estimate natural prevalence.
2. **Label-only experiment.** Use one fixed rollout dataset and common initialization. Compare geometric teacher, current DUET SPL/nDTW teachers, a reversible chunk pointer, a milestone/reward-machine teacher, and the proposed annotated teacher. Match action candidates, sampled states, human-label budget, optimizer updates and ordinary-task data. First keep the policy architecture identical. For controlled closed-loop recovery, apply common externally specified disturbances; policy-dependent perturbations can otherwise change the test difficulty across methods (matrix **05-R10**). Report natural self-induced deviations separately.
3. **State-estimator experiment.** Compare no new state token, learned state, oracle parse, oracle state, shuffled state labels, and current-view-only state prediction. This separates better supervision from an extra architecture or privileged information.
4. **Held-out evaluation.** Use scene-disjoint R2R plus non-shortest RxR/R4R cases, the established R2R-Back history task, and validated NavNuances-style contrasts, with correctly charged return travel. Preserve R2R-Back's stage-aware success and full-route SPL conventions, and report it separately from newly annotated deviations. Report ordinary SR/SPL, nDTW/sDTW, explicit ordered-obligation completion, recovery success, extra distance, and preserved correct behavior. Report paired episode/scene uncertainty and model-seed variation separately.

Reject the mechanism if teacher disagreement is negligible after annotation, if labels are too unreliable, if a simple reversible/milestone teacher removes the gap, or if gains disappear under equal samples and candidates. Reject an architectural claim if label correction alone explains the gain. The principal risks are parser/grounding error, expensive annotation, product-state growth, and a benchmark slice too rare to justify a general method. The current workspace has annotations/connectivity but no established visual simulator assets or checkpoint, so the labelled-rollout stage remains unexecuted.

## P2 — Validity and difficulty

### Hypothesis and target

**Falsifiable hypothesis.** Within one fixed generated instruction–route pool, selection that uses independently audited semantic validity and preserves valid difficult examples will improve held-out semantic instruction following per accepted training example compared with navigator-success, compatibility-only, random, and standard noisy-label selection controls. This is a hypothesis about **selection at fixed data and compute**, not that more synthetic data makes a policy ignore language.

SRDF and RSI-Nav provide useful model-relative filtering, but rollout success is not an independent label of instruction truth. RCM Figure 6 directly illustrates similar compatibility reward for a correct and an incorrect-stop trajectory. APS already targets difficult examples; learned compatibility ranking predates SRDF. These observations establish a possible confound, not its frequency or a proven loss of performance. [SRDF](../dossiers/srdf.md), [RCM](../dossiers/rcm.md), matrix **05-D01/D02/D06/D07/D08**, claim **rcm-C01**. RSI-Nav Appendix H and its source-path-versus-verification-rollout distinction are important controls.

### Mechanism and formulation

For each candidate i=(instruction I_i, reference route tau_i, observed route evidence O_i), distinguish:

- v_i in {valid, invalid, unresolved}: whether the route evidence supports the instruction under the stated task semantics;
- d_i: difficulty for a **frozen, named navigator**, measured separately by route-following loss or rollout behavior;
- w_i: the actual evidence witness or contradiction for the important clause(s).

Train a small validity scorer p_phi(v=valid|I,tau,O) on a scene-disjoint audit subset labelled without exposing navigator scores. Include minimal semantic counterexamples only when their physical/linguistic validity is verified. A second model agreeing with the generator does not create an independent truth label. Unresolved annotations stay unresolved and are tracked separately.

Partition the pool by difficulty and semantic type. Under a fixed accepted count B, select high-validity examples subject to explicit quotas that retain difficult valid strata. One transparent selector is

\[
\max_{s_i\in\{0,1\}}\sum_i s_i p_\phi(v_i=valid),\quad
\sum_i s_i=B,\quad \sum_{i\in D_k}s_i=B_k.
\]

D_k form a **disjoint, exhaustive partition of the eligible candidate pool** and sum_k B_k=B, with each B_k no larger than its stratum. Implement this as a declared cross-product of difficulty bins and a mutually exclusive semantic label/signature; publish the assignment rule and how rare cells are merged. Each candidate belongs to exactly one D_k. Overlapping semantic tags require a different constraint system and are not silently substituted into this formula. This is ordinary constrained selection, **not a claimed new optimization algorithm**. Compare a fixed quota with data-driven quotas using a separate development set. The scorer's probabilities are empirical estimates; selecting by them does not confer a correctness guarantee. Cross-fit scorer training and selection evaluation by scene/route families, and do not use the final evaluation set to tune quotas or thresholds.

The more substantive component is an evidence interface: each accepted difficult example retains a clause-to-view/transition witness, so a mismatch can be reviewed rather than hidden inside an aggregate navigator score. One concrete prototype encodes C instruction clauses and M observed route units, with time/order and transition endpoints retained. A clause-to-unit attention matrix alpha in R^(C by M) has rows summing to one; a separate clause head predicts supported, contradicted or unresolved. For a clause with an adjudicated witness set W_c, add `-log(sum_{u in W_c} alpha_cu)` and clause-status CE to the three-way pair-validity CE. Mask witness loss where no witness is established. A recorded non-detection is not automatically a contradiction: it needs an explicitly justified visibility/detectability annotation. The validity head pools the clause/evidence representation and supplies the selection score.

This is a specified, ordinary alignment-supervision implementation, not a new attention loss. [GELA](../dossiers/gela.md) is a direct grounding-supervision precedent. Compare a witness-trained scorer with the same architecture and human validity labels but no witness loss, as well as a GELA-style alignment control. Account for the additional clause/witness annotation time; equal pair-label count alone does not equal human effort. The navigator trains on ordinary action CE with the chosen subset; the validity model and witness generator are absent at inference. Initial scoring costs scale with the C by M evidence interface on cached route features.

### Novelty and benefits

Data cleaning, curriculum learning, hard-example mining, and clean-validation reweighting are established. Include [Dataset Cartography](https://aclanthology.org/2020.emnlp-main.746/) and [Learning to Reweight Examples](https://proceedings.mlr.press/v80/ren18a.html) as outside-VLN controls, alongside Marky human wayfinding audits, APS, CCC, compatibility ranking, SRDF and RSI-Nav. Their detailed implementation comparison is reserved for evaluation 09/10; only the primary records and, for reweighting, the author method page were consulted in this preparatory review.

[GSA-VLN](../dossiers/gsavln.md) is another close construction precedent. Appendix A.2, PDF pp 16–17, uses an EnvDrop speaker trained with validation data, a DUET selector trained on unselected paths from the same evaluation environments or two-fold path cross-validation, and GPT-4o corrections of failed instructions using route views and path visualizations. This is disclosed benchmark construction, not automatic evidence of leakage into the evaluated navigator. It does mean that GSA's generated instructions cannot be treated as a navigator-independent validity test without auditing their construction. Compare the proposed scorer against this existing model-selection-plus-visual-correction approach; freeze any correction budget and distinguish selection from changing the candidate pool.

[Meta-Explore's supplementary Table 2 and §C.3, p 4](../dossiers/metaexplore.md), also show that category-label normalization changes REVERIE results. Freeze label cleaning and object-category processing across selection controls, and include a correction-only comparison where relevant. A gain from changed category processing is not evidence that the new selector identifies valid difficult instructions. Any normalization must preserve distinctions required by the particular instruction rather than assume every merged category is interchangeable.

The candidate contribution is **a measured separation of route-language validity and policy competence, plus a witness-supervised selector that preserves useful hard data under equal budgets**. Without a downstream gain over ordinary quality/difficulty selection, it is an informative dataset audit rather than a novel learning method. Potential benefits are reduced semantically wrong supervision and better retention of rare compositional examples. Costs move into annotation, scorer training and evidence extraction; these must be counted, not described as free data quality.

### Minimal implementation

Use the same pinned DUET loader/trainer as P1 for the first equal-count study, with a new manifest that records selected pair IDs, audit labels, evidence witnesses, score version and difficulty-policy checkpoint. The policy loss stays unchanged. The [RSI-Nav release, commit 5e8fa2c74cf8e7dc305b33088a963b200e20805d](https://github.com/OpenGVLab/RSI-Nav/tree/5e8fa2c74cf8e7dc305b33088a963b200e20805d) provides an inspectable comparison path: `evo/evo_rollout.py:1490–1540,1570–1600` filters verification outcomes and `evo/evo_utils.py:1013–1105` constructs later supervision. Do not port its large VLA training recipe for the initial diagnostic. An original-SRDF reproduction is not established by its README-only snapshot; any recreated filter must be labelled a paper-derived implementation.

Freeze the generation pool, visual features, backbone, initialization, accepted count, optimizer updates, path/scene mixture and token budget. Compare DUET, DUET+ScaleVLN and matched later systems only after these variables are reconciled. SkillNav's Temporal DUET and union-data controls prevent assigning all data gains to expert routing.

### Experiment and stop rules

**Cheapest gate:** audit a balanced sample across the four provisional validity/difficulty cells, with two independent annotators blinded to model behavior. Sample the natural pool separately to estimate prevalence with correct sampling weights. Measure false acceptance of invalid easy pairs and false rejection of valid hard pairs at equal acceptance rates. Include random, navigator-success, nDTW/SPL, compatibility, training-dynamics and witness-free validity scorers. Do not construct invalid examples only through trivial word swaps that the selector can recognize stylistically.

Only if this gate passes, train equal-sized subsets from one fixed pool. Compare: random; success-only; compatibility-only; difficulty-only; validity-only; validity-plus-difficulty; and witness versus no-witness scorer at equal human labels. Match path lengths, scene frequencies and label/token counts, or report a factorial analysis when exact matching changes the scientific question. Evaluate on independently authored instructions, validated semantic alternatives and paraphrases, ordinary navigation, and non-shortest route fidelity. Do not evaluate only on the same generator's language.

Reject a method claim if the validity model merely reproduces success filtering, if the hard examples are mostly annotation errors, if a simple compatibility/clean-validation baseline matches it, or if improvement disappears with equal data and training compute. Reject a semantic-following claim if only action sensitivity changes. Label-quality gains without downstream gains remain an audit result. Main risks: human ambiguity, scene-dependent detector errors, selection overfitting, and extra supervision masquerading as a better selector. No validity labels, selector training or navigator experiment have been executed here.

## P3 — Joint future events (speculative)

### Hypothesis and the exact missing distinction

**Falsifiable hypothesis.** For decisions depending on two ordered future events on the **same continuation**, a compact predictor trained on their joint occurrence will yield better candidate ranking than an additive expected-semantic target of the same size and data budget, when available history contains enough information to predict the distinction. The claim is restricted to that slice; it does not imply universally better world models or knowledge of hidden facts.

NavQ predicts a discounted average of future caption features under a specified shortest-continuation distribution. Its source explicitly aggregates node features across downstream branches; its own gamma and oracle-feature experiments leave useful but non-diagnostic headroom. [NavQ §3, Tables 3–4 and pinned dataset code](../dossiers/navq.md), claims **navq-C01/C02/C03/C04**, matrix **06-W01/W03**. An expectation of additive features need not retain whether two events can occur together in order. The original finite check contrasts equally weighted futures {AA,BB} with {AB,BA}: every timewise event marginal and every tested discounted first moment agree, while P(A before B) is 0 versus 0.5. The analytic reason is identical marginals, not a special gamma. [Executed check](../evidence/07/mechanism_check_results.json).

This does **not** show that NavQ's full network cannot distinguish those cases: current/history branches can carry extra information, and real caption embeddings can be richer than the constructed event vectors. Real target-level and decision-level measurements are a prerequisite.

### Mechanism and formulation

For an observed history h, first candidate action a, a fixed continuation policy mu, horizon H, and a set of K grounded event queries e_1...e_K, predict both single-event occurrence and ordered joint occurrence. Let Z_i^k indicate that event k occurs at future step i. Define

\[
p_k=P_\mu(\exists i\le H:Z_i^k=1\mid h,a),\quad
J_{k\ell}=P_\mu(\exists i<j\le H:Z_i^k=1,Z_j^\ell=1\mid h,a).
\]

J is K by K and p has K entries. These are event probabilities under **mu**, not reachability certificates or an optimal-policy Q function. An A-B-A route can satisfy A before B and B before A; neither relation is inferred by multiplying marginal probabilities. Repeated occurrences and event identity need explicit labels. For the first prototype, restrict to two-event requirements and a small K; arbitrary long instruction satisfaction is outside the representation's claim.

Train an action/history/query-conditioned head with masked binary cross-entropy on independently labelled continuation events. Use identical sampled continuations to train comparison heads: NavQ-style discounted features, single-event occurrence, full short event sequences, and direct future-question answers. Optionally project p,J to the same downstream feature width, but charge the K^2 outputs and projector parameters. If K is instruction-dependent, include exactly the same instruction queries in the matched single-event and future-QA controls. Independent BCE heads need not produce mutually consistent probabilities at finite sample size; report violations of necessary bounds such as J_kl <= min(p_k,p_l), and compare a bounded parametrization or projection as an explicit variant. Do not call the raw heads a coherent joint trajectory distribution or use them for a formal risk certificate.

A simple navigator readout scores the relevant ordered pair or a learned combination of the few supported event predicates and then fuses it with the existing policy. The first experiment freezes the candidate generator and continuation policy to isolate ranking. Closed-loop receding-horizon execution changes the continuation distribution; evaluate this shift explicitly rather than claiming the training probabilities still describe the new policy. Generated futures remain model outputs, never independent observations.

### Prior art and novelty gate

Generic decision-aware world modeling is already occupied by CVAML, UA-NWM, [Semantic World Models](https://arxiv.org/abs/2510.19818v1), [What Must a World Model Distinguish for Planning?](https://arxiv.org/abs/2609.33030v1), and [CompactNav's reviewed v1](https://arxiv.org/abs/2607.23181v1). Predictive state representations, successor features, temporal logic and distributional models are also mandatory novelty comparisons. The new source cards document precisely which sections were read; their results/code have not been reproduced.

The candidate distinction is **a controlled test and compact target for branch-consistent ordered co-occurrence in language navigation**, not “predict semantics instead of images,” “optimize decision quality,” or “use a low-rank bottleneck.” If future-QA, distributional successors or a simple sequence predictor already implement the same target with comparable cost, this may be a useful baseline study rather than a new method. Novelty confidence is low until assignment 09's search.

### Implementation, experiments and falsification

Start from [NavQ commit 9465fe43c9c1583bd7ba390b6be30996ea4f5a5c](https://github.com/woyut/NavQ_ICCV25/tree/9465fe43c9c1583bd7ba390b6be30996ea4f5a5c). `code/Q_pretrain/dataset.py:189–222` creates the additive targets; replace/augment only this target construction first. `GPT_VL.py:236–239` contains the prediction head and MSE loss. Add a matched event head and mask-aware BCE. All branches, training scenes, samples and model widths remain controlled. Downstream integration requires separately verifying the selected navigation entrypoint; assignment 02 audited the Q-model most deeply and did not certify all downstream code.

**Cheapest gate:** create a small labelled branch-outcome dataset and compare oracle representations using one common frozen-capacity readout. Include matched marginal/correlated branch cases and ordinary cases. If full event sequences and J provide no decision headroom over the existing features, stop. If headroom exists, compare learned predictions and check whether the available observations actually identify it. An oracle representation gain with no predictable signal is not a viable method.

Next measure candidate ranking regret under a fixed continuation policy, false confident pair completion, Brier/log loss for the stated event, and cost per candidate. Hold candidate generation and real sensing fixed; compare single-event, joint-event, sequential, future-QA, direct action-value and capacity-matched history-only heads. Match available inputs, training cases and annotation budgets; make any change in supervision target explicit. Ablate shuffled pair targets, event order, additional labels without J, and larger first-order features. Test unseen combinations and scenes. For deployment, use an explicit route-following extension of DUET/RxR; NavQ's original REVERIE/SOON object-goal setting does not by itself supply ordered-route supervision. Report that task change.

Reject P3 if richer representation merely adds capacity, if ordinary history already carries the distinction, if direct future-QA/sequence prediction matches it, or if closed-loop policy shift destroys the ranking gain. K^2 output cost, event-grounding noise, sparse joint events and limited higher-order expressivity are substantial risks. No forecast model or navigation experiment has run. This is the weakest candidate and may be dropped after independent review.

## P4 — Failure-aware memory

### Hypothesis and evidence

**Falsifiable hypothesis.** When terminal navigation failures have mixed causes, separating incorrect goal-identity evidence from control/arrival failure before writing persistent negative semantic memories will reduce later wrong suppression and improve repeated-task navigation, compared with outcome-only writes, no writes, simple quarantine and existing tri-state memory, at the same memory and feedback budget.

The direct implementation premise is bounded: the inspected PSG-Nav release adds recent detections to a false-positive bank after an unsuccessful episode (`PSG_Nav.py:1537–1554`). This can misattribute a control failure to recognition, but the audit has not measured its rate or established which snapshot produced the paper. [PSG-Nav](../dossiers/psgnav.md), **psgnav-C02**. Its native task is category/open-vocabulary ObjectNav: another instance of the requested category must not be labelled wrong merely because it is a different object. Begin with category-claim errors in that setting. A particular-instance extension needs its own target identity annotations and must be reported separately. PolarMem, HAM-VLN, CMMR-VLN, Context-Nav and CLOSER already supply negative, uncertain, experience or verification mechanisms; they are required comparisons. Matrix **06-M03/A01/A02**, **05-S05/P03**.

A simple truth-table check illustrates non-identifiability: under the restricted contract S=D AND R, failure S=0 is compatible with correct identity D=1 and failed reachability R=0. This only refutes treating that outcome as a logical proof of wrong identity. It is not a model of every benchmark or an empirical poisoning rate. [Executed check](../evidence/07/mechanism_check_results.json).

### Mechanism

Represent a persistent entry as an **explicit claim with evidence and scope**, such as “this crop/track satisfies the requested category or specified instance,” “this approach is currently traversable,” or “the stopping condition was met.” A terminal failure may update the episode outcome record without automatically negating all three claims. Repeated frames from one track remain one correlated evidence unit.

Train a small attribution model on controlled simulator branches and audited real traces. It predicts separate correctness labels for identity, approach and completion when those labels are observable; uncertain cases remain unresolved. Training interventions vary the controller while keeping the recorded perception/identity evidence fixed, and vary target identity while keeping the controller protocol fixed. This separates the information needed for each claim. At inference, use only observed images/track associations, executed actions, pose/control residuals that the chosen platform actually supplies, and the declared feedback contract. Do not substitute simulator identity or future success for deployable inputs.

A negative semantic write requires high predicted probability that the **identity claim itself** is wrong and a recorded contradicting evidence item. Otherwise quarantine the failed episode as an unresolved outcome, or update the route/control record. New observations can revise a claim; stored failure is not permanent object absence. The conservative no-write/quarantine policy is a mandatory simple baseline, not an ablation to omit.

### Formulation and limits

For candidate claim c and observable evidence E, let q_phi^D(E,c) estimate P(identity claim is incorrect), q_phi^R estimate a defined approach failure event, and q_phi^C estimate a completion-check error. Use separate masked supervised losses:

\[
L_{attr}=\sum_{k\in\{D,R,C\}}m_k\,BCE(q_\phi^k,y_k).
\]

Here y_k is a training label and m_k indicates whether it is actually known. There is **no assumption that the three causes are mutually exclusive or independent**. For a controlled pair E,E' differing only in a verified controller intervention while retaining the same identity evidence, an optional consistency term is

\[
L_{inv}=\big(q_\phi^D(E,c)-q_\phi^D(E',c)\big)^2.
\]

This term applies only to such verified pairs. Natural failures that reveal new identity evidence must not be forced invariant. Let W(E,c) indicate a saved, claim-specific contradiction witness; then a proposed semantic-negative gate is `qD >= tauD AND W=1`. For a deployable prototype, W means that an actual reinspection crop/view is linked to the same track and the attribution head identifies a conflicting identity observation. It does not mean that the conflict has been certified as true. Terminal failure alone cannot satisfy this input condition. Human-verified witness labels may supervise training, but are not supplied during ordinary deployment. Keep a simpler gate using only witness presence as a baseline, since qD and a predicted contradiction can contain redundant evidence.

Thresholds are selected on a held-out development set. The output is not a calibrated guarantee under policy/scene shift; report its precision, coverage and uncertainty empirically. An unreliable witness or mistaken track link defeats the gate, so oracle-witness, predicted-witness and witness-presence-only results must be separated.

The contribution would be a **failure-attribution training and evaluation method for persistent navigation memory**, not the data structure alone. Bayesian negative information with visibility/occlusion modelling is old robotics prior art; see [Hoffmann et al., §4](https://faculty.kfupm.edu.sa/COE/mayez/ps-coe499/core/Vision-Localization/2005-08-ijcaj05ws-HoffmannSprangerGoehringJuengel-NegativeInformation.pdf). Neither a source tag nor a three-state memory is new. If removing an incorrect write rule entirely captures the gain, report a baseline correction rather than a scientific mechanism.

### Minimal implementation and evaluation

Use [PSG-Nav commit 181d878ff207845a31e542a19a6592925c1e9bab](https://github.com/Chenrf1121/psg-nav/tree/181d878ff207845a31e542a19a6592925c1e9bab), with the existing detector/planner frozen. Instrument `PSG_Nav.py:1484–1554` and the RAG manager's write/read interface. First log what would be written without changing actions. Create a separate typed outcome ledger and an attribution wrapper; do not silently rewrite the upstream baseline. Keep bank capacity, eviction, retrieval weights, model calls and scene-order exposure identical. Inspect the RAG manager implementation before editing; its full internals were not independently audited in assignment 07.

**Cheapest gate:** replay paired traces with (a) correct target but blocked/failed approach, (b) incorrect category detection with otherwise successful control, (c) occlusion, (d) identity-track swap, and (e) a truly moved object. Only in the separately specified instance-goal extension should a wrong same-class instance count as case (b). Count false semantic writes and later suppression of independently verified correct candidates. Compare current outcome-only writes, memory frozen, all failures quarantined, recency decay, tri-state/source-tag memory, and attribution. This can begin with stored observations; full navigation comes later. The natural prevalence must be measured separately from deliberately balanced perturbations.

If the gate passes, run repeated ObjectNav/instance-goal episodes with scene-disjoint calibration and evaluation, explicit reset/persistence conditions, randomized episode orders and matched feedback. Compare no external feedback, self-assessed feedback and oracle terminal feedback as different protocols. The first test shares the exact observed traces across methods; extra reasoning over the same wrong crop is not independent evidence. If active reinspection is later added, give controls matched new views or isolate its benefit and cost as a separate factor (matrix **05-R11**). Evaluate memory false-write precision, harmful suppression, stale-claim persistence, recovery time, false FOUND/NOT-FOUND, SR/SPL and total sensing/inference cost on **all episodes**. Use a fixed memory budget and fresh tasks; never allow evaluation labels to become undeclared training data.

Reject P4 if harmful writes are rare, if no-write/quarantine matches the model, if distinguishing failure causes needs unavailable privileged evidence, or if safer semantic updates leave obsolete facts indefinitely and harm overall performance. Controller noise, pose drift, track association, correlated observations and changing scenes are major risks. A trained classifier may merely move the attribution error upstream. No trace audit, attribution training or repeated-navigation benchmark has been run.

## Provisional recommendation and independent review questions

Begin with **P1's teacher audit**, because its central distinction can be annotated and falsified before model training, and it uses an inspectable graph-policy codebase. P2 is a useful second track if the fixed candidate pool and independent annotations are available; its strongest likely contribution is a controlled empirical finding, with method novelty contingent on witness supervision. P4 could be more valuable in persistent deployment, but requires a validated mixed-cause failure distribution. P3 should receive the most aggressive novelty and learnability challenge before resources are committed.

The next decisive evidence can change this ranking. A negligible P1 teacher-error rate moves priority to P2; a high harmful-write rate that quarantine cannot fix moves P4 upward; a large predictable ordered-event ranking gap with no equivalent prior method could rescue P3. Rejecting a candidate is preferable to filling a quota.

Assignment08 should audit state/teacher assumptions, set-loss degeneracy, future-policy conditioning, event probability semantics and failure-cause identifiability. Assignment09 should search independently for equivalent recovery teachers, partial-label imitation, semantic data selection, temporal predictive representations and fault-attributed memory. Assignment10 should inspect the actual selected entrypoints and assets, validate common action sets and budgets, and identify the cheapest executable test in this workspace. These are open review requests, not approvals.

All navigation benefits remain **untested hypotheses**. The only new executed evidence in this draft is three small original graph/arithmetic checks; the original source snippets were inspected, not run. Available RTX4070 laptop hardware is not usable by the currently installed CPU-only PyTorch build, and simulator/checkpoint/visual asset prerequisites remain unresolved.
