# Independent logical and mathematical evaluation

Assignment 08. Cutoff: 22 October 2026. Evaluated all four proposals against primary paper passages and pinned source files, then executed nine groups of original isolated checks. **None of the four proposals is an established navigation result.** P1, P2 and P4 are reasonable diagnostic research candidates after the corrections below; P3 remains a speculative representation study. No proposal needs to be retained to fill a quota.

| Proposal | Decision | What survives | Required correction before implementation |
|---|---|---|---|
| P1 Recovery labels | Revise; proceed with the audit | Identical physical poses can have different optimal recovery actions after different event histories. A product-state oracle can express this. | Audit complete teacher continuations for semantic failure; separate validity from efficiency; solve for the realizable action interface; fix the termination contract and ambiguous-label handling. |
| P2 Validity and difficulty | Revise; proceed with the audit | Navigator competence and instruction–route validity are different quantities; disjoint-stratum selection is well defined. | Distinguish alternative witnesses from jointly required evidence; handle selective/unresolved labels; report quota-induced quality tradeoffs. |
| P3 Joint future events | Revise; retain as speculative | Additive first moments and ever-event marginals can both lose ordered co-occurrence. | Replace the weak example with the matched-marginal example below; use “expected discounted sum” for NavQ; preserve continuation-policy conditioning and measure real target/readout headroom. |
| P4 Failure-aware memory | Revise; proceed only after a write audit | Terminal failure alone cannot identify a perception error. | Establish observable identifying evidence; address selective labels; evaluate the joint write gate rather than claiming calibration from either component. |

The decisions concern logical support, not novelty certification. Assignment 09 assesses novelty and assignment 10 assesses practical implementation. Formal deductions below have explicit assumptions; finite numerical checks are not general proofs, and neither is evidence of benchmark improvement.

## Evidence and execution scope

I independently read CorrectNav's deviation and recovery construction (PDF pp. 3–4); NavQ's target definition and training (pp. 4–5); PSG-Nav's memory method and static-memory ablation (pp. 6, 9); RSI-Nav Appendix H; FeedTTA Eqs. 3–9 and Appendix B.1 (pp. 4–5, 13–14); ATENA Eqs. 1–5 and supplement A.7; CPNAV §§III.D–F (pp. 4–5); and UAGM Table 11 on an actual rendered PDF page. I also inspected the precise DUET, NavQ, RSI-Nav, PSG-Nav and ATENA source passages listed below. Additional contextual paper passages were read but are not represented as whole-paper reviews by this evaluator. These scopes supplement the original-paper assignments; they do not replace them.

Scoped read records include local SHA256 hashes in [source_reads.jsonl](../evidence/08/source_reads.jsonl). The UAGM table and CPNAV equation were visually checked in [UAGM page 19](../evidence/08/uagm_table11.png) and [CPNAV page 5](../evidence/08/cpnav_page5.png). Supplied UAGM PDF SHA256 is `a4a90b10da489a4fd6b88eedcc73ceeae413732ce55d511b5e25494de73dff3c`; assignment 01's independent version record identifies it with arXiv 2605.26503v1.

All executed checks are in [logic_checks.py](../experiments/08/logic_checks.py), with environment versions, script hash and exact outputs in [logic_results.json](../experiments/08/logic_results.json). They use original Python/NumPy/SciPy calculations, not imported project modules. **No downloaded project code, learned model, simulator, navigation benchmark or training job was executed.** One initial assertion used exact equality for two floating-point representations of 0.9; it was corrected to a numerical tolerance. The final nine check groups pass. This was a test-harness correction, not a change to a scientific result.

## P1: Recovery labels

### Valid core and its boundaries

CorrectNav projects the first detected deviation onto an interpolated reference path, identifies the reference segment, and resumes through the next reference waypoint and suffix (PDF p. 4, “Error-correcting Trajectory”). That geometric construction does not itself prove that every earlier procedural requirement has been fulfilled. This establishes a question to audit; it does not establish a measured error rate. In the original toy graph J–A and J–G, “visit A, then finish at G” has optimal action J→A before A has been visited and J→G afterward. The product states distinguish these histories; the physical node alone does not. Going to G first is still recoverable if the agent later visits A and returns. The result is elementary history dependence of efficient completion, not proof that every suboptimal first move violates semantics.

For a finite deterministic product graph with positive movement costs, known event transitions, a realizable action set and finite optimal remaining cost, Bellman's identity gives

\[
V^*(s)=\min_{a\in\mathcal A(s)}[c(s,a)+V^*(T(s,a))].
\]

Thus the proposal's zero-slack set is nonempty and contains optimal first actions **if the same action and transition contract defines both sides**. Following zero-slack movement actions decreases remaining cost by their positive cost, so a finite graph with a positive minimum movement cost cannot cycle forever before acceptance. This is a conditional graph-planning fact, not a guarantee for a learned policy, uncertain event detector or changing environment. STOP must be modeled as a terminal decision at a valid completion state, rather than as a zero-cost self-loop.

The executed check also demonstrates why nonzero slack changes this argument: epsilon=2 admits an unnecessary J→A action even after A was already visited; A→J is optimal, and repeatedly selecting that pair can loop indefinitely. Keep epsilon=0 in the initial study, as the draft already proposes. If slack is later introduced, add an explicit termination/progress rule or finite-horizon criterion rather than inheriting the zero-slack guarantee.

### Corrections required

**Semantic error and inefficient recovery are separate labels.** The set Y contains efficient acceptable actions, not all semantically valid continuations. An action outside Y can still complete every obligation after a harmless detour. Report at least: (i) semantically invalid/unrecoverable action or violated obligation, (ii) semantically feasible but cost-suboptimal action, (iii) optimal acceptable action, and (iv) unresolved. Teacher disagreement with Y alone is not a semantic-violation metric. Annotators must be blinded to the teacher and adjudicate the intended semantics independently.

For eventual-visit requirements in a connected reversible graph, almost every movement may remain semantically feasible if unlimited return travel is allowed. Therefore audit the **teacher's complete proposed continuation including STOP**, or an irreversible rejecting transition, to substantiate a semantic supervision error. In the toy graph, the geometric continuation J→G→STOP fails when A is unfinished; the single move J→G does not. Its shortest eventual satisfying completion costs 5, versus the optimum 3. If the experiment declares a remaining budget B, define a separate feasibility set `F_B(s)={a: c(s,a)+V*(T(s,a))<=B}` under the fixed transition model. J→G is feasible for B=5 and infeasible for B=3. This does not turn Y into a semantic set; Y is still the efficient subset. Budget choice must be preregistered and shared rather than selected to manufacture errors. The executed continuation check records these distinctions.

**The oracle must match the actual candidate interface.** DUET's `_teacher_action_r4r` iterates candidate destinations, excludes visited destinations when the supplied mask marks them, and uses either nDTW or current-to-candidate-plus-goal distance. If V* is computed on unrestricted primitive edges but the policy cannot select its optimal first transition, Y can be empty even when a longer feasible recovery exists. The toy graph demonstrates this by excluding A from the available menu before the mandatory visit. Define either a shared primitive/macro interface with all required actions, or solve the product oracle over the actual realized macro-action process. For a changing observed graph, the information state/candidate availability may also enter the oracle state. If an augmented interface is required, give it to every comparison and separately report native DUET. [DUET commit 93e8b233…, agent.py:239–285](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/agent.py#L239).

The minimal rigorous pilot is a **fixed common primitive-neighbor action interface plus STOP for all arms**, evaluated on the static annotated graph. Then `(v,q)` is sufficient for the teacher dynamics. Native DUET's observed map, visited mask and frontier menu depend on history, so native integration generally needs state `(v,q,m)` with m encoding the realized map/menu transition, or an explicitly frozen-menu subproblem. An oracle with `(v,q)` alone must not claim to solve that changing macro-action system. This pilot is a controlled teacher study, not a reproduction of native DUET's action space.

Here v must contain every physical variable needed for the transition/event model. A node-only landmark-visit grammar is the clean initial test. Orientation-dependent turns or control-state predicates require heading/control state or explicitly represented turn events; a position-only graph does not automatically model them. Likewise, the automaton state must contain all relevant procedural history under the declared grammar.

**The completion contract extends beyond the loss.** In this same release, teacher/sample training forces termination at the reference endpoint even if action 0 was not chosen; any termination may then append a return to the node with the largest cached STOP score. Merely replacing the target at lines 411–423 leaves these semantics unchanged. The proposed recovery protocol must specify stage-aware termination and how the executed final return is assessed. Consume every actual intermediate event along a macro route and any appended return; recompute completion from the final physical/event state. Distinguish budget/candidate exhaustion from legitimate completion. At inference, no oracle event labels may decide whether to terminate. All controlled recovery baselines need the same revised interface; native benchmark results remain separately identifiable. [agent.py:404–478](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/agent.py#L404). This concern was also independently found by assignment 10.

**Training and observation states must remain distinct.** The full graph can supply teacher labels during training. It does not make unseen landmarks or intermediate macro-route observations available to the student. The state estimator may be unable to recover the teacher state from its actual inputs. Define which event witnesses are available at each time; stratify perceptually ambiguous cases; include oracle-state versus observed-state comparisons. For instruction-specific automata, explain how state indices/embeddings are shared: an arbitrary state number is not a transferable semantic representation. A compositional encoding of the automaton/remaining clause descriptions, or a restricted canonical state schema, is an implementable choice. A per-training-instruction lookup table cannot substantiate generalization to unseen task structures.

### Loss audit

Writing p=softmax(z) and S=sum(a in Y) p_a, the action-loss derivative is

\[
\frac{\partial[-\log S]}{\partial z_j}=p_j-\mathbf1[j\in Y]\frac{p_j}{S}.
\]

The numerical central-difference check agrees to below 1e−8. This objective is differentiable for a fixed, nonempty label set and finite logits. It encourages total acceptable probability, not equal mass or exploration among tied actions. For example, almost all mass on one acceptable action and approximately 1e−8 on another can have essentially zero loss. If every action is labelled acceptable, its gradient is zero. These are expected properties of partial-label supervision, not defects if only membership matters.

Do not turn uncertain state labels into the union of every possible state's action set without qualification: an action valid for one unresolved state may be invalid for the actual state. Mask genuinely unresolved action supervision, adjudicate it, or specify a probabilistic ambiguity model and evaluate that claim separately. Masking state CE while supplying an oracle-derived hard action label does not by itself remove the ambiguity. All action/state losses remain ordinary empirical training objectives; no convergence, calibration or instruction-compliance guarantee follows.

**Decision:** revise the evaluation definition and execution contract, then run the blinded teacher audit before training. The strongest evidence is a logical possibility plus inspectable teacher behavior. Natural prevalence, reliable event annotation, deployable state estimation and downstream benefit are all untested.

## P2: Validity and difficulty

### Mechanism fits the proposed diagnostic question

RSI-Nav explicitly acknowledges that valid but difficult instructions can be rejected and semantic errors can survive successful execution (Appendix H, p. 19). Its release accepts by success/SPL thresholds and then constructs training samples from the original source frames/actions. That directly supports separating execution competence from semantic validity; it does not show the proposed selector improves training. [RSI-Nav commit 5e8fa2c…, evo_rollout.py:1509–1540 and 1578–1598](https://github.com/OpenGVLab/RSI-Nav/blob/5e8fa2c74cf8e7dc305b33088a963b200e20805d/evo/evo_rollout.py#L1509).

The fixed-budget selector is well defined when the D_k form a disjoint exhaustive partition and 0≤B_k≤|D_k| with sum B_k=B. Its optimum is simply the highest B_k validity scores in each stratum. An exchange argument proves this: replacing a chosen lower-scoring item by an omitted higher-scoring item in the same stratum preserves feasibility and increases the objective. Exhaustive enumeration of a six-item example confirms the implementation logic. No statistical independence between candidates is needed to maximize their *estimated expected number* of valid examples, since expectation is additive. The estimates still can be wrong, and maximizing valid count need not maximize learning benefit.

### Corrections required

**Witness loss needs the right logical connective.** The proposed `-log(sum_{u in W_c} alpha_cu)` is an existential/alternative-witness loss. If W_c contains two separately necessary pieces of evidence, it can be zero while all attention goes to only one. The executed example assigns essentially all attention to A and none to B; the union loss is zero. For a relation or ordered clause needing both, encode a *complete witness bundle* as one route unit, or define required groups and supervise each with an explicit joint representation. Distinct attention heads or a support graph are possible ordinary implementations. Merely replacing the loss by a sum of terms without accounting for a single row's normalization is insufficient to require each group to get unit mass: use separate distributions or appropriate coverage targets. Keep alternative complete witnesses distinct from jointly necessary sub-witnesses. The counterexample concerns what the loss enforces; a rich encoder might learn the relation despite this weak supervision.

**Unresolved labels are an epistemic category.** A three-way valid/invalid/unresolved head can predict the annotation outcome; it is not automatically the truth posterior of an inherently three-valued physical world. Hard-to-observe examples may be preferentially unresolved. Distinguish supported-validity precision among resolved cases, unresolved coverage and downstream outcomes. If binary validity or witness losses are masked, their population optimum is conditional on being labelled unless the missingness mechanism is ignorable given the inputs. The finite check constructs identical observed evidence with a true 50% positive rate but a selectively labelled 10% positive rate; masked BCE learns 0.1. Use a randomized independently audited subset, report label availability by difficulty/semantic type, and avoid unsubstantiated population calibration.

**Quota quality tradeoffs must be explicit.** A hard quota can force selection of low-validity members of a difficult stratum. Predefine whether the experiment holds B fixed and reports that risk, or applies a validity floor and permits infeasibility/abstention; these answer different questions. Do not silently refill from easy strata and then claim the quota was preserved. A quality floor and equal accepted count can be mutually infeasible.

**Annotation independence means procedural separation, not a probabilistic theorem.** Blind annotators to navigator scores and use held-out scenes/route families. Validity, difficulty and visual observability can still be statistically correlated. The proposal does not need them to be independent; it needs a validity assessment that is not defined by the navigator's own success. Equal sample count is also not equal annotation cost when one arm receives clause-level witnesses. The draft already recognizes this; retain that control.

**Decision:** revise witness semantics and selection/annotation contracts; accept the diagnostic audit as a testable question. The selection objective is correct under stated feasibility assumptions, but semantic labels, scorer calibration and downstream benefits remain untested. Improving label quality without a navigation gain would be a useful audit result, not proof of a better navigation method.

## P3: Joint future events

### What NavQ actually predicts

NavQ Eq. 8 is an **expected discounted sum** of future node features, not a normalized average or scalar Q-value. The paper uses task-agnostic caption embeddings and a shortest-continuation policy; released `dataset.py:189–222` groups nodes by the first edge of a stored shortest path, assigns branching weights, and sums 512-dimensional caption features with decay. `GPT_VL.py:236–239` applies MSE to the predicted feature. Therefore, the strongest controlled comparison must preserve the actual continuation distribution and target construction, not substitute arbitrary random walks or an unrelated semantic average. [NavQ commit 9465fe43…, dataset.py](https://github.com/woyut/NavQ_ICCV25/blob/9465fe43c9c1583bd7ba390b6be30996ea4f5a5c/code/Q_pretrain/dataset.py#L189), [prediction loss](https://github.com/woyut/NavQ_ICCV25/blob/9465fe43c9c1583bd7ba390b6be30996ea4f5a5c/code/Q_pretrain/GPT_VL.py#L236).

The original AA/BB versus AB/BA example matches timewise marginals but not the proposed p-only ever-occurrence control: the first distribution has p_A=p_B=0.5 and the second has p_A=p_B=1. It is therefore insufficient to motivate a benefit of J beyond p. Replace it with this stronger example.

### A corrected, stronger counterexample

Let 0 be a neutral event. Compare uniform distributions over three equally long continuations:

| Distribution | Three possible continuations | P(A ever), P(B ever) | P(A before B) |
|---|---|---|---|
| Left | AB0, B0A, 0AB | 1, 1 | 2/3 |
| Right | A0B, 0BA, BA0 | 1, 1 | 1/3 |

At **every** time position, both distributions assign probability 1/3 to A, B and 0. Each event occurs exactly once per continuation; repetition is unnecessary. For any fixed per-event feature map r and any fixed discount gamma, linearity gives identical `E[sum_t gamma^t r(Z_t)]`. Yet ordered joint occurrence differs. The executed check verifies all marginals, p and J exactly to the floating-point representation, and discounted sums for gamma 0, .3, .5, .7 and 1. The equality for arbitrary fixed gamma follows analytically from matching timewise marginals, not merely from this finite grid.

This establishes a representational distinction beyond both additive first moments and ever-event probabilities. It does **not** establish that actual NavQ caption vectors collide, that the complete NavQ navigator cannot recover order from other branches, that the distinction is predictable from h,a, or that relevant cases occur often in REVERIE/SOON. Rich caption features might encode additional context; the example uses deliberately limited event representations. Conditional on identical deployable observations with different hidden futures, no predictor can determine which world it inhabits beyond the available posterior. An oracle representation gain must therefore be followed by an observability/learnability test.

### Probability semantics and gradients

The proposed p and J definitions are legitimate finite-horizon event probabilities conditional on h,a, a specified continuation policy mu and the environment distribution. Separate masked BCE heads can consistently estimate each event's conditional mean with representative labels, sufficient model capacity and successful optimization. They are not a normalized trajectory distribution; no independence assumption is needed for those separate marginal estimates.

The bounds `0 <= J_kl <= min(p_k,p_l)` are necessary but not sufficient for joint coherence. If A and B each occur exactly once at distinct times with probability one, then `J_AB + J_BA = 1`; setting both to 1 passes the individual upper bounds while violating that constraint. Repeated events change the identity, so do not impose it without the single-occurrence assumption. Diagonal J_kk refers to repeated occurrence at two distinct times, not ordinary p_k. The draft's proposed projection is a bounded-output variant, not a general construction of a coherent trajectory law.

Changing the continuation policy, horizon, event definition or downstream candidate generator changes the target probability. Under a new closed-loop policy, an old J_mu is not automatically J_new, and it is never an optimal reachability guarantee. First freeze continuation policy and action candidates; if deployed receding-horizon behavior differs, evaluate/retrain for that shift or explicitly retain the feature-only interpretation. Instructions supplied to a new event-query head are additional inputs relative to NavQ's task-agnostic Q target; match them in p-only, future-QA and sequence controls as the draft specifies. All pair labels must come from the **same continuation**, not independently sampled occurrences spliced across branches.

**Decision:** revise the example and NavQ wording, then retain only as speculative. The representation argument is correct under its stated feature-map assumptions. Whether a compact J head beats direct future-QA, full short sequences, larger first-order features or a history-only policy is entirely untested. A theorem about symbolic aliasing cannot establish a novel or useful VLN method.

## P4: Failure-aware memory

### The inspected premise is real but its impact is unknown

PSG-Nav's paper describes positive/negative banks with ground-truth episode feedback (§3.4, p. 6). In the inspected release, unsuccessful episodes with a nonempty sliding window cause **every** stored recent detection to be passed to `add_false_positive`, gated by collection being enabled. The code reads the simulator success metric. That confirms an outcome-to-negative-write path, not that the published experiments used exactly this snapshot or that harmful writes are common. [PSG-Nav commit 181d878…, PSG_Nav.py:1483–1555](https://github.com/Chenrf1121/psg-nav/blob/181d878ff207845a31e542a19a6592925c1e9bab/PSG_Nav.py#L1483).

The paper's Table 4 is especially relevant: static memory already gives HM3D SR 66.0 versus dynamic PSG-Nav 66.1; MP3D 44.5 versus 44.8; HSSD 67.7 versus 67.9, with HSSD SPL 33.7 versus 33.4. These reported numbers do not prove statistical equivalence, but they make memory-frozen and quarantine controls essential. Do not infer large attributable gains from correcting a logical weakness when the paper's dynamic-update increment is small on those reported settings.

Under the restricted model S = identity_correct AND arrival_correct, S=0 is compatible with identity_correct=1 and arrival_correct=0. The executed truth table independently verifies this. Real benchmarks can have more success conditions, which only increases the need to define the claim and feedback contract. The native category goal must not be changed into a particular-instance goal without separate labels/protocol.

### Identifiability and objective audit

The three masked BCE heads are well defined for separately labelled binary events; they do not require mutually exclusive or independent causes. But naming heads does not identify causes. If two equally frequent hidden causes produce identical E and S=0, their binary attribution cannot beat 50% error using E alone. The finite check represents this case. Extra learned reasoning over the same observation cannot create missing distinguishing information. A reinspection may help only if it supplies informative evidence and its sensing cost is counted.

As with P2, masking labels estimates `P(y_k=1 | E, m_k=1)` without a missing-at-random/support assumption; it need not estimate the target population conditional probability. Report which failure types have reliable labels, use an independently sampled audit and test unresolved cases separately. Controlled simulator interventions can create labelled training cases, but their distribution may not match natural failures. A balanced intervention set cannot establish natural poisoning prevalence.

The proposed identity-invariance term is logically valid only when the intervention really preserves the identity evidence and label. A changed controller may reveal a new view, change occlusion, move an object, or change track association. Such pairs are not controller-only counterfactuals. For valid pairs the squared difference encourages equal identity predictions; it neither proves causal identification nor prevents the trivial constant prediction. The labelled BCE term and informative inputs must supply discrimination.

### The write gate needs joint validation

`qD >= tau AND W=1` is a measurable decision rule, not a product of independent evidence. The draft correctly avoids claiming independent observations. An additional subtlety is that even a *marginally calibrated* qD can lose calibration after conditioning on W. In the executed finite example, qD=.9 for 100 observations and exactly 90 have incorrect identity, so qD is globally calibrated; W selects the remaining 10 cases, giving 0% identity-error precision for the gate. This does not claim that such a distribution occurs in PSG-Nav. It shows why gate precision/coverage must be evaluated jointly and by scene/failure type.

If qD were the exact conditional probability given **all** evidence including W, then the selected group's expected precision would be at least tau by the tower property. That is a strong condition, not established by a global calibration plot or threshold tuning on a small set. Evaluate predicted-witness, witness-only, oracle-witness, quarantine and no-write variants separately. Optimize overall repeated-task utility with fixed storage/exposure, since reducing false writes can retain stale harmful beliefs.

**Decision:** revise identification, missing-label and gate-validation details. Start with shadow write/retrieval audits; advance only if erroneous writes cause later harm and simple quarantine does not already remove it. No attribution model, natural failure audit or repeated-navigation benefit has been validated here.

## Additional primary-evidence audits

These checks strengthen the evidence record. None supplies empirical support for the new proposals, and none warrants an allegation about researcher intent.

### UAGM Table 11: five-run paired-t reporting

The caption says mean±SD and paired t-test p-values over five runs. For paired observations X,Y, the ordinary statistic is `t = sqrt(n) (mean(Y)-mean(X))/s_D`, with n−1 degrees of freedom and D=Y−X. The [NIST paired-observation reference](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) specifies that test. For sample standard deviations,

\[
s_D^2=s_X^2+s_Y^2-2r s_Xs_Y,\qquad |r|\le1,
\]

so `|s_X-s_Y| <= s_D <= s_X+s_Y`. This is Cauchy–Schwarz applied to the two centered sample vectors and holds for every possible pairing. Allowing ±0.005 rounding in both displayed SDs lowers the minimum difference SD by at most .01; rounding both means increases the mean gap by at most .01. These relaxed bounds are deliberately favorable to obtaining a tiny p-value.

| Baseline / metric | Printed p | Smallest possible two-sided p under relaxed rounding bounds |
|---|---:|---:|
| DUET SR | 3.42e−12 | 4.13e−9 |
| BEVBert SR | 8.91e−11 | 8.83e−7 |
| VER SR | 1.58e−10 | 4.16e−8 |
| DUET SPL | 6.60e−15 | 1.16e−8 |
| BEVBert SPL | 7.30e−10 | 2.05e−8 |
| VER SPL | 2.90e−13 | 4.92e−8 |

All printed values are below even the corresponding **one-sided** lower bound, which is half the last column. Thus those means, SDs and p-values cannot all describe an ordinary paired t-test on the same five run-level results under the caption's straightforward interpretation. If the printed SD uses the population denominator n rather than n−1, conversion to sample SD only strengthens this incompatibility. Confidence intervals were not used to infer a confidence level, because the table does not specify it.

This does **not** show that the mean gains are absent. Even the largest possible paired-test p-values under these bounds remain below .05. It also does not reveal raw runs, pairing, analysis code, or intent. Tests over episodes or another larger statistical unit might yield smaller values, but that would be a different protocol from a five-run paired test and requires explicit reporting and appropriate dependence handling. Request raw per-run outcomes and the exact testing unit/code before using the printed extreme significance values as evidence. No misconduct inference is made.

### FeedTTA: scaling and magnitude

Let p be the coordinate-reversion probability, alpha<0 its multiplier, and D=1−p+alpha*p>0. The literal Eq. 5 has multiplier alpha on selected coordinates and 1/D on the rest. Its expectation is

\[
E[M]=\alpha p+(1-p)/D=1+\alpha p(D-1)/D>1
\]

for 0<p<1. Appendix B.1 instead divides **both** branches by D and has E[M]=1. These are different rules. For the literal equation,

\[
E[|M|]-1=-\alpha p(1+1/D)>0;
\]

for the appendix version, `E[|M|]-1 = -2*alpha*p/D > 0`. Thus neither negative-reversion version shrinks the expected absolute coordinate multiplier under these assumptions. The approximation in Eqs. 8–9 cannot establish that claim. The test includes settings satisfying the paper's stated |alpha|<p approximation regime: alpha=−.001,p=.01 gives exact absolute multiplier 1.0000201 for Eq. 5 versus approximate .99991. For alpha=−.2,p=.05, the main/appendix absolute multipliers are 1.0206383/1.0212766.

Expectation preservation of a signed random multiplier also implies `E|Mg| >= |E Mg|=|g|` by Jensen; nontrivial sign reversal makes the inequality strict for nonzero g. This is not a statement about learned weight norms, actual forgetting or empirical performance. Actual FeedTTA source was not available in the reviewed record, so which rule was run remains unknown. [FeedTTA Eqs. 5, 8–9 and Appendix B.1](../../Paper/FeedTTA.pdf).

### ATENA: sharper probabilities do not imply uniformly stronger gradients

The target is `q = lambda*onehot(selected) + (1-lambda)*pi`, as both paper and code confirm; it is not uniform mixing. Away from argmax ties, hold the chosen action fixed. For a binary policy with selected-action probability p=sigmoid(z)>1/2, the entropy gradients are

\[
\frac{dH(\pi)}{dz}=p(1-p)\log\frac{1-p}{p},\quad
\frac{dH(q)}{dz}=(1-\lambda)p(1-p)\log\frac{(1-\lambda)(1-p)}{\lambda+(1-\lambda)p}.
\]

Supplement A.7 includes the (1−lambda) factor in Eq. 13 but then compares only log-probability gaps. That comparison is insufficient for gradient magnitude. With p=.8 and lambda=.5, the mixture/original magnitude ratio is .79248; with p=.51 and lambda=.5 it is 14.0664. As lambda→1, the product of (1−lambda) and the logarithmic term tends to zero. Central differences agree with the derived gradients. Full-network directions/norms depend additionally on the probability Jacobian, so no uniform strengthening theorem follows. The paper's empirical improvement can still be valid; the correction narrows its theoretical explanation. [ATENA supplement A.7](../evidence/04/supplements/atena_5981_SUPPLE.pdf), [pinned mixture implementation](https://github.com/kuai-lab/NeurIPS25_att_vln/blob/f6903760c2dccdc9fb3524aa3b83f62ee9331e35/ATENA_DUET/map_nav_src/reverie/agent_obj_atena.py#L480).

### CPNAV: quantile direction and what coverage means

The rendered Eq. 12 uses `Beta^-1(delta; n+1-v, v) >= 1-alpha` with delta=.99, and nearby prose calls this 99% confidence; later text says confidence 1−delta. This is internally inconsistent. Under the ordinary continuous-iid order-statistic setting, conditional coverage C at rank k has a Beta(k,n+1−k) distribution across calibration samples. To ensure `P(C >= target) >= .99`, the **.01 lower quantile**, not merely the .99 upper quantile, must be at least the target.

The independent numerical counterexample uses n=500,v=225, consistent with Table I's calibration count: `Beta(276,225)` has .99 quantile .60218 and .01 quantile .49902. Eq. 12's upper-quantile condition passes target .55, but `P(C>=.55)=.51733`, not .99. This demonstrates insufficiency of the printed criterion even in a favorable iid continuous setting. No claim is made about an unreleased implementation's actual threshold.

Further, retaining an oracle path in a prediction set is distinct from executing it successfully within a finite budget. A higher-scored admissible bad branch can consume the budget while a correct branch remains unpruned. The finite check explicitly constructs that distinction. A repaired coverage result would still need precisely defined episode-level scores, unsuccessful calibration cases, exchangeability, action proposal coverage, search/termination behavior and finite sensing/motion limits. The paper alternates observation/action and episode descriptions in calibration; unavailable code prevents resolving its exact unit. Do not import a broad navigation-safety/success guarantee into any proposal from this source. [CPNAV §§III.D–F and Eqs. 12–15](../followups/sources/cpnav.pdf).

## Revision priorities and decisive tests

1. **P1:** First decide the semantic versus efficient-action annotation contract and shared termination/action interface. Then measure natural teacher disagreements with independently validated obligations. Do not train if the supposed semantic errors are mostly harmless alternative routes or ambiguous annotation.
2. **P2:** Validate the witness definition and sample unresolved cases independently before using predicted validity for selection. Stop if an equal-cost witness-free validity scorer or simple filtering baseline gives the same selection quality.
3. **P4:** Measure harmful negative writes and subsequent suppression before training an attribution model. Stop if freeze/quarantine suffices or the causes are unobservable from deployable evidence.
4. **P3:** Run a representation/readout study with the corrected matched-marginal cases and natural data, then test predictability. Drop the method claim if added task information, labels, capacity or an ordinary future-QA/sequence control explains the result.

No empirical navigation benefit, universal novelty, calibrated deployment risk or theoretical convergence guarantee is established for any proposal. The final package should preserve that status even after incorporating these mathematical corrections.
