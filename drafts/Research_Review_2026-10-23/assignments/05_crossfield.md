# Assignment 5 — Cross-field precedents and equivalence audit

2026-10-23 · Bounded primary-source review · No navigation training or empirical performance test

**Finding:** the original is a coherent outcome-supervised action-value baseline. Its generic architecture and likely value/ranking extensions are substantially anticipated. A squared all-pairs utility-gap loss is exactly centered regression, and its weighted version is graph-Laplacian regression. The remaining opportunity is an empirical, instruction-specific finding about which predicted consequences improve CE choices beyond direct value learning—not a newly named pairwise loss. No defensible set of two or three novel CVPR contributions is established by this audit.

## Scope and independence

**[User specifications]** Preserve the three policy backbones, four world-model trial families, CE-only added training and 8 GB target. Original materials are read-only. This assignment reviews cross-field mechanisms without repeating Assignment 4's VLN catalog.

**[Inspected]** Read `CE_World_Model_Approach.md` and `Scorer_Design.md` in full. Saved [independent candidates](05_independent_candidates.md) before reading other workers' proposals. Subsequently read Assignment 2's independent candidates and relevant mathematical sections, and Assignment 4's direct-novelty audit. Primary sources below were fetched independently; another worker's verdict is not treated as evidence. The report distinguishes published findings, deductions and hypotheses. No code from an external paper was executed.

## 1. Exact reductions of the original

Let `x=(I,h)` include the instruction and full real history; `a` is a primitive command; `Zhat=F_phi(x,a)` is the predicted next feature. The original learns

`q_theta(x,a) = S_theta(x,a,F_phi(x,a)) ≈ Q_ref(x,a) = E[U | x, force a, then pi_ref]`.

**[Deduction]** With frozen dynamics this is supervised Monte Carlo action-value regression with a structured intermediate representation. It is not new model-based policy optimization merely because an internal input is named a future. For deterministic `F`, its output is a function of already available history/action. It can improve finite-data representation, transfer and optimization; it cannot add observation information unavailable to an unrestricted direct `q(x,a)`. The direct value model with matched labels and capacity is therefore a primary competitor.

The complete-path utility remains a valid history-dependent return. Treat the history/prefix metric state as state, put the route-fidelity/success term at termination and the remaining action/collision charges on transitions. This gives a conventional finite-horizon value problem. Eight memory slots and frozen VLM features need not be sufficient statistics, so this formal reduction supplies no approximation guarantee.

**[Important correction after cross-reading]** Different continuation and deployment policies are not intrinsically a logical contradiction. Exact `Q_ref` on every visited sufficient history supports repeated one-step policy improvement. The practical difficulty is estimation on reranker-visited histories. My independent candidate 5 originally suggested refreshing the continuation policy; the simpler first response is Assignment 2's occupancy refresh while retaining `pi_ref` labels. Changing the continuation is a separately versioned approximate-policy-iteration experiment, not a required fix.

## 2. Decision-changing primary precedents

Each row states what was inspected and what it does—and does not—reduce. Recent preprints are prior disclosures; their reported empirical gains are not independently reproduced evidence.

| Primary source and verified status | Inspected mechanism and precise relation |
| --- | --- |
| **Ross & Bagnell, 2014: AggreVaTe / NRPI.** [arXiv record](https://arxiv.org/abs/1406.5979); 23 June 2014 preprint, no separate venue verified here. [PDF](https://arxiv.org/pdf/1406.5979), §2 Algorithm 1; §3 Algorithm 2 and Theorem 3.1. | AggreVaTe explores an action then uses expert cost-to-go, with aggregated cost-sensitive learning. NRPI instead rolls out the current learner after the exploration action. The original CE branch collector is a one-iteration relative of rollout-based policy improvement, with a frozen reference continuation and full enumeration of four actions. It is not literally AggreVaTe: no expert continuation, no iterative no-regret learner, and no inherited theorem under arbitrary scene sampling/compressed memory. Occupancy refresh and optional continuation refresh have established antecedents. |
| **VAML, AISTATS 2017.** [Proceedings](https://proceedings.mlr.press/v54/farahmand17a.html); [PDF](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf), §3 Eq.7, §4. | Measures model error through next-state expectations of value functions, with a supremum over a chosen function class. Therefore “predict what matters for downstream value rather than every state detail” is established. Original frozen-WM scorer training is not VAML because it does not optimize the dynamics using value. An extension that backpropagates value mismatch into dynamics is directly in this lineage; a single learned CE scorer is a much narrower function class than VAML's general formulation. |
| **IterVAML, NeurIPS 2018.** [Proceedings](https://papers.nips.cc/paper_files/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html); [PDF](https://papers.nips.cc/paper/2018/file/7a2347d96752880e3d58d72e9813cc14-Paper.pdf), Eq.11, Algorithm 1. | Fits transitions using the current value function rather than a supremum over all functions, alternating with value learning. A fixed actual-future evaluator `v` and loss `(v(F_phi(x,a))-v(z_real))²` instantiate the same basic value-consistency objective under deterministic one-step latent dynamics. Conditioning `v` on instruction/history/action changes the application and function family, not the generic objective. Measured terminal branch-return targets differ from a teacher's next-state value but remain standard outcome/value targets. |
| **Value Equivalence, NeurIPS 2020.** [Official PDF](https://proceedings.neurips.cc/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf), §3 Definition 1. | Equivalence is equality of Bellman updates over specified policy and value-function sets. Correct scores for four actions at sampled CE histories establish neither equality of all relevant Bellman operators nor a generally valid dynamics model. Correct ranking is weaker still: any strictly increasing score transformation preserves ranking but need not preserve values/Bellman updates. Avoid claiming a new value-equivalence theorem from empirical action-order agreement. |
| **VaGraM, ICLR 2022.** [Paper](https://arxiv.org/pdf/2204.01464), conference status printed on paper, §3.1–3.2 Eqs.5–7. | Identifies inaccurate value evaluations off the empirical state distribution and unstable value-aware optimization. Uses value gradients at real next states to weight squared model errors; Eq.7 is a diagonal quadratic feature/state metric. This directly anticipates “avoid scorer exploitation by weighting real prediction errors using task sensitivity.” CE outcome-gap weighting is not exactly VaGraM: it uses measured between-action differences rather than coordinatewise derivatives. That distinction needs a reason and evidence, not a novelty assertion. |
| **CVAML, ICML 2025.** [Proceedings](https://proceedings.mlr.press/v267/voelcker25a.html); [full text](https://arxiv.org/html/2505.22772v2), §3 Eqs.1–2; §4.2 Eq.3 and Propositions 2–3; §4.3. | Sample-based value losses can acquire a model-dependent variance penalty when optimizing stochastic models. CVAML corrects this; it also separates model-learning from value-learning bias. This is relevant if a stochastic CDiT predictor is optimized through sampled scorer outputs. It does not prove that the original frozen-WM Monte Carlo utility regression is invalid. Direct terminal-outcome regression has different optimization variables and is not the same bootstrapped MuZero loss. See §4 below for the precise conditional decomposition. |
| **Donti et al., NeurIPS 2017; Wilder et al., AAAI 2019.** [Donti PDF](https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf), §3.1–3.3, Eqs.3–7; [Wilder PDF](https://ojs.aaai.org/index.php/AAAI/article/download/3982/3860), “Problem description” and “Continuous relaxations”; [venue record](https://ojs.aaai.org/index.php/AAAI/article/view/3982). | Task-based / decision-focused learning optimizes predictions through the decisions they induce, including differentiable relaxations of discrete optimization. “Lower decision regret despite worse generic prediction error” is established motivation. CE argmax over four scores is a simple special decision problem; squared score gaps are a surrogate, not literal differentiation of closed-loop navigation return. A new softmax relaxation, hinge loss or learned weighting would need comparison with standard surrogates. |
| **Future-trajectory distillation, ICLR 2021.** [Author repository](https://github.com/neitzal/learning-to-distill-trajectories), README “Applying to new tasks”; [paper link](https://openreview.net/pdf?id=ECuvULjFQia). | A teacher processes privileged trajectories and supervises present-only student activations, with meta-learning machinery in the implementation. The paper was indexed as an ICLR 2021 paper, but direct OpenReview access hit a browser challenge; claims here are limited to the indexed abstract and author README. It establishes a relevant alternative: use future observations in training without generating them at deployment. It does not duplicate CE branch-value estimation. |
| **Privileged Foresight Distillation (PFD), 2026 preprint.** [Metadata](https://arxiv.org/abs/2604.25859): v1 28 April, v2 2 May; no venue verified. [v1 methods](https://arxiv.org/html/2604.25859v1), §3.2–3.4 Eqs.4–10; [v2 inspected](https://arxiv.org/html/2604.25859v2). | Defines teacher-minus-current-only action-denoising residual using privileged future video, then trains a small adapter; future tokens are absent at inference. This is a close prior to the broad “future-induced correction” framing. My candidate 1 predicts scalar utility residuals with candidate futures at deployment, so it is not identical. Nevertheless a future-residual slogan or small correction head alone is not novel. A present-only distilled value residual is a necessary cheaper competitor if this route is pursued. |
| **World4Scorer, 2026 preprint.** [Metadata](https://arxiv.org/abs/2609.36438): v1 29 September; no venue verified. [Full text](https://arxiv.org/html/2609.36438v1), §3.2–3.4; especially Appendix D.2. | D.2 freezes LeWM and adds an MLP over predicted final/goal features, their combinations, distance and action summaries. Simulator candidates supply negative minimum cube-goal-distance targets, standardized within groups; smooth-L1 plus listwise cross-entropy trains the scorer. This is a direct frozen-world-model outcome-scoring precedent outside driving. The main method also jointly supervises candidate predictor states with outcomes and observed-future anchoring. Remaining CE differences are instruction/real-history conditioning, exhaustive primitive choices and reference-policy terminal returns. Those are substantive task differences, not proof of a new general mechanism. |
| **MCR, 2026 preprint.** [Metadata](https://arxiv.org/abs/2608.09073): v1 10 August; no venue verified. [Full text](https://arxiv.org/html/2608.09073v1), §3.3.2 Eqs.6–7; §3.3.3. | Trains action-conditioned latent dynamics with a pairwise hinge cost-ranking objective plus rollout anchor. Ordering/margins come from dead-reckoned trajectory deviation under graded perturbations of demonstrated actions. Therefore “ranking-preserving dynamics plus native prediction anchor” is already explicit navigation prior. Measured CE outcome gaps are a different supervisory target and may disagree with geometric deviation—especially for turns, detours and instruction order. That disagreement is the testable remaining difference. The paper also reports a failed action-contrastive variant; arbitrary negatives should not be assumed useful. |
| **CEM proposal audit, 2026 preprint.** [Metadata](https://arxiv.org/abs/2610.00921): v1 1 October; no venue verified. [Full text](https://arxiv.org/html/2610.00921v1), §3–4, §6. | Separates scoring from candidate-pool generation by rescoring fixed pools, then intervenes on the first elite update. In CEM, score errors change later proposals as well as the current choice. The original exhaustive four-action set has no such within-decision adaptive proposal mechanism. Fixed candidates cleanly remove this confound, but that is an experimental property rather than a new method. If candidate chunks/CEM are later added, crossed pool/scorer controls become essential and are already disclosed. |
| **TRM, 2026 preprint.** [Metadata](https://arxiv.org/abs/2605.22164): v1 21 May, v2 31 August; no venue verified. [Full text](https://arxiv.org/html/2605.22164v2), method Eqs.5–6; “Horizon Matching and Temporal Structure Drive the Repair,” Table 4. | Learns a temporal pairwise terminal cost on fixed latent dynamics; whole-episode pair sampling can outperform short-range supervision in its navigation settings. Temporal separation is a proxy for reachability and has coverage limits. This anticipates repairing a frozen model's decision geometry and matching supervision scale to use. It does not prove that CE must predict many actions: a one-step future plus full-return continuation value already represents long-term outcomes. Longer prediction horizon is a hypothesis to activate after measuring inadequate one-step action headroom. |

## 3. Pairwise gap mathematics: exact equivalence and limits

**[Derivation, not a novelty claim]** At one root write true targets `y ∈ R^K`, predicted scores `s ∈ R^K`, errors `e=s-y`, and `P=I-11ᵀ/K`. For unweighted unordered pairs,

`L_pair = Σ_(i<j) [(s_i-s_j)-(y_i-y_j)]²`

`       = Σ_(i<j) (e_i-e_j)²`

`       = K Σ_i e_i² - (Σ_i e_i)² = K ||P e||²`.

Thus average pair loss is `2/(K-1) ||Pe||²`. If centered MSE means `(1/K)||Pe||²`, their ratio is `2K/(K-1)`; for `K=4`, it is `8/3`. Ordinary MSE decomposes as `(1/K)||Pe||² + mean(e)²`. Adding all-pair squared gap loss therefore reweights the centered component against the root-level mean error. It does not create a different ranking principle or extra labels. A learned per-root additive offset is unidentifiable from gaps alone, which is harmless for same-root argmax but matters for absolute calibration.

Let nonnegative symmetric edge weights be `w_ij`, graph incidence matrix `B`, and `L=Bᵀ diag(w) B`. Then

`L_weighted = eᵀ L e`.

This is exact graph-Laplacian regression. A complete graph with unequal edge weights generally is not ordinary scalar sample weighting. Edges involving only the base action form a star; positive weights connect all actions and determine differences up to one common offset. Disconnected graphs leave component offsets unconstrained. Zeroing near-tie weights needs a connectivity check before invoking global gap preservation. Rootwise positive rescaling of every target preserves argmax but alters across-root training emphasis and any fixed utility-unit override margin.

For the feature-contrast candidate, let rows of `E` be `Zhat_a-Zreal_a`. Then

`Σ_(i<j) w_ij ||(Zhat_i-Zhat_j)-(Zreal_i-Zreal_j)||² = tr(Eᵀ L E)`.

With uniform complete weights it is centered feature regression; a native prediction anchor also controls common-mode error. Outcome weights change the geometry/data emphasis, but “counterfactual contrast preservation” alone does not evade this exact reduction. Unlike VaGraM's within-feature-coordinate sensitivity, the Laplacian couples action-indexed prediction errors. This is a real mathematical difference; whether it matters empirically is unknown.

**Huber caveat:** Assignment 2 uses Huber gaps. The quadratic identity applies when the chosen Huber convention is in its quadratic region for every pair, up to its scale factor. Outside that region it is robust pairwise regression, not exactly centered MSE. It is still not automatically a new class of decision-focused learning. Hinge ordering, gap regression and listwise probability matching have different optima under noise; do not call them equivalent merely because all compare actions.

**Conditional action-regret bound.** Let `a*=argmax y` and `ahat=argmax s`. Then

`0 ≤ y_a* - y_ahat ≤ e_ahat - e_a* ≤ sqrt(2) ||Pe|| = sqrt(2 L_pair/K)`.

The first inequality uses score selection; the second is Cauchy–Schwarz on the two-action contrast. For average pair loss this becomes `regret ≤ sqrt((K-1) L_pair_avg)`. On a connected weighted graph,

`regret ≤ sqrt(R_(a*,ahat) L_weighted)`,

where `R_ij=(unit_i-unit_j)ᵀ L⁺(unit_i-unit_j)` is effective resistance; a looser bound is `sqrt(2 L_weighted/lambda_2(L))`. These are per-root algebraic bounds on the same target, not generalization or navigation guarantees. Empirical fit to single sampled branch returns does not bound error against expected `Q_ref`. A calibrated bound on every selected-versus-base gap would justify an override threshold; an observed development margin is not such a bound.

## 4. Stochastic predictions and scorer exploitation

**[Deduction with stated assumptions]** Fix a root/action and a frozen scorer `f`. Let `X=f(Zhat)` for a model sample and let `Y` be an independent measured branch return. If optimizing dynamics,

`E[(X-Y)²] = (E[X]-E[Y])² + Var(X) + Var(Y)`.

`Var(Y)` is constant to model parameters; `Var(X)` is not. Independent single-sample loss thus also suppresses the model's score variance. For average of `m` independent model samples, the extra term is `Var(X)/m`. For a pairwise gap use `X=X_a-X_b`; its variance includes `-2 Cov(X_a,X_b)`, so whether candidates share randomness changes the training penalty. This is the appropriate CVAML connection. If `f` is instead being learned with frozen model samples, its dependence on `X` also changes; one cannot transplant the model-only conclusion unchanged.

A frozen one-sample predictor is not necessarily an estimator of expected action utility merely because it is a valid generative model. Noise is independent of the realized hidden environment conditional on the deployed inputs; the population-optimal direct outcome regressor can ignore noise that carries no conditional information. Finite networks may nevertheless fluctuate across sampled futures. Test mean-score aggregation and decision stability at matched runtime before giving samples an uncertainty interpretation.

Freezing the evaluator does not prevent exploitation. A model can map to off-distribution features that receive the desired scalar/gap while being visually false. Native feature/diffusion anchoring limits this tradeoff but does not certify truth. In a local approximation `f(z+delta)-f(z) ≈ gᵀdelta`, scalar matching leaves directions orthogonal to `g` unconstrained. VaGraM's real-state sensitivity and anchored feature alternatives are necessary baselines for this exact issue.

**Brief assessment of coordinator's later projected-gradient idea.** This idea was received after the independent record. For a fixed orthogonal projector `P_c` onto observed action contrasts, `P_c ∇f(z)` need not be a scalar gradient field: its Jacobian is `P_c H_f`, generally nonsymmetric unless `P_c H_f=H_f P_c`. The scalar `f(z_real+P_c(z-z_real))` is valid and differentiable, with gradient `P_c ∇f` at the projected input, but its loss hides orthogonal feature errors. A native anchor can expose them. Nothing here establishes novelty beyond task-weighted local metrics, nor that discarding orthogonal directions preserves value outside the measured local contrasts. Do not add this mechanism without a demonstrated failure of simpler alternatives.

## 5. Independent candidates after comparison

| Independently saved candidate | Relation after literature and cross-reading | Disposition |
| --- | --- | --- |
| 1. Future-conditioned residual beyond history/action-only value | Shares Assignment 2's concern about advantages; PFD already gives a privileged-future correction framing. `b(x,a)+r(x,a,Zhat)` has no extra expressivity over a sufficiently capable full scorer; separate/cross-fit fitting changes optimization. | Best scientific control, not established contribution. Include present-only distilled residual and equal-capacity direct value competitors. |
| 2. Preserve within-root value contrasts when training dynamics | Strongly overlaps Assignment 2 A. VAML/IterVAML give value consistency; MCR gives anchored ranking; World4Scorer gives outcome-supervised predictive states. Squared version reduces exactly as above. | Plausible simple experimental intervention, with novelty currently unproven. CE measured-return semantics is the possible differentiator to test. |
| 3. Paired continuations and adaptive branch budget | Classical variance reduction: `Var(U_a-U_b)=Var(U_a)+Var(U_b)-2Cov(U_a,U_b)`. Common randomness helps only with positive covariance and valid synchronization. | Collection efficiency/control, not an independent contribution. Preserve reset state, task budget, policy cache, and metric prefix. |
| 4. Predict until a discriminating consequence | Related to MPC, options, TRM scale matching and CEM proposal feedback. One-step prediction does not imply one-step value. | Defer until measured primitive headroom or actual-next-view diagnostic fails. A horizon expansion needs matched execution/labels/runtime. |
| 5. Refresh continuation to reranker | NRPI/approximate policy iteration precedent; Assignment 2 shows frozen-reference labels can suffice if accurate on visited histories. | Revised: first refresh root occupancy, keep target fixed. Continuation change is an optional separate pivot, not a correctness requirement. |
| Assignment 2 B, read later: outcome-weighted feature contrast | Gradient-free variant avoids direct optimization of arbitrary scorer outputs. Exact feature Laplacian reduction above; distinct from VaGraM coordinate weights, but no standalone novelty proof. | Useful competing objective against gap-through-scorer training. It may be the simpler choice if off-support scorer exploitation is observed. |

## 6. Best bounded research intervention and honest gap

**[Recommendation / hypothesis]** First qualify the predictive channel, then compare two simple dynamics losses inside one agreed world-model family. Keep the other families and backbones as recorded staged comparisons; do not remove them. The central mechanism candidate is measured instruction-conditioned utility-gap preservation with a native prediction anchor. Describe it plainly as an application/extension of value-aware learning. Its rival is outcome-weighted feature-contrast regression, plus an ordinary per-action outcome-weighted native loss. Do not count these alternatives as multiple contributions.

The decisive evidence sequence is:

1. On preserved roots, establish measured alternative-action headroom and the distribution of meaningful gaps. Repeated continuations are needed only if policy/environment stochasticity makes uncertainty material.
2. Establish an actual-next-view advantage over the matched direct value model; then determine how much is retained with predicted views. Treat actual-future inputs as a capacity/representation diagnostic, not a deployable result or a bound guaranteed to dominate all learned predictors.
3. At matched data, training budget and scorer weights, compare native prediction, per-action utility/value matching, centered regression, Huber/hinge gaps and weighted feature prediction. Squared pairwise and centered formulations should agree after accounting for normalization; any discrepancy is implementation/optimization, not a new objective.
4. Demonstrate a CE-specific reversal: geometric action perturbation/deviation ordering versus measured instruction-following utility ordering. Predefine legal cases involving turn timing, detours, route order and STOP; recompute valid labels. Quantify how often this happens on ordinary held-out roots, not only hand-selected examples.
5. If the proposed objective corrects those reversals and yields held-out closed-loop gains over direct value and training-only future distillation at the actual latency/memory budget, the potential contribution is a measured semantic failure mode plus its simple remedy. Refresh scorer-fitting root coverage if closed-loop deployment changes occupancy.

**[Unmet target]** None of steps 1–5 has been executed on CE here. The present evidence does not establish the pivotal semantic reversal prevalence, a future-input advantage, the advantage of a new loss over existing surrogates, or an 8 GB full-pipeline fit. Accordingly the defensible output is a conditional research direction with a clear kill criterion, not a promise of a CVPR paper. If no-future value or training-only distillation matches prediction-assisted inference, report that result and revise the central hypothesis.

## 7. Search/access ledger and reproducibility

Cutoff: 2026-10-23 Asia/Seoul. Source status was taken from primary arXiv/venue pages, not search-engine relative timestamps. No post-cutoff version was intentionally used. Recent preprint dates/status are listed in §2. Primary arXiv HTML snapshots and parsed text for the six recent/detailed sources are saved under `evidence/a5_*.html` and `evidence/a5_*.txt`; `a5_fetch_primary.py` records URLs and parses those files. They are source evidence, not model outputs or implementations.

Queries executed (quotation marks denote actual query text, not source excerpts):

- `AggreVaTe No-Regret Learning in Sequential Decision Making cost to go Ross Bagnell 2014`
- `Value-Aware Model Learning Iterative Value-Aware Model Learning VaGraM CVAML Voelcker 2025`
- `decision focused learning Donti Amos Kolter 2017 Wilder Dilkina Tambe 2019`
- `privileged future distillation visuomotor policy future observations teacher student world model`
- `"Value Gradient weighted Model-Based Reinforcement Learning" paper`
- `Grimm value equivalence principle model based reinforcement learning 2020`
- `"Learning by Cheating" "Future" ICLR 2021 ECuvULjFQia` (initial title guess; actual title resolved below)
- `"Iterative Value-Aware Model Learning"`
- `"ECuvULjFQia" title`
- `"Task-based End-to-end Model Learning in Stochastic Optimization" paper`
- `"Melding the Data-Decisions Pipeline" AAAI 2019`
- `"A teacher-student framework to distill future trajectories" arxiv`

Direct lead opens: arXiv `2609.36438`, `2608.09073`, `2610.00921`, `2605.22164`; followed to their versioned full texts. Search returned other privileged-future/robotics leads, including TWIST and FPD-Drive, but these were not fully inspected and do not support verdicts in this report. No absence-of-search-hit argument is used.

Access issues: incorrect guessed IterVAML PMLR path failed; resolved through the actual NeurIPS 2018 proceedings. CVAML's raw GitHub PDF returned unsupported content type; full arXiv v2 HTML and PMLR metadata were inspected instead. OpenReview ICLR 2021 page/PDF returned a browser challenge; MPI publication page returned 403, so evidence is limited as stated. Python HTTPS retrieval failed the local certificate chain; PowerShell `Invoke-WebRequest` fetched the same public arXiv HTML successfully without disabling certificate verification. The parser was then run on local snapshots. PFD v1 methods were inspected fully at relevant sections and v2 was opened; no v1-only novelty claim is made.

Verification limits: primary-paper mechanisms were inspected, but reported experimental results were not reproduced, current code releases/checkpoints were not audited, no source edits or training were performed, and no statistical novelty guarantee is possible. Algebra in §3–4 is an independent deduction with assumptions shown. Assignment 4 supplies the complementary direct-VLN audit; this report does not certify its entire source set.
