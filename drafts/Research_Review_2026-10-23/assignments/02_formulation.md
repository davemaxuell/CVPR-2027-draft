# Assignment 2 — Constructive mechanism and mathematical formulation

Date/cutoff: 2026-10-23. This is an independent reconstruction of the original design, followed by explicitly labeled extensions. No other assignment result was read before this file was drafted. Independent creative suggestions were recorded first in [02_independent_candidates.md](02_independent_candidates.md). Original materials were not modified. AI-assisted analysis used native tools and primary-source browsing, without external model APIs or large training.

**Assessment:** the original is a coherent approximate policy-improvement system, but neither its predictive benefit nor its novelty is established. Its strongest interpretation is an instruction-conditioned estimate of the frozen policy's action value, using a learned future as an intermediate representation. A one-action prediction does not make this a one-step reward optimizer. Exact full-history values would support repeated policy improvement; the practical difficulties are representation, estimation, stochastic prediction, and coverage of the new policy's histories. The original openly leaves these empirical questions unresolved.

## 1. Evidence and scope

Labels used below: **[Specification]** = original recorded design; **[Code]** = inspected reference implementation; **[Deduction]** = derivation here; **[Assumption]** = condition needed by a proposition; **[Hypothesis]** = untested mechanism or proposed intervention; **[Published]** = independently inspected primary source; **[Executed]** = bounded numerical check in this assignment.

Read in full: `Research Approach/CE_World_Model_Approach.md`, `Scorer_Design.md`, `Scorer/scorer.py`, `labels.py`, `config.json`. These paths are relative to `261022/`. Code locators below use class/function names, which are more stable than line numbers; the review inventory records original hashes. Prior investigation dossiers were not used as evidence.

**[Specification]** Preserve the three frozen-policy trials (SmolVLM2-500M, OneVision-0.5B, Qwen3-VL-2B) and four world-model trials (transferred DINO-WM, transferred NWM/CDiT, fresh DINO dynamics, fresh CDiT). Staging the study around one policy does not erase the other recorded trials. Transfer claims require architecture-matched pairs. All added fitting data are CE training data; external pretraining remains acknowledged. No complete 8 GB fit is established by scorer parameter counts.

## 2. Exact information flow

Let the actual history be `h_t=(o_0,a_0,...,a_(t-1),o_t)`, instruction `I`, elapsed action count `t`, and full episode budget `B`. Hidden simulator/task state exists for transition generation and labels but is excluded from deployed inputs. The frozen navigation policy `pi_ref` uses one-action replanning. Its actual cache must be reconstructible from the forced executed actions and real observations. A stale cache containing an unexecuted action chunk defines a different policy and invalidates the intended labels.

**[Specification/Code]** Cache instruction-only final hidden states `T∈R^(L×D)` once per episode. Obtain `q_t∈R^D` at the last input token of the actual multimodal prompt, before generating candidates. The configured policy widths are 960, 896, and 2048, respectively. These are feature contracts pending backbone-hook verification, not executed integration evidence.

Let `E` be the fixed DINOv2-S/14 representation and `P` its 2×2 spatial average pooling. Current features are `Z_t=P E(o_t)∈R^(64×384)`, pooled from 256 patches. Project text/context and visual features to width 256 with separate layer-normalization/linear maps. The current source sequence has `L+66` tokens: text, 64 current visual tokens, one policy-context token, and one previous-action/time token. Text padding is masked.

The real memory is `M_t∈R^(8×256)`, initially zero. Its update is

`C_t = concat(project(T), project(Z_t), project(q_t), embed(a_(t-1))+project(t/B))`,

`R_t = Attention(LN(M_(t-1)+slot_queries), C_t, C_t)`,

`M_t = shared_GRU(R_t+FF(R_t), M_(t-1))`.

Role and fixed 2D visual position embeddings are included. Eight distinct learned slot queries break slot symmetry; recurrent updates can encode order, but are not guaranteed sufficient statistics. `[C_t,M_t]` contains `L+74` tokens. Only `observe()` updates this persistent memory, exactly once per real observation. During training, replay prefixes with current scorer parameters; retain the final 32 steps' graph and detach earlier memory. Caching old trainable memory states would change the optimization problem.

For actions `A={FORWARD,LEFT,RIGHT,STOP}`, generate

`Zhat_a = P Adapt(F_phi(h_recent,a;xi_a))` for movement actions, and `Zhat_STOP=Z_t`.

For compatible DINO dynamics the adapter is the compatible feature interface; for CDiT it is frozen VAE decoding to RGB followed by the same frozen DINO encoder/preprocessing. The model input must use commanded action information available before execution, never realized displacement or collision labels. The original uses one sample `xi_a` per movement action. It has no extra imagined rollout beyond that action.

Each candidate independently forms 65 width-256 tokens (action-summary plus 64 future tokens), passes two self-attention/cross-attention/feedforward blocks, then a 256→128 readout and three scalar heads. The utility score is

`qhat_a = S_theta(T,Z_t,q_t,M_t,a,Zhat_a)`.

Candidate order cannot affect an individual score in the reference implementation. The base action is always present because all four primitives are scored. `score()` substitutes current features for STOP regardless of a supplied placeholder. STOP terminates; it is not a physical transition and needs no model invocation.

**[Code]** `observe()` detaches frozen text, current view, q_t and elapsed inputs; `score()` detaches predicted features. Gradients train the scorer's projections, attention, recurrent memory, and heads. There is no gradient to any world model in the original code. Scorer parameters total 2,890,115 / 2,873,603 / 3,170,819 for the three configured widths. `select_action()` uses strict estimated gain `>delta`, retains the base action on a base tie, and otherwise resolves a top tie by canonical action ID. It raises on nonfinite scores; the deployment fallback remains unimplemented.

**[Deduction]** Full instruction tokens plus q_t permit instruction grounding but do not force it. A solution can ignore future features, reconstruct action preferences from q_t, or exploit scene regularities. Attention weights are not evidence that the instruction, memory, or predicted consequence caused a decision. The no-future, instruction-regenerated, and history-controlled tests in the original are therefore structurally necessary.

## 3. What Q_ref means, including the whole path

At a fixed root, force candidate `a`, then follow the same frozen `pi_ref` on real observations to STOP or the remaining original budget. Define

`U_t = .70 S + .30 N - .01 n_t/B - .02 c_t/B`,

`Q_ref(I,h_t,a) = E[U_t | I,h_t, forced a, then pi_ref]`.

`S` is configured terminal success; `N` is official nDTW of the entire path, including its prefix; `n_t,c_t` count actions and collisions after the branch root. The candidate and an executed STOP are counted. Restore metric prefixes, random states, policy caches and remaining budget identically across branches. A common random-number protocol can reduce comparison noise, but one branch is still a realized outcome, not an expectation in a stochastic environment.

**[Deduction]** Define a global episode return

`J = .70 S + .30 N - .01 n_total/B - .02 c_total/B`.

At a root, `U_t = J + k(h_t)`, where `k(h_t)=.01 t/B+.02 c_prefix/B` is constant across that root's candidates. Therefore Q differences computed from original remaining-cost labels equal Q differences for a single globally consistent objective. Path-dependent terminal nDTW is compatible with history-based dynamic programming; it need not be a Markov reward of the current image. One can assign per-step action/collision costs and a terminal reward `.70 S+.30 N` in the finite-horizon history process. The reference trajectory is part of the hidden task/metric definition, not a permitted scorer input.

The bounded labels satisfy `-.03≤U_t≤1`, with realized success at least .67 and realized failure at most .30. This does **not** prioritize success probability lexicographically: ignoring equal costs, expected `(S,N)=(.8,.1)` gives .59 while `(.7,.8)` gives .73. These numbers demonstrate the utility's algebraic tradeoff; they are not claimed observed CE trajectories. Any claim of improved SR must be tested separately.

For deterministic simulator and deterministic continuation, one exact saved-state branch determines that continuation's return. Across observationally identical histories with different hidden states/reference tasks, an observation-based Q remains a conditional expectation. The evaluator's hidden information is not acquired by the model merely because labels used it.

## 4. What the original loss learns

**[Specification/Code]** Keep policy, visual encoders, VAE and dynamics frozen. Train scorer with

`L = mean_(roots,actions)(qhat-U)^2 + .1 BCE(success_logit,S) + .1 mean_FORWARD BCE(collision_logit,collision)`.

The collision average is over FORWARD samples only, not divided by all four actions. The code shares a representation across all three heads, so auxiliary tasks can influence utility ranking even though their logits do not enter selection. The initial optimizer is AdamW, 1e-4 learning rate, .01 decay, eight roots × four actions, gradient norm clipping at 1; none is CE-validated.

**[Deduction: population statement]** For any fixed input `X`, minimizing unrestricted expected squared error gives `s*(X)=E[U|X]`, by

`E[(s-U)^2|X]=(s-E[U|X])^2+Var(U|X)`.

Here `X` includes compressed real context, action, and the chosen actual/predicted future representation. It is not the full history. A finite shared network, truncated recurrent optimization, incomplete coverage and auxiliary losses need not reach this conditional mean. Successful MSE fitting is also not a guarantee of useful action margins.

There is a material distinction between actual and synthetic future labels. With actual next observation `Y`, the scorer can learn `E[U|X_real,a,Y]`. With a model sample `Zhat` independently drawn from model noise given the full actual history, it learns `E[U|X_real,a,Zhat]`, which is generally a different function. In the ideal full-history case, independent model noise tells nothing about the realized branch outcome, so that latter conditional mean is `Q_ref(h,a)`, constant across sample seeds. Actual→mixed→predicted training is a sensible proposed curriculum, not a proof that the two conditionals coincide.

Pooling four models without identity estimates a mixture-domain conditional mean. Holding this scorer fixed is a useful controlled comparison, but does not establish equal calibration within every component model. Finishing on predicted inputs addresses the deployment modality; it cannot remove an unmeasured domain bias. Report per-model residuals and rankings rather than only pooled loss.

## 5. Conditional mathematical guarantees and limits

### Proposition 1 — Exact values support repeated improvement

**[Assumptions]** Finite horizon B; same initial task distribution, sensing and one-action schedule; Q_ref defined on full histories for the same global return above; policy/cache execution consistent; candidate set includes the reference action. For a possibly randomized reranker `mu`, assume at every history reachable under mu,

`E_(a~mu(.|h))[Q_ref(h,a)] >= V_ref(h)`,

where `V_ref(h)=E_(a~pi_ref(.|h)) Q_ref(h,a)`.

**[Deduction]** Then `E[J(mu)]>=E[J(pi_ref)]`. Let `A_ref(h,a)=Q_ref(h,a)-V_ref(h)`; the root-dependent prefix constants cancel in A. Repeated conditioning and telescoping of `r_t+V_ref(h_(t+1))-V_ref(h_t)` gives

`E[J(mu)]-E[J(pi_ref)] = sum_(t=0)^(B-1) E_(h_t,a_t~mu)[A_ref(h_t,a_t)]`,

padding terminated episodes with zero-cost absorbing steps. Every term is nonnegative under the premise. Strict improvement needs positive advantage with positive mu occupancy, not merely a better action at an unreachable history.

For a deterministic reference, exact greedy selection or exact positive-margin overrides satisfy the premise. If the reference is stochastic, compare to its expected V_ref, or couple a sampled base action to an override which never reduces its exact Q and average over that base sample. Comparing against an arbitrary mode of a different stochastic continuation needs separate justification.

This is an adaptation of standard policy improvement to the design's history/terminal-path return, not a new theorem. [Sutton and Barto, §4.2, Eqs. 4.6–4.9](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) state the underlying deterministic-policy result. The proof above makes the path-dependent application explicit. **A mismatch between pi_ref continuation and repeated reranking alone does not refute the method.** What fails in practice is the unproven premise that estimated advantages remain correct on the reranker's histories.

### Proposition 2 — Score error, margins, and regret

**[Assumption]** At one visited history, for all four actions, `|qhat_a-Q_ref(h,a)|≤epsilon`. This must cover model-sampling variation if selection uses one sample.

**[Deduction]** If an override from b to a has estimated gain `g=qhat_a-qhat_b>delta`, then

`Q_ref(h,a)-Q_ref(h,b) >= g-2epsilon > delta-2epsilon`.

Thus `delta>=2epsilon` prevents negative exact advantage at that history under a deterministic base; a development-tuned delta has no such guarantee without an applicable error bound. Unconstrained argmax has root regret at most `2epsilon`. The margin rule has root regret at most `delta+2epsilon`: if it retains b, every predicted gain over b is at most delta; if it overrides, use the ordinary argmax bound.

For deterministic pi_ref, gain is zero when retaining its action, and at least `delta-2epsilon` at each override. Summing along mu occupancy yields the corresponding finite-horizon lower bound. A small average MSE on training roots is not a uniform epsilon. More generally, if negative advantage is bounded in expectation by eta_t under **mu's** t-step history distribution, the total loss is bounded by `sum eta_t`. A density-ratio coverage assumption can transfer training-distribution error bounds; no such ratio is established by the original data split.

### Counterexample 1 — Accurate root ranking is insufficient under new occupancy

At root h0, the base chooses a path worth .6; an alternative leads to h1 where following pi_ref is worth .8. Root values are exact, so reranking selects h1. At h1 the reference action is worth .8 and a bad action worth 0, but the off-support scorer assigns .1 and .9. Repeated reranking returns 0. Exact Q_ref at h1 would have prevented failure; changing continuation labels is not logically required to fix it. Collecting and labeling reranker-visited roots with the original frozen continuation is a minimal coverage intervention.

### Proposition 3 — Prediction adds a representation, not a new observation

**[Assumptions]** Condition on the fixed trained model parameters and complete inference information `(I,h)`. Model noise xi is independent of the current episode's hidden state/outcome given those inputs. `Zhat=f_phi(h,a,xi)`.

**[Deduction]** For branch outcome U, `U` and `Zhat` are conditionally independent given `(I,h,a)`; equivalently there is no extra current-episode information in a simulated sample once its full inputs are known. Any deterministic scorer composed with a deterministic world model is itself a function of `(I,h,a)` and could in principle be represented by a direct action-value model.

This does **not** make world models useless. With finite data/compute/function classes, dynamics pretraining, useful factorization, or richer world-model history than the compressed scorer pathway can make the composed estimator better. These are inductive-bias/representation hypotheses, not information gain from observing an unexecuted future. An actual simulator branch supplies privileged additional information and is therefore a diagnostic ceiling, not an achievable deployed input. A same-capacity no-future scorer tests the architecture claim; also report the predictive pipeline's extra compute, data and pretraining.

### Counterexample 2 — One perfect stochastic sample can induce a bad decision

Action A has utility .6 surely; B has utility 1 or 0 with equal probability, so Q(B)=.5. Suppose a perfect world model samples B correctly and a perfect **actual-future** evaluator returns the utility of the sampled outcome. The planner chooses B when its model sample is 1, otherwise A. The real environment draw is independent of the model draw: expected return is `.5×.5+.5×.6=.55`, below the .6 base. This refutes equating one sampled imagined state's value with expected action utility, not the fidelity of the generator.

The original predicted-input/actual-label training can learn to average away this noise in the population full-history limit. It is not guaranteed to do so after actual-view pretraining or with compressed context. Multiple samples should aggregate expected scores, not the best-looking sample; they reduce Monte Carlo variance but do not remove model bias or critic bias. This change has explicit runtime cost.

### Proposition 4 — A useful model must reduce decision error, not only patch error

For any fixed critic g and action, add and subtract its model-expected and environment-expected scores:

`|g(X,a,Zhat)-Q_ref(h,a)| <= |g(X,a,Zhat)-E_model g| + |E_model g-E_env g| + |E_env g-Q_ref|`.

These are sampling error, model-induced value error, and critic/compression error. If g is L-Lipschitz in the future features, the middle term is bounded by `L W1(P_model,P_env)` under a corresponding coupling; for deterministic next features it is bounded by `L ||Zhat-Zreal||`. The design does not establish a useful L, nor that pooled DINO distance matches action relevance. Lower mean patch MSE alone therefore does not imply lower ranking regret. Conversely a model can improve important margins despite a higher global feature error.

### Counterexample 3 — Memory/representation aliasing

Two legal histories produce identical scorer inputs, including M, q and generated futures, but require opposite action rankings because one has passed a landmark and the other has not. Any deterministic scorer gives identical scores and must fail on at least one. The recurrent architecture can distinguish some such histories but cannot certify sufficiency. Even before recurrence, 2×2 average pooling has an exact nullspace: opposite within-cell feature perturbations cancel. If a decisive cue lies in that difference, pooling removes it. This establishes a possible failure, not evidence that the specified DINO features actually discard a needed CE cue.

## 6. Minimal implementable objective paths — explicit extensions

The original trains only the scorer around frozen dynamics. The following are redesign candidates saved independently before other assignments. They should not all become components or separate claimed contributions. Use one focused prediction objective only after actual-future and no-future controls pass.

### A. Measured utility-gap preservation

For measured root outcomes U_a and frozen qualified scorer g, define

`dhat_ab(phi,xi)=g(X,a,Zhat_a(phi,xi_a))-g(X,b,Zhat_b(phi,xi_b))`,

`L_gap = mean_(a<b) w_ab (dhat_ab-(U_a-U_b))^2`,

`L_phi = L_native + lambda L_gap`.

Use all six pairs, or three base-action pairs, with predeclared bounded weights. STOP features stay fixed. Targets use actual branch returns, not an expert policy or maximized scorer output. Native reconstruction/diffusion training anchors outputs. Scorer-only paired supervision is a strong control: for fixed prediction errors e_a, `sum_(a<b)(e_a-e_b)^2=K sum_a(e_a-mean(e))^2`, so paired squared loss is centered regression, not an automatically novel learning principle.

**Implementation:** first train/freeze g; freeze its parameters with `requires_grad_(False)` but allow autograd through its feature input. Add an explicit future-gradient pathway to `score()` rather than removing frozen-input protection globally. Recompute only current predicted features; real context/memory may be cached under the frozen scorer. Optimize phi only. For CDiT, backpropagation through an entire denoising sampler plus VAE/DINO could exceed 8 GB; a clean-sample estimate from a native diffusion training step is a cheaper surrogate and must be named as such, since it is not the deployed full sampler. No feasibility claim precedes profiling.

**Stochastic caution:** if `D=dhat_ab` and target `Y=U_a-U_b` are independent conditional on the full root, then

`E[(D-Y)^2]=(E[D]-E[Y])^2+Var(D)+Var(Y)`.

Optimizing model phi therefore penalizes its value-gap variance as well as its mean error. This is not a calibrated objective for recovering the full stochastic transition law. It can intentionally stabilize single-sample decisions, but must not be described as learning truthful uncertainty. A two-independent-model-sample product `(D1-Y)(D2-Y)` estimates the mean-gap squared error up to target variance; its gradients may be noisy and native anchoring is still needed. This is an optional comparator with extra sampling cost, not a required new module or a novelty claim. The closely related sampling-calibration problem is established in [CVAML, §4.2 Eq. 3 and Proposition 3](https://arxiv.org/html/2505.22772v2); its paper is published at ICML 2025.

Measured targets reduce the temptation to optimize arbitrary high scores, but do not eliminate exploitation: a fixed g can be fooled off its training support even while matching observed gaps. Accept only improvement in held-out **actual branch regret and complete episodes**, with native prediction quality reported. Refitting g changes the experimental intervention; version it and rerun the fixed-scorer comparison. With no demonstrable sensitivity to futures, `grad_phi L_gap` can be zero or irrelevant.

### B. Outcome-weighted action-contrast prediction

An alternative avoids any learned-scorer gradient:

`L_contrast = mean_(a<b) w_ab ||(Zhat_a-Zhat_b)-(Zreal_a-Zreal_b)||^2`,

`L_phi=L_native+lambda L_contrast`, with bounded weights based on reliable `|U_a-U_b|`.

This prioritizes observed visual contrasts between action alternatives that change the frozen-policy outcome. It uses existing branch labels and features, no parser, reward judge, ensemble or additional model. For DINO dynamics it is directly implementable. A CDiT version through frozen decoder+DINO is costlier; weighting native diffusion examples by root/action usefulness is cheap but **not mathematically the same contrast objective** and must be reported separately.

The contrasts cancel common-mode errors, including harmful ones, so native anchoring remains necessary. Feature differences need not track language semantics; high outcome differences can arise from a brittle distant continuation. Generic value-aware and value-gradient-weighted learning already exist: [VAML, §2](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf) and [VaGraM, ICLR 2022 paper](https://openreview.net/pdf?id=4-D6CZkRXxI) are mandatory comparisons. Neither A nor B is established novel merely by applying utility weights to a CE world model.

### Coverage safeguard, not an extra architecture contribution

After an initial fit, collect some roots visited by the reranker in permitted scorer-fitting scenes, branch each candidate, and continue the **same** frozen pi_ref. Refit the scorer with aggregated roots while retaining untouched development scenes and a fixed continuation policy. This targets Proposition 1's required occupancy rather than redefining Q_ref. It is conceptually close to existing data aggregation/cost-to-go methods; [DAgger, §3 Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) and [AggreVaTe, §2](https://arxiv.org/html/1406.5979) preclude a broad novelty claim. World-model fitting on these new scenes would change the original A/B scene protocol and must be separately versioned rather than done silently.

## 7. Decisive next tests and decision rules

1. **Branch-label support first.** Run the specified 64-root/256-branch debugging pilot. Record candidate gaps, successful continuation counts, near ties, root-source mix, STOP effects, and repeated-seed variance where relevant. No candidate lift means the present candidate/horizon/reference-continuation combination lacks a useful lever; it does not prove all predictive navigation impossible.
2. **Actual-future critic against the same trained no-future critic.** Hold data, action/context paths, optimizer, capacity and episode splits fixed. If actual futures add nothing, extra world-model training is premature. If neither critic ranks measured useful alternatives, fix coverage/representation before attributing failure to prediction.
3. **Use of action consequences.** Evaluate non-STOP future permutation, action contrast sensitivity, and legal instruction/order tasks, while accounting for instruction/history leakage through q_t. A positive permutation effect only demonstrates sensitivity; correct held-out rankings establish usefulness.
4. **Measure all three error sources.** Compare actual versus predicted input regret, model-seed ranking variance, and new-policy occupancy errors. Report calibration per world-model source under the fixed pooled scorer. Keep raw action rank diagnostics separate from episode improvements.
5. **Only then one objective experiment.** Compare native training, A or B, and the same scorer improvement without future features. Keep policy/labels/candidate budget fixed; label any scorer refit and model-pool version. Test complete held-out episodes with prespecified development-selected margins and matched-runtime controls.

**What would support the mechanism:** repeatable action/episode gains beyond the no-future scorer, loss of gains when future/action correspondence is broken, lower actual-label regret attributable to improved predicted contrasts, and tolerable full pipeline runtime/memory. **What would not:** better images alone, good auxiliary success AUC, attention plots, synthetic tensor checks, or root ranking improvement without closed-loop improvement.

## 8. Executed checks and limitations

**[Executed]** [02_formal_checks.py](../evidence/02_formal_checks.py) exhaustively evaluates histories inside 500 random deterministic binary trees of horizon four, with path-dependent synthetic terminal success/route scores and incremental action/collision costs. All 7,500 history policy-improvement comparisons and 79 bounded-error margin cases passed. It also records the .55 sampling counterexample and the 0-versus-.6 occupancy counterexample. Results: [02_formal_checks.json](../evidence/02_formal_checks.json).

These are sanity checks of algebra and examples, not formal proof substitutes, CE metrics, world-model tests, navigation training, or 8 GB profiling. The reference scorer's existing synthetic tests were inspected through their documented scope, not rerun here. No backbone feature hooks, exact world-model checkpoints, actual CE labels, stochastic continuation calibration, or learned scorer accuracy were established by this assignment.

## 9. Primary-source and search ledger

All searches performed 2026-10-23. URLs below are primary sources. Word-limited summaries are intentionally brief; the detailed propositions above are our explicit deductions, not claims that these sources prove this CE system.

| Source / publication status | Exact locator inspected | Supported use and limit |
|---|---|---|
| Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd-edition author-hosted draft; textbook, not a new research paper | [March 2018 draft](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf), §4.2 pp. 78–79, Eqs. 4.6–4.9; primary-domain search extraction | Standard policy improvement. Direct PDF open timed out, but indexed primary-source text exposed exact statement/equations. Our finite-horizon path-return proof is supplied above. |
| Farahmand, Barreto & Nikovski, *Value-Aware Loss Function for Model-based RL*, AISTATS 2017, PMLR 54 | [Published PDF](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf), §2, Algorithm 1, pp. 2–3 | Task/value-relevant model error is established prior art; generic claim is unavailable. Does not establish CE language grounding or these specific branch contrasts. |
| Voelcker et al., *Calibrated Value-Aware Model Learning with Probabilistic Environment Models*, ICML 2025, PMLR 267:61745–61768 | [Proceedings record](https://proceedings.mlr.press/v267/voelcker25a.html); [author arXiv v2, 2025-06-09](https://arxiv.org/html/2505.22772v2), §4.2 Eq. 3/Prop. 3 and Appendix A.2.2 | Sampled value-matching can introduce model-variance bias; calibrated correction exists. We do not claim its MDP theorem directly guarantees our finite-horizon nonlinear scorer. |
| Voelcker et al., *Value Gradient Weighted Model-Based RL*, ICLR 2022 conference paper | [Official OpenReview PDF](https://openreview.net/pdf?id=4-D6CZkRXxI), p. 1 abstract/publication header | Establishes value-gradient-based weighting and model-objective mismatch as prior art. Only this scope is used; no uninspected theorem is imported. |
| Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011, PMLR 15:627–635 | [Proceedings record](https://proceedings.mlr.press/v15/ross11a); [PDF](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf), introduction and §3 Algorithm 3.1 | Learner-induced distribution and data aggregation are established issues/ideas. Its guarantees do not automatically transfer to this scorer. |
| Ross & Bagnell, *Reinforcement and Imitation Learning via Interactive No-Regret Learning* | [arXiv 1406.5979](https://arxiv.org/abs/1406.5979), submitted 2014-06-23; [full HTML](https://arxiv.org/html/1406.5979), §2 | Cost-to-go-aware aggregation is a close conceptual precedent. Status verified here only as arXiv technical report; no conference publication claimed. |

Search strings: `site.proceedings.mlr.press v267 voelcker25a value aware model learning`; `site.proceedings.mlr.press Ross Bagnell 2014 Reinforcement and Imitation Learning via Interactive No-Regret Learning`; `site.incompleteideas.net Sutton Barto policy improvement theorem 2018 pdf`; `value aware model learning Farahmand 2017 official proceedings`; `"Value Gradient weighted Model-Based Reinforcement Learning" "ICLR"`; `site.incompleteideas.net/book "4.2" "Policy Improvement"`; `site.proceedings.mlr.press "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"`.

Access failures recorded: guessed PMLR nested CVAML PDF URL returned 404; the proceedings' actual raw-GitHub PDF link returned unsupported octet-stream through the browser tool; arXiv v2 HTML succeeded. Author-hosted Sutton 2018 and RLbook2020 PDF opens timed out; primary-domain indexed extraction succeeded. A guessed PMLR 2022 VaGraM URL was inaccessible; official OpenReview PDF search result verified publication header. No absent search hit is used as evidence of novelty.

## 10. Claims ready for synthesis

- **Supported by reconstruction:** original information flow and label target are well-defined enough for an implementation; feature-level code does not implement CE integration or usefulness-aware model training.
- **Supported by deduction:** a full-history Q_ref improvement argument remains valid with terminal whole-path nDTW and root-relative remaining costs; practical failure arises when its value/coverage premises fail.
- **Promising but untested:** outcome-gap or action-contrast model losses could concentrate finite prediction capacity on useful action distinctions without extra modules.
- **Not established:** that predicted futures improve over a direct scorer; that the objectives are novel; that a scalar margin is calibrated safety; that the chosen memory is sufficient; that all world-model families fit the target GPU; or that benchmark SR improves under the composite utility.

The simplest defensible path is therefore the existing frozen-policy baseline plus one carefully isolated prediction objective, with paired-branch diagnostics demonstrating that it changes correct action decisions. If the no-future control or occupancy evaluation removes the gains, the predictive contribution should be rejected or revised rather than protected by additional machinery.
