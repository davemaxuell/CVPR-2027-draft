# Assignment 3 — Adversarial logical and mathematical review

Date/cutoff: 2026-10-23. **Verdict: the original baseline is logically coherent; revise the statistical interpretation and evaluation protocol before claiming that prediction improves decisions.** No fatal contradiction was found. The important risks are measurable, and several were already openly acknowledged in the original. They must not be presented as newly discovered defects or as proof that the approach cannot work.

Independence: I reconstructed all ten original approach/scorer files and saved [03_initial_reconstruction.md](03_initial_reconstruction.md) before opening Assignment 2. I then inspected `02_formulation.md` and its test source/results. I did not read other peer assignments. This is a separately executed native-model review, not cross-model independence. Original source files were not edited; no CE training, large model execution, or weight download was performed.

Evidence labels: **[Specification]** original proposed design; **[Code]** inspected source; **[Executed]** bounded local tests; **[Deduction]** reasoning here; **[Published]** independently checked primary evidence; **[Hypothesis]** empirical claim still requiring CE evidence.

## 1. What is actually proposed

**[Specification]** This is a frozen-policy action-value reranker. It predicts one observation for each movement primitive, scores the four actions using instruction/context, real-history memory and candidate features, executes one action, then observes reality. STOP is terminal and uses the current view as a placeholder. Labels come from an actual forced candidate followed by the frozen base policy, with the original remaining budget and whole-path metric prefix. The primary value is the composite

`U_t = .70 S + .30 N - .01 n_t/B - .02 c_t/B`.

The initial world models are frozen during scorer training. Utility MSE and auxiliary losses train the small scorer. A usefulness-aware dynamics objective is only a later, explicitly unfinished extension. Prediction benefit, CE performance, novelty, and full 8 GB feasibility are not established by the reference tensor checks. See original `CE_World_Model_Approach.md` §§3–8; `Scorer_Design.md` §§3–7.

The three recorded navigation backbones and four world-model trial families remain experimental options. Nothing in this mathematical review warrants silently deleting those comparisons or treating an unimplemented extension as the baseline.

## 2. Important claims that survive adversarial review

### Fixed-policy continuation does not invalidate repeated reranking

**[Deduction]** Let the global episode return be

`J = .70 S + .30 N - .01 n_total/B - .02 c_total/B`.

At a given root, `U_t=J+k(h_t)` where `k=.01 t/B+.02 c_prefix/B` is the same for every candidate. Thus root-relative labels preserve action advantages for one globally consistent return. Whole-path nDTW can be a terminal trajectory reward in a full-history, finite-horizon process. Nonadditivity in the current image is not an obstruction.

If at every history visited by the new policy its expected selected `Q_ref` is at least `V_ref`, the performance-difference telescoping sum is nonnegative. The base action's inclusion is useful but insufficient without accurate advantages on that occupancy. For a stochastic reference, the comparison must be to expected `V_ref` or to a coupled sampled base action, not merely the reference's mode. Assignment 2 states these conditions correctly. The underlying standard policy-improvement result is independently verified in [Sutton and Barto §4.2, Eqs. 4.6–4.9](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf); the trajectory-return adaptation above is our deduction.

**[Executed]** A2's unchanged 500-tree test was replayed to a separate output directory and matched its saved result exactly. An independently written stochastic four-action tree test, including terminal STOP and stochastic reference policies, passed 2,150 history comparisons. This is a sanity check, not evidence that the learned CE scorer satisfies the premise.

### The margin arithmetic is correct, but the required bound is not measured

**[Deduction]** If all four score errors at a visited history satisfy `|qhat_a-Q_a|≤epsilon`, an estimated gain `g>delta` implies true gain `>delta-2epsilon`. Hence `delta≥2epsilon` is sufficient locally for a deterministic base. The margin rule's regret is at most `delta+2epsilon`; unrestricted argmax gives `2epsilon`. A development-selected scalar margin and average validation MSE do not establish a simultaneous error bound on new histories and predictor seeds.

**[Executed]** Independent enumeration of 1,296 four-action cases passed both local safety and regret checks. The original already calls its margin heuristic and leaves overrides disabled until validation; it does not falsely claim certified safety (`Scorer_Design.md` 157–165).

### The model is neither inherently myopic nor logically obliged to use futures

**[Deduction]** One predicted action does not mean one-step reward: its learned target includes the later frozen-policy continuation. Conversely, the scorer can predict that target directly through action/context/history and ignore its future tokens. This is an allowed empirical degeneration. The original explicitly requires no-future controls, future permutation, instruction-context controls and closed-loop results (`Scorer_Design.md` 167–189). A critic ignoring future tokens would defeat the predictive contribution, not contradict the architecture.

## 3. Findings that need operational fixes

### F1 — A sampled model future and an actual next view answer different conditional questions

**Severity:** major for any claim that a high-value imagined sample identifies the real branch outcome; not a fatal flaw of predicted-input MSE training.

**[Deduction]** With full available history `h`, fixed model parameters and independent model noise, `Zhat=F(h,a,xi)` contains no extra information about which current hidden state or future environment draw is real. Real next observation `Y` can contain such information. Consequently,

`g_real(h,a,Y)=E[U|h,a,Y]`, whereas `g_pred(h,a,Zhat)=E[U|h,a]=Q_ref(h,a)`

in the full-history idealization. Conditional independence need not hold after replacing `h` by the scorer's compressed input `X`: predicted features may encode useful history information discarded by `X`. That is a valid representation/finite-capacity benefit. It is not an observation of an unexecuted future.

**Utility-compatible counterexample, independently executed.** A has utility `.8` surely. B has `.9` or `.1` with equal probability, so `Q(B)=.5`. A perfect generative model samples B correctly, and a perfect actual-future evaluator assigns its sampled realized utility. Choosing B whenever its imagined utility is `.9` yields

`E[U_selected]=.5×.5+.5×.8=.65 < .8`.

All three numbers fit the proposed utility with `B=500`, `n=2`, `c=0`: their `(S,N)` values are `(1,.3334667)`, `(1,.6668)`, and `(0,.3334667)`. These are abstract utility-compatible outcomes, not claimed CE trajectories. This repairs A2's `.6`-sure example, which is valid for a generic reward but lies between the original realized-failure and realized-success ranges.

**Exact fix:** choose the intended critic semantics explicitly. The simplest baseline is the original predicted-input/actual-label conditional-mean scorer, finishing on predicted inputs and measuring variation across generator seeds. If instead treating the critic as a realized-next-state evaluator, use an explicit expectation over predicted futures and validate that estimator. Neither an actual-view curriculum nor a perfect transition distribution by itself justifies plugging one sample into the actual-view critic. Report separate actual-input and predicted-input calibration; do not call their outputs interchangeable.

**Decisive test:** at repeated real roots, cross the real continuation seeds with independently sampled model seeds. Check whether per-seed score variation predicts anything about real return beyond the root/action mean. Test the mean score and single-sample selector separately at their actual runtime. In deterministic real branches, model-seed variability remains a robustness issue even though real label variance vanishes.

### F2 — One-rollout “oracle headroom” can be selection optimism

**Severity:** major if a stochastic 64-root pilot is interpreted as expected candidate improvement; absent for deterministic branches with deterministic continuation.

**[Specification]** The original correctly says one stochastic rollout is not an expectation (`Scorer_Design.md` 99), but its proposed best-measured-candidate pilot and regret summaries do not specify a correction for maximizing noisy labels (169–185).

**[Deduction/Executed]** Suppose three movement actions each independently return `.9` or `.1` with equal probability. Each has `Q=.5`; the expected maximum of their single measured outcomes is `.8`. The apparent oracle lift over any base is `.3`, while an independently evaluated selected action still has expected return `.5`. A fixed inferior STOP can be added without changing the example. Even common random numbers do not guarantee removal of optimism: two equal-Q actions with anticorrelated seed responses can have a measured maximum `.9` for every shared seed.

This distinction also changes the meaning of “ranking regret”: regret to the maximum sampled outcome is not regret to `max_a Q_ref(h,a)`. Reproducible seeds do not make those estimands equal.

**Exact fix:** first establish whether simulator and continuation are deterministic under the intended evaluation protocol. If stochastic, use repeated paired seed sets to estimate means, select actions on one set, and estimate selection performance on independent continuation seeds. Report seed variance and episode/scene uncertainty. Keep “best realized branch” as a privileged debugging statistic, labeled accordingly. Do not launch a larger model experiment because one noisy pilot maximum looks favorable.

**Decisive test:** estimate the gap between same-seed best-branch utility and held-out-seed utility of the selected action. An apparent gain that disappears here is label selection noise, not useful one-step control opportunity.

### F3 — Privileged labels are legal, but root-specific outcomes may not be identifiable from deployed inputs

**Severity:** major for claims of exact instruction-state recognition; a limitation already partly acknowledged by the original.

**[Deduction]** At two equally likely hidden roots, suppose the deployed scorer inputs are identical and the two useful actions have values `(.9,.1)` and `(.1,.9)`. Every observation-based deterministic scorer must give the same scores at both. The Bayes conditional mean is `(.5,.5)`, minimum scalar MSE is `.16`, and expected regret to a hidden-state oracle is `.4`. These values were checked numerically. This is a possibility theorem, not evidence that actual CE episodes have exactly identical learned features.

Loss of information can arise from physical partial observability, truncated policy context, learned recurrent compression, or pooling. A reference path or simulator pose used only in supervision does not become available at inference. The benchmark proxy also cannot certify each instruction clause. Original text explicitly acknowledges both facts; do not present them as newly detected leakage or inconsistency.

**Exact fix:** distinguish simulator-state root labels from conditional values under available observation history. Interpret irreducible residuals accordingly. Use legal pairs of tasks requiring different route stages, and compare full real-history features, added memory and actual-next observations. If a cue is unobservable even with the available history, change the claim or sensing requirement; additional attention layers cannot resolve it.

**Decisive test:** find high-confidence errors and determine whether task-relevant cues were present in raw available history, lost in the policy/scorer representation, or absent altogether. Report this taxonomy before blaming world-model feature fidelity.

### F4 — Future usefulness training needs a separate differentiable contract and a declared statistical objective

**Severity:** major for a later implementation or claimed novelty; not a bug in the original frozen-model baseline.

**[Code/Executed]** `Scorer/scorer.py` 149–153 explicitly detaches future patches. The input gradient is `None` even though changing those patches changed the candidate score by approximately `.009851` in the bounded check. A separate frozen linear layer still passed input gradients `[2,3]`, confirming that frozen parameters and stopped input gradients are distinct. No world-model objective can train through the original reference route unchanged. Original code does exactly what its frozen-world-model specification requires.

**Exact implementation fix for the extension only:** keep scorer parameters fixed but expose an opt-in differentiable future-input route. Ensure the VAE-decoder/DINO adapter uses differentiable tensor operations without `no_grad`, detached caching, image serialization, or discrete preprocessing that severs the route. The present baseline need not change. Test nonzero finite gradients into intended dynamics/action parameters and zero gradients into frozen parameters. A clean diffusion-training estimate is a different surrogate from the deployed denoising sampler and must be evaluated as such.

**[Deduction]** For an independent generated score gap `D` and real gap label `Y` conditioned on a complete root,

`E[(D-Y)^2]=(E D-E Y)^2+Var(D)+Var(Y)`.

Thus A2's single-sample gap objective also penalizes model score variance. A scalar example with independent symmetric `±1` samples has loss `2` for the truthful stochastic distribution but `1` for a deterministic zero predictor. That is not automatically undesirable for action selection, but it is not calibrated learning of the transition distribution. A two-independent-model-sample product removes the model variance term in expectation; this adds variance/compute and does not identify arbitrary multimodal dynamics.

The related sampling-calibration issue and correction are established in **[Published]** [CVAML §4.2, Eq. 3 and Propositions 2–3](https://arxiv.org/html/2505.22772v2), published at [ICML 2025, PMLR 267](https://proceedings.mlr.press/v267/voelcker25a.html). These are prior results, not a new contribution of the CE proposal. Their MDP theorem is not a ready-made guarantee for the proposed nonlinear history-conditioned scorer.

**Decisive test:** report decision regret, model-seed score variance, native prediction quality and actual episode outcomes together. A decrease in the gap loss alone cannot distinguish useful mean correction, intentional variance suppression, or critic exploitation. Keep the original native-loss baseline and do not add every proposed safeguard as a separate contribution.

## 4. Direct review of Assignment 2

| A2 item | Adversarial disposition |
|---|---|
| Full-history reconstruction, STOP and memory semantics | Agree; matches original source and code. |
| Root-relative costs and terminal nDTW | Agree; ranking constants cancel. No invented Markov-reward objection. |
| Policy improvement and margin propositions | Agree under their stated occupancy/error assumptions; independent stochastic and four-action checks passed. |
| Actual versus predicted conditional means | Agree; central distinction independently identified before peer exposure. |
| Information redundancy | Agree only conditioned on the full model inputs and fixed parameters; compressed-scorer representation gains remain possible, as A2 states. |
| `.6`-sure stochastic counterexample | Correct generic mathematics; replace with utility-compatible `.8` vs `.9/.1`, return `.65`, when discussing this design. |
| Actual-next views described as a “diagnostic ceiling” | Tighten to “privileged diagnostic.” A finite trained actual-view scorer is not a proven numerical upper bound on a predicted-view scorer. An information-set/Bayes-optimal comparison would need additional assumptions. |
| Measured oracle regret | Add repeated/split-seed evaluation when stochastic; the same-label maximum can exaggerate usable headroom. |
| Gap objective and input gradients | Agree as a labeled extension, with the gradient and variance qualifications above. It is not implemented, and generic paired regression is not novel. |
| Action-contrast objective | Plausible surrogate, not a proof of instruction-relevant prediction. Common-mode error cancellation and continuation-dependent labels need actual-outcome evaluation, as A2 notes. |
| Proposed coverage collection | Agree as a targeted remedy for new occupancy; do not describe it as a new general data-aggregation principle or silently alter the original scene split. |

The same-capacity no-future scorer remains necessary but does not match the full predictive pipeline's extra learned computation, pretraining and runtime. A positive result supports explicit predictive representation under that budget; it does not establish information gained by hallucinating the future. The original and A2 already call for complete cost accounting.

## 5. Executed evidence and reproducibility

Source: [03_adversarial_checks.py](../evidence/03_adversarial_checks.py). Results: [03_adversarial_checks.json](../evidence/03_adversarial_checks.json). Command: `py -3.13 -B evidence/03_adversarial_checks.py` from the review directory. CPU PyTorch was `2.9.0+cpu`; the first attempt with the default MSYS Python lacked Torch, then the installed Python 3.13 environment succeeded without package installation.

- Original utility helper reproduced `.8`, `.9`, `.1` with valid bounds and equal step costs.
- The sampled-selector return was `.65`; noisy three-action oracle mean was `.8` versus true action value `.5`.
- Aliasing calculation produced minimum MSE `.16` and hidden-state oracle regret `.4`.
- Sample-gap variance and two-sample-product arithmetic matched the stated expectations.
- 1,296 four-action margin/regret cases passed; 2,150 stochastic-history policy-improvement cases passed.
- Original future-input gradient was absent while forward score dependence was present; frozen weights alone retained input gradients in the separate check.
- Unchanged A2 source replay exactly matched its JSON output, saved separately under `evidence/03_a2_replay/`.
- All original scorer file hashes were identical before and after checks. No original check runner was invoked because it writes into its own source directory.

Limits: these are finite arithmetic and tensor checks, not CE evidence, formal-proof substitutes, semantic tests, model accuracy estimates, GPU profiling, or novelty verification. The stochastic examples are utility-compatible abstract constructions; none is claimed to occur with measured frequency in R2R-CE/RxR-CE.

## 6. Primary-source/search ledger

Only a focused verification was performed; no broad literature sweep or absent-search-hit novelty claim was used.

| Source | What was independently inspected | Status/limits |
|---|---|---|
| Original approach/scorer ten-file set | Entire files; critical locators cited above. `scorer.py` SHA-256 `296c15901df8c806d0cc21f5bf5880895929dfe1c181538ac0f29d44b7336e8a` | User-provided design/code, not published empirical evidence. |
| [Sutton and Barto author-hosted draft](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) | Primary-domain indexed text of §4.2, Eqs. 4.6–4.9 | Textbook primary source. Direct PDF fetch timed out; exact indexed source passage was available. No secondary theorem summary used. |
| [CVAML official proceedings](https://proceedings.mlr.press/v267/voelcker25a.html), [author arXiv v2](https://arxiv.org/html/2505.22772v2) | Proceedings publication metadata; full §4.2 Eq. 3/Propositions 2–3 and surrounding assumptions | ICML 2025 published paper; arXiv v2 dated 2025-06-09. Checked sampling claim and scope directly. |
| [Pinned VLN-CE measure code](https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/habitat_extensions/measures.py) | `StepsTaken` 123–138; `NDTW` 218–263 including stored positions, reference lookup, duplicate-position skip and Euclidean point distance | Official code primary evidence; supports retaining exact evaluator configuration/prefix. Does not establish clause-level instruction correctness. |
| [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd) | `requires_grad`, no-grad, and input-graph semantics | Current official documentation; local executed check uses 2.9.0 CPU and is the direct evidence for this reference. |

Search strings: `site.incompleteideas.net/book "Policy Improvement" "4.6" "4.7"`; `site.docs.pytorch.org requires_grad False gradients input detach autograd`. Other sources were opened directly at the above URLs. Inaccessible source: author-hosted Sutton PDF direct fetch timed out; primary-domain indexed extraction succeeded. No external model APIs were used.

## 7. Prioritized next decisions

1. Keep the existing baseline and its explicit unknowns. Define deterministic versus stochastic continuation before collecting and interpreting the branch pilot.
2. Estimate usable candidate lift without maximizing and evaluating on the same noisy outcomes. If it is absent, revisit candidates/horizon with a matched label protocol.
3. Require predicted-input scoring to beat the matched no-future scorer in held-out episodes. Use actual-next views to localize error, without assuming a learned actual-view model is a guaranteed ceiling.
4. Only after that evidence, choose one labeled usefulness objective. Verify its input-gradient route, semantics under stochastic futures, held-out real-outcome gains and full cost.

The strongest counterargument remains: **the method may mainly be an additional supervised action-value network with expensive, redundant predicted tokens.** The existing no-future and consequence-correspondence tests can answer that criticism. The current evidence cannot decide the empirical outcome; it does show that neither fixed-policy continuation nor path-dependent nDTW is, by itself, a mathematical reason to reject the experiment.
