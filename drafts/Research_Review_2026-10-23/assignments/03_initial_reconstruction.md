# Assignment 3: independent reconstruction before peer exposure

Recorded 2026-10-23. This file was written before opening `02_formulation.md`, `02_independent_candidates.md`, or any other assignment. Only the review brief and original ten-file approach/scorer materials were consulted. The ARS devil's-advocate contract is used as a scoped review aid; this is a native model review, not cross-model verification.

## Strongest faithful reconstruction

The proposal is a finite-horizon, history-conditioned action-value reranker around a frozen CE navigation policy. It trains the policy first, fits CE dynamics separately, and then freezes both during initial scorer fitting. The scorer consumes instruction/context features, current patches, eight real-history slots, action identity, elapsed budget, and a one-step predicted observation. It ranks all four primitive actions. STOP is terminal, takes the current view as an identity placeholder, and receives no dynamics prediction. Candidate evaluation does not update real memory. Only real execution does.

Training labels come from actual candidate interventions at the same restored simulator/history root, then real-observation continuation under the frozen base policy. The target is `0.70 success + 0.30 whole-path nDTW - 0.01 remaining_steps/B - 0.02 remaining_collisions/B`. The scorer is trained by utility MSE plus auxiliary success and immediate-FORWARD-collision BCE. Real-next inputs are a capacity diagnostic/curriculum, and predicted-next inputs are the deployment condition. Predictions from the four agreed world-model variants are pooled into a common fixed scorer per policy. A development-selected positive override margin is required operationally, but is explicitly not a safety theorem.

The initial scientific hypothesis is empirical: explicit predicted features may improve finite-capacity action ranking and closed-loop navigation beyond the same scorer/context/action without predicted views. No implemented usefulness-aware world-model objective or proven novelty is claimed. The three backbones and four world-model families remain recorded comparisons, not optional deletions.

## Independent mathematical assessment

1. **The baseline is internally coherent, with an important distinction about the target.** Exact history-based `Q_ref` can be defined under a fixed continuation policy. The deployed compressed scorer learns a conditional mean given its available inputs, which need not equal the value of the particular privileged simulator root. Distinct hidden states or lost-history prefixes can produce identical inputs and different labels. Privileged labels remain legitimate supervision; zero Bayes error is not promised. The source already acknowledges compressed-history approximation (Scorer Design 88–115).

2. **Replanning is not by itself a contradiction.** With exact values at every reachable full history, common finite-horizon utility, and a candidate set containing the base action, selecting an action with nonnegative base-policy advantage supports the standard policy-improvement argument. Nonadditive whole-path nDTW can be a terminal trajectory reward with prefix/history augmentation; it does not alone invalidate the argument. Approximation, restricted coverage, stochastic sampling, and deployed history compression remove an automatic guarantee. The source already says closed-loop validation is needed.

3. **Generated observations add computation/inductive bias, not a new observation of the current hidden world.** For fixed dynamics weights and independent random seed, a generated future is a function of real history, candidate command and model randomness. Conditional on the full real history and action, it cannot reveal which possible hidden state or stochastic real outcome is true. It can still help a capacity-limited scorer or one receiving a more compressed history than the model. A no-future control matches scorer capacity but not the full system's learned computation; a positive result must be framed accordingly.

4. **The actual-next curriculum and sampled-next deployment do not have identical conditional prediction problems.** Real next features may correlate with the actual branch's latent outcome; a separately sampled model future usually does not, even with a perfect conditional transition distribution. Under independent noise, the population MSE minimizer for sampled futures need not reproduce an oracle actual-next value function. With complete history, it should ignore residual model noise. Selecting the largest single-sample oracle value can underperform expected-value action choice even for a perfect generative model. The source acknowledges sampling variability but does not spell out this identification distinction.

5. **Frozen weights versus stopped input gradients must be separated for the later extension.** The existing scorer explicitly calls `future_patches.detach()` (scorer.py 149–153), so a loss passed through that reference cannot train the world model. This is intentional/correct for the frozen-world-model baseline, not an implementation bug. A future usefulness-aware dynamics extension would need a distinct differentiable score-input route plus a frozen-but-differentiable DINO/VAE adapter; setting scorer parameters to frozen alone would not block input Jacobians. No extension is implemented or falsely claimed in the original.

6. **Degeneration is a valid baseline outcome, not a contradiction.** Action/context paths can predict labels without imagined patches. A fixed pooled scorer may exploit or ignore artifacts of specific dynamics families despite receiving no model ID. Permutation/no-future tests and held-out-model tests distinguish these possibilities. Instruction and added-memory ablations must also remove the corresponding policy-context information; the original explicitly requires this.

7. **The composite utility is not SR or instruction truth.** Realized successes outrank failures under its bounds, but expected-utility ranking need not maximize success probability; route fidelity can favor an annotated path over a semantically valid alternative. Both are already explicitly acknowledged. These are scope restrictions, not newly discovered flaws.

8. **One actual continuation is one conditional outcome, not an identified expectation under stochastic continuation.** Restoring a random state makes comparisons reproducible and may reduce variance; it does not establish averaging over stochastic policies or physical outcomes. Repeated independent-seed rollouts are required to estimate an expected target. Deployment-only predictor seeds should not be coupled to real simulator randomness so as to leak the outcome. Original wording already distinguishes deterministic and stochastic labels.

## Concrete checks to perform next

- Independently verify full-history policy improvement with a terminal path-dependent reward and an explicit score-error margin bound; do not repeat a simplistic claim that fixed-policy continuation necessarily invalidates replanning.
- Construct a small two-action sampled-future counterexample and an aliasing lower bound, without any CE training.
- Verify the input-gradient boundary directly without writing into the original reference directory.
- Compare Assignment 2's exact target and candidate theorem to these conditions; distinguish a baseline flaw from a condition added for a redesign.
- Inspect primary metric/policy-improvement sources and key official code paths directly, rather than inheriting cited prior-report verdicts.

## Source ledger and uncertainty

Inspected in full: `CE_World_Model_Approach.md`, `Scorer_Design.md`, `README.md`, both trial JSON files, `Scorer/scorer.py`, `labels.py`, `config.json`, `check_design.py`, and `check_results.json`. Key source locations are given above. No external primary evidence or numerical tests had yet been executed when this reconstruction was saved; any source-dependent claim remains to be independently checked. No peer assignment was read. No original file was edited.
