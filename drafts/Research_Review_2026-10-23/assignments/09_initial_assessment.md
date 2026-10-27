# Assignment 9 — Independent initial correctness assessment

Recorded 2026-10-23 before reading Assignment 8, the main report, Assignment 10, or Assignments 1–7. This is a native independent worker assessment, not external-model review.

Reviewed frozen proposal: `proposal_v1.md`, SHA256 `4CBC57E6781D83BF3C3A88F9030C54EFBD39A45B8B98535EB68D5DD163D77F9F`. I directly checked the hash, read the proposal, original `CE_World_Model_Approach.md`, `Scorer_Design.md`, and `Scorer/scorer.py`, and consulted the source index only as a locator. Original materials and the proposal remain unchanged. I directly opened CVAML v2, World4Scorer v1 and MCR v1 primary HTML, including CVAML §4.2 Eq.3/Proposition3 and MCR §3.3.2 Eqs.6–7. Additional detailed source/experiment inspection follows this saved assessment.

## Initial disposition

Conditionally coherent, mathematically defensible research candidate; not validated implementation, not established improvement, and not two established contributions. I see no immediate algebraic contradiction in the centered deterministic score loss, its candidate coupling, or the stated regret inequality. The frozen critic can propagate gradients to predictor inputs if the existing detach is explicitly bypassed. The actual code currently prevents that path, exactly as v1 admits. The original scorer assumes 64 pooled patches and hard-coded 8×8 positions, so the proposed native-grid correction is an implementation change, not merely a tensor reshape.

The scientific argument can cohere as (1) identify consequential instruction/route-specific ordering errors that finite predictive computation can help with, then (2) correct those same errors by changing predictor training. C1 alone does not justify C2: a geometry/outcome mismatch could instead arise entirely from a mis-specified geometric scoring rule, a history bottleneck, or return-label noise. Demonstrating C2 requires an accurate frozen critic near the adapted feature distribution and benefit over equally supervised direct/weighted losses. Calling these two contributions remains conditional on a consequential, generalizable finding and a correction that survives close controls.

## Strongest initial objection

The finite critic provides a useful objective only where it evaluates altered predictions faithfully. Optimizing its frozen output is otherwise an ordinary route to off-support score manipulation. A native prediction anchor is no guarantee. The proposal acknowledges this, but qualification must be an executable held-out rejection criterion, not visual examples or reduced training task loss. If adaptation improves learned scores without improving measured independent continuation outcomes, the mechanism claim fails.

A separate critical attribution issue is exact information matching: a deterministic predictor cannot add Shannon information beyond its own real input history and action. If it receives additional recent raw features that the direct critic does not receive, improvement can reflect the missing-history input rather than predictive computation. “Full allowed context” must be instantiated as the union of legal predictor/scorer histories with explicit capacity and compute controls, while recognizing that actual-next observations are privileged information. This is a clarification to check in A7 before final disposition.

## Preliminary checks and limits

- For residual e=s−y, P=I−11ᵀ/K is K×K, and the task term is ||Pe||²/K. Its score gradient is 2Pe/K; the displayed chain-rule form is correct up to the expressly omitted scale.
- The all-pairs squared residual-gap sum equals K||Pe||². A pair-averaged implementation requires an explicit factor; the written identity itself is correct.
- With true expected values and argmax scores, regret ≤ sqrt(2)||Pe|| follows by bounding the best-versus-selected centered residual difference. It does not bound finite-sample estimation, generalization, margin-based decisions, or actual closed-loop performance without additional assumptions.
- The utility includes whole-path nDTW and remaining-budget costs. Exact policy-improvement reasoning is available on the full augmented history, but need not survive compressed scorer inputs or approximate values. Fixed-reference continuation labels are not inherently a logical error.
- Stochastic-model squared-gap fitting adds a model score-variance penalty. CVAML §4.2 directly supports the need to distinguish mean matching from sample matching. Starting with deterministic DINO avoids this specific model-sampling term, not partial-observability or noisy-label difficulty.
- STOP requires no predicted state and produces no predictor gradient of its own; its fixed score still anchors candidate-relative residuals. This is coherent but needs testing at terminal/budget edge cases.
- The current code detaches predicted futures, uses independent candidate processing, and raises on nonfinite scores. v1 correctly labels the gradient route and fallback as proposed changes.
- Existing code/synthetic evidence cannot establish CE efficacy or full 8 GB feasibility. A meaningful decision requires branch reproducibility, independent-seed opportunity, direct-value controls, channel tests, frozen-critic qualification after adaptation, and clustered closed-loop evaluation.
- An exact-Q greedy improvement result would be compatible with occupancy concerns: occupancy threatens approximation/support, not the exact full-history policy-improvement theorem.

## Initial stop criterion

Stop expansion of this particular expensive prediction mechanism if adequate direct/full-history controls match it, if no repeatable candidate opportunity exists, or if adaptation increases frozen-critic scores without held-out outcome improvement. Underpowered evidence should remain inconclusive. Do not infer absence of all predictive signal from one underfit actual-next critic.

## Primary locators already consulted

- [CVAML v2, §4.2, Eq.3 and Proposition3](https://arxiv.org/html/2505.22772v2#S4.SS2).
- [World4Scorer v1, Appendix C.4 and D.2](https://arxiv.org/html/2609.36438v1); direct primary opening confirmed the joint-predictor/outcome/future-anchor precedent and the frozen-LeWM outcome-scorer variant; detailed appendices will be checked next.
- [MCR v1, §3.3.2, Eqs.6–7](https://arxiv.org/html/2608.09073v1#S3.SS3.SSS2): dead-reckoned deviation-based cost ranking plus native rollout anchor is already explicit.

