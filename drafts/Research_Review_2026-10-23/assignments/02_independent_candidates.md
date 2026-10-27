# Assignment 2 — Independent mechanism candidates

Recorded 2026-10-23 before reading any other assignment's suggestions or results. Sources inspected at this point: REVIEW_BRIEF.md; original CE_World_Model_Approach.md and Scorer_Design.md in full; original Scorer/scorer.py, labels.py and config.json. These are independent mathematical proposals, not established novelty or effectiveness claims. No additional machinery is presumed; the three backbones and four agreed world-model trials remain intact.

## Candidate A: Preserve root-relative utility gaps, anchored to measured outcomes

Reasoning: action selection depends on Q(h,a)-Q(h,b), whereas ordinary patch MSE spends capacity equally on visually prominent errors that may not change this difference. Fit the existing scorer on actual and predicted features with measured branch outcomes. Then freeze a qualified scorer and train only the dynamics predictor using a native prediction loss plus a measured-gap loss:

`L_gap = average_(a,b) w_ab Huber[(S(x,a,Zhat_a)-S(x,b,Zhat_b))-(U_a-U_b)]`.

Use all within-root pairs or only pairs involving the base action. `w_ab` can ignore unreliable near ties using repeated branch outcome variance. STOP has fixed current features. Keep native prediction/diffusion loss as an anchor, and do not maximize scores: target observed signed differences. Predictions must be differentiable through the fixed scorer; the current code explicitly detaches future inputs and therefore must gain a separate, opt-in training pathway. This remains an extension, not the original scorer implementation.

Why promising: directly exposes the decision-relevant quantity, preserves the frozen policy and lightweight scorer, uses branch data already requested. Main risk: a fixed imperfect scorer can still be exploited; measured held-out outcome regret and real-feature fidelity are indispensable. The claim cannot simply be 'value-aware model learning', which already exists.

## Candidate B: Contrast-preserving prediction without a scorer gradient

Reasoning: use measured branch utility gaps to decide which observed changes deserve model capacity, while never allowing a learned scorer to reward hallucination. For DINO predictions use

`L = L_native + lambda average_(a,b) w_ab ||(Zhat_a-Zhat_b)-(Zreal_a-Zreal_b)||^2`,

with bounded `w_ab` increasing with measured `|U_a-U_b|`, and/or apply the same weights to per-action native loss. Pair differences remove a common-mode visual bias. This uses the existing encoder, model and branch labels; no new network or language parser. For CDiT, a directly comparable DINO contrast loss requires differentiating the denoised estimate through frozen VAE decoding and DINO encoding, which has real memory cost; branch-weighted native diffusion loss is the cheaper, less equivalent implementation.

Why promising: mechanistically clean and easily falsified. Main risks: DINO coordinate differences need not reflect semantic decisions; root-level outcome gaps may reflect long continuation sensitivity rather than one-step visual distinctions; weighting can neglect hard roots with nearly equal bad outcomes. It is a focused objective candidate, not a verified new method.

## Candidate C: Repair distribution coverage before adding architecture

Reasoning: the scorer estimates Q_ref from training roots, while repeated overrides induce another history distribution. Keep pi_ref frozen for labels, but collect a small fraction of new branch roots from the evolving reranker on permitted scorer-fitting scenes. Label each candidate by forced first action followed by the same pi_ref; retrain the same scorer. This is data aggregation for value estimation, not changing the continuation target to an expert or the reranker. Reserve untouched development scenes.

Why promising: exact Q_ref improvement is mathematically sufficient for repeated one-step improvement, so the important gap is accuracy on histories the reranker actually visits. No new module is needed. Main risks: extra simulator cost, adaptive evaluation leakage, and substantial overlap with existing dataset aggregation / policy iteration literature. This is likely an implementation safeguard rather than a standalone contribution.

## Candidate D: Residual advantage supervision within the existing scorer

Reasoning: retain the existing utility head and MSE, add within-root differences during scorer fitting. Pairing cancels root-level offsets, including past route damage and common costs. It can reduce optimization attention paid to predicting absolute utility where only comparisons matter. Keep an absolute loss so scores remain comparable to the base action and auxiliaries remain unchanged. This intervention trains the scorer only, not the world model, and therefore must be evaluated as a strong no-future baseline too.

Why promising: tiny implementation delta and meaningful baseline. Main risks: it could account for all apparent gains while futures are ignored; ranking losses and advantages are established ideas. Do not claim prediction benefit unless the same paired training without futures is worse at matched data and runtime.

## Preference before external comparison

Use D as a baseline/control, C as a coverage safeguard, and compare A versus B only after an actual-next-view scorer and a matched no-future scorer establish a useful predictive channel. A targets the quantity most directly; B offers a lower-exploitation alternative. Do not combine all four into a complicated system or count routine controls as separate paper contributions. The final mathematical report will derive conditional guarantees and counterexamples, verify closest mathematical precedents, and mark any later revisions explicitly.
