# FeedTTA

Test-Time Adaptation for Online Vision-Language Navigation with Feedback-based Reinforcement Learning

Sung June Kim; Gyeongrok Oh; Heeju Ko; Daehyun Ji; Dongwook Lee; Byung-Jun Lee; Sujin Jang; Sangpil Kim. ICML 2025. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [FeedTTA.pdf](../../Paper/FeedTTA.pdf). Pages reviewed: [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18].

## Problem and operating conditions

A pretrained VLN policy receives a stream of test tasks and a binary success/failure label after each episode. It must improve future navigation without stepwise expert trajectories. Feedback is an additional supervision source: validation experiments use outcome labels, while the hidden-goal test split uses an LLM oracle. This setting differs fundamentally from unlabeled FSTTA and from frozen-policy zero-shot inference.

## Mechanism, training, and inference

Terminal feedback F∈{−1,+1} is discounted backward as G_t=γ^(T−t−1)F and multiplies each chosen action’s log-probability gradient (Eqs 1–3). Language/visual encoders are frozen; cross-modal and subsequent policy modules adapt. Batch 1, lr 5e−6, one A100 are reported (§4.3). Stochastic Gradient Reversion (SGR) independently selects gradient coordinates with Bernoulli probability p and multiplies them by negative α, with a scaling correction for other coordinates. It is stochastic optimization noise, not an actual alternate trajectory or causal intervention.

The printed Eq 5 and AppendixB.1 disagree: Eq 5 scales only unselected coordinates by 1/(αp+1−p), while the appendix normalizes the entire modified gradient. The latter is expectation preserving; the literal Eq 5 generally is not. Moreover both negative-α forms have expected absolute multiplier greater than 1 when the denominator is positive. The claimed shrinkage in Eqs 8–9 does not follow. The paper’s approximation assumes |α|<p, whereas REVERIE uses α=−.2 with p=.01/.05. R2R/CE implementation details even list α=+.1, outside the stated negative-reversion mechanism. These are mathematical/reporting limitations, not a refutation of the benchmark results.

The LLM oracle first extracts the goal, then judges a final panorama. AppendixA explicitly permits category substitution and partial alignment, ignoring exact placement. It therefore evaluates broad goal appearance rather than complete route or instance compliance. Adapted Success Rate averages success retention and conversion conditional on the frozen base outcome; it balances the two conditional groups rather than measuring ordinary population success.

## Novelty and closest predecessors

Relative to FSTTA, the key change is outcome-conditioned online REINFORCE plus coordinate gradient perturbation. The meaningful experimental comparison needs matched access to binary labels, update parameters and episode order. It is prior art for learning from episodic failure and for using a VLM as a terminal verifier, but its oracle is intentionally lenient about spatial detail.

## Results and independent verdict

PAPER EXPERIMENT: Table 1/p6 reports DUET unseen REVERIE SR 46.98→66.49/SPL 33.73→45.38 with outcome feedback, but hidden test SR 52.51→53.58 using the LLM oracle. Table 4/p8 gives GPT-4o feedback accuracy .72, precision .68, and navigation SR 52.29/SPL 33.56: SR improves over base 46.98 but SPL is slightly below base 33.73. Reliable feedback is a major boundary condition. Table 5 separates unregularized feedback RL 63.14 SR from SGR 66.49.

AppendixD shows sequence dependence: randomly interleaved episodes 65.33 SR versus scene-contiguous 54.81, both above 46.98 base but far from invariant behavior. In per-scene case 7 SPL falls 35.21→30.53 despite SR rising. Table 6’s seen-retention result raises SR but lowers SPL 63.94→62.74, so “no forgetting” is metric-specific. Randomly corrupted feedback experiments do not establish robustness to systematic false positives from the lenient LLM prompt. Verdict: strong evidence that extra outcome supervision can improve this stream protocol; mathematical explanations of SGR and claims of broad oracle reliability need correction and controlled verification.

## Implementation and reproducibility

No official FeedTTA implementation was located in the supplied paper, official PMLR page, author-site search, or exact-title/GitHub searches as of the cutoff. The available ATENA repository is a subsequent distinct method and is not relabeled as FeedTTA code. Consequently the SGR implementation variant, REINFORCE sampling/greedy behavior, and exact terminal feedback path cannot be verified here. Reproduction requires these missing details and released policy assets. Only independent scalar checks of the printed mathematics were executed.


## Specific bottlenecks

**feedtta-B1: Lenient final-image feedback can reinforce a semantically wrong instance or position.**

Conditions: Similar-category distractors, partial visibility, or instructions requiring an exact spatial relation or route. Cause: Oracle prompt explicitly accepts broad category and partial alignment and ignores exact placement. Evidence: AppendixA/PDF p13; GPT-4o precision .68 in Table 4.

Type: methodological. Status: Partially addressed. Confidence: High for prompt and measured precision; specific failure rates unmeasured.. Follow-ups: ATENA reduces query frequency with a learned outcome head; Context-Nav/PolarMem add verification cues; none removes the need to validate oracle semantics in this protocol.

**feedtta-B2: The stated SGR mean-preservation and shrinkage explanations do not jointly follow from the printed formulas.**

Conditions: Negative α with positive normalization denominator, including reported REVERIE settings. Cause: Main equation and appendix scale different branches; first-order approximation drops terms affecting the inequality. Evidence: Eqs 5,8–9 and AppendixB.1; isolated scalar checks recorded separately.

Type: methodological. Status: Uncertain because evidence is insufficient. Confidence: High for algebraic inconsistency; actual implementation unknown.. Follow-ups: No correction located in this bounded review; empirical utility could remain even with an incorrect theoretical explanation.

## Sources and limitations

- [Supplied main paper](https://raw.githubusercontent.com/mlresearch/v267/main/assets/kim25ad/kim25ad.pdf)
- [Official ICML proceedings and bibliography](https://proceedings.mlr.press/v267/kim25ad.html)
- Official code not located in the bounded search. No navigation benchmark executed. The statistical association between weight norm and plasticity is not a proof of forgetting prevention.
