# Rejected method proposal: P3 Joint future events

Cutoff: 22 October 2026. **P3 is rejected as an original-method proposal and is not included among the three retained studies.** The original draft remains available for provenance. No experiment is proposed merely to retain a fourth idea.

The underlying intuition survives as a diagnostic: an expected additive feature can lose an ordered future distinction. However, [predictive state representations](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) already encode ordered future tests, and [Horde/general value functions](https://josephmodayil.com/papers/horde-final.pdf) can predict cumulants on an augmented event state. Distributional successor methods, semantic future QA and sequence predictors supply further close precedents. A K-by-K pair-event head is not a new world-model principle. See the [independent novelty review](../evaluations/09_novelty.md), especially `09-P3-01/02` and its source records.

Assignment 09 gives the exact representational reduction: augment state with a three-stage monitor (first event unseen, first observed, ordered pair completed) and a finite-horizon counter. Update the second-event transition from the monitor state **before** the current observation, preserving strict order. Keep completion absorbing, use zero transient reward, and give terminal reward one exactly when completion occurred by H. Under the same continuation policy, expected terminal reward equals the ordered-event probability. This is an evaluator deduction about representational inclusion, not a new theorem attributed to those papers or a learnability claim. Occupancy-based distributional methods are close controls, not automatically equivalent encodings of arbitrary temporal order.

NavQ's target is an **expected discounted sum** of 512-dimensional future caption features under its specified continuation construction, not a normalized semantic average or scalar optimal Q-value. Source: `code/Q_pretrain/dataset.py:189–222` and `GPT_VL.py:236–239`, commit `9465fe43c9c1583bd7ba390b6be30996ea4f5a5c`. Its downstream predictor is frozen/detached in the inspected navigation path; adding a navigation-side head would not automatically train it.

The draft's AA/BB versus AB/BA example matched per-time marginals but changed ever-occurrence probabilities, so it did not establish a distinction beyond the proposed single-event control. Assignment 08 supplied and executed the stronger example:

| Uniform distribution | Continuations | P(A ever), P(B ever) | P(A before B) |
|---|---|---|---|
| Left | AB0, B0A, 0AB | 1, 1 | 2/3 |
| Right | A0B, 0BA, BA0 | 1, 1 | 1/3 |

Here 0 is neutral. At each time, both distributions assign probability 1/3 to A, B and 0. Thus, for any fixed per-event feature map and discount, linearity gives the same expected discounted sum, while ordered occurrence differs. The [independent check](../experiments/08/logic_results.json) verifies the finite example; equality for arbitrary fixed discount follows from matching timewise marginals. This is not proof that real NavQ caption vectors collide, that its full history-aware policy lacks the information, that the future distinction is observable, or that the event is common in its tasks.

Separate binary p/J heads also need not form a coherent trajectory distribution. Bounds `J_AB<=min(p_A,p_B)` are necessary but not sufficient; when A and B each occur once at distinct times with certainty, `J_AB+J_BA=1` adds another constraint. Repeated occurrences change the constraint. None of these outputs supplies reachability or calibrated risk under a changed continuation policy.

A future replication could compare controlled representations, history-only/direct value, future QA and sequences with identical query inputs, labels, continuations and capacity. It would require real target-level headroom, learnability and a separate route-task port. That is a possible diagnostic project, **not an endorsed new method or an additional recommended proposal**. No such model or benchmark experiment has run.
