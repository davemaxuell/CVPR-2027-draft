# ULN

ULN: Towards Underspecified Vision-and-Language Navigation

Weixi Feng; Tsu-Jui Fu; Yujie Lu; William Yang Wang. EMNLP 2022. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [ULN.pdf](../../Paper/ULN.pdf). Pages reviewed: ['3–9', '13–19'].

## Problem and operating conditions

An agent trained on detailed route instructions often fails when users omit details or provide only a destination. R2R-ULN progressively simplifies existing instructions into L1 (redundancy removed), L2 (partial route), and L3 (goal description). It tests English indoor graph navigation, not arbitrary contradictory instructions.

## Mechanism, training, and inference

A pooled-word classifier selects low-level versus high-level instruction processing. Granularity Specific Sub-networks (GSS) are identified by transplanting components from a separately trained goal-only agent and measuring validation gain: the recurrent VLN-BERT text encoder or HAMT cross-modal encoder is the selected component. That component is retrained on R2R-last with the remaining agent frozen. At inference an uncertainty MLP consumes visual-to-text attention and action logits. It is trained from randomly dropped training sub-instructions, with teacher-action disagreement as the binary label. Above threshold 0.5, the agent physically explores top candidates and scores current plus future maximum logits (Eq. 7); history state is frozen to delay premature text progression. Reported settings use one-step lookahead, gamma 1.2, and at most three exploration triggers. Training uses original R2R and last sentences, mixed imitation/A2C, AdamW on one GPU: base/classifier 300,000 iterations, selected high-level component 10,000 iterations, uncertainty model 10 epochs. No ULN evaluation instructions are required as training examples, but the architecture-selection protocol measures L3 validation performance and should be separated from held-out testing.

## Novelty and closest predecessors

Combines granularity-specific routing with a learned grounding-error trigger and explicit lookahead. Active VLN instead trains a separate exploration policy using navigation reward; ULN reuses its policy and freezes history. It is prior art for uncertainty-triggered navigation exploration and text-conditioned expert routing, not a zero-training method.

## Results and independent verdict

PAPER EXPERIMENT: Table 4, PDF p. 7, recurrent baseline unseen SR/SPL 39.3/35.0; GSS 42.3/37.7; GSS plus E2E 44.7/29.7, with path length 12.95→23.02 m relative to GSS. HAMT GSS 43.5/39.5 becomes 44.6/25.9 with E2E and path length 12.91→28.31 m. Table 7 makes the tradeoff explicit: threshold zero can collapse SPL although SR rises. Table 5 does not uniformly support improvement at every level: HAMT L1 SR 56.4→55.5 with full method. Human L3 SR is 75 versus L0 86 (Table 2), so underspecification can also increase human ambiguity. Appendix Table 12 finds adding REVERIE/SOON is not automatically useful. Default evaluation subsamples levels at unequal rates; full-set ablations must not be mixed with Table 4 denominators. The paper establishes useful SR gains, not efficient or calibrated uncertainty resolution.

## Implementation and reproducibility

INSPECTED CODE BY ASSIGNMENT 05 at 0f251df3ffc6dd9c538d075db8674e6ad54a596b: The released recurrent VLN-BERT implementation feeds attended visual/language feature vectors to the uncertainty MLP, ignoring the score tensors; this differs from the raw-score formulation in paper Eq 6 (model.py:865–882). Classifier training uses agreement with a stored rollout-result path, produced by train.py:267–287, through _teacher_action(from_env=False); these labels are not independently verified oracle correctness. Exploration scores explicitly accumulate raw future logits, confirming the lack of cross-state additive-gauge invariance in this scoring path (agent.py:683,754–755,857–892). The first step cannot trigger exploration and at most three events are allowed (1065–1074). Parallel exploration simulators and filtered/spliced reported viewpoints mean physical travel accounting needs a separate matched replay audit; no direction or magnitude of cost error is established. Granularity routing and dual language encoders are present; the exact isolated high-level-encoder freeze/reset phase was not located. Full-method test switches are commented in the launcher and the HAMT implementation remains unreleased according to README. Full exact paths/ranges: evidence/05/uln_code_audit.md. No project code was executed.

Inspected release reference: [https://github.com/weixi-feng/ULN at 0f251df3ffc6dd9c538d075db8674e6ad54a596b](https://github.com/weixi-feng/ULN/tree/0f251df3ffc6dd9c538d075db8674e6ad54a596b). File-level inspection is restricted to the paths listed below.

- Assignment 05: evidence/05/uln_code_audit.md; ULN-Rec-VLN-BERT model.py:865–882; agent.py:270–294,620–627,683,754–755,857–892,1065–1074; train.py:267–287; README and launcher

## Specific bottlenecks

**uln-exploration-cost: Uncertainty-triggered exploration increases success while sharply reducing path efficiency.**

Conditions: Underspecified instructions, repeated uncertainty triggers, and physical lookahead/backtracking. Cause: The trigger predicts disagreement with a teacher action rather than calibrated value of information; future-logit scores are heuristic and exploration is not free. Evidence: PAPER EXPERIMENT: Tables 4 and 7, Appendix Table 13; AUTHOR ACKNOWLEDGEMENT: §5.2.3 uncertainty is least accurate near the early corrections where intervention matters.

Type: methodological. Status: Shifted into another cost or limitation. Confidence: High for the measured SR–SPL tradeoff; no universal impossibility claim. Follow-ups: ['Meta-Explore', 'Explore until Confident', 'KnowNo', 'AIUTA']

## Sources and limitations

- [Official proceedings](https://aclanthology.org/2022.emnlp-main.429/)
- [Author-linked implementation](https://github.com/weixi-feng/ULN)
- The recurrent model source was inspected independently by assignment 05; HAMT release remains incomplete. Parallel branch simulations and spliced reported paths require replay to establish comparable physical cost. No such replay executed.
- Associated material: Appendices A–B embedded in main PDF, pp. 13–19
