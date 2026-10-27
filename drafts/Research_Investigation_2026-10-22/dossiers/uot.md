# UoT

Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in LLMs

Zhiyuan Hu; Chumin Liu; Xidong Feng; Yilun Zhao; See-Kiong Ng; Anh Tuan Luu; Junxian He; Pang Wei Koh; Bryan Hooi. NeurIPS 2024. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [UoT.pdf](../../Paper/UoT.pdf). Pages reviewed: ['1', '3–10', '14–29'].

## Problem and operating conditions

Select useful follow-up questions when the correct answer is hidden. Evaluations are simulated 20 Questions, simplified diagnosis, and troubleshooting, not clinical deployment or embodied navigation.

## Mechanism, training, and inference

An LLM proposes candidate questions from dialogue history and a candidate hypothesis set. Each simulated answer partitions hypotheses into disjoint yes/no sets; experiments assume a uniform prior. Up to three question/answer levels form a tree. Binary answer entropy is divided by 1+abs(p_yes−p_no)/lambda (lambda 0.4 in the plotted reward) to favor balanced partitions. Rewards accumulate from root to leaf; expectation backs up using answer probabilities and uniform averaging over future generated questions. The current root question with highest backed-up score is asked. Thus future question nodes average candidates rather than maximizing a Bellman-optimal continuation. Closed-set evaluation provides the hypothesis list; open-set mode uses LLM-generated lists of 5–10 candidates and refreshes them, with three initial direct-prompt turns for 20 Questions. No model fine-tuning occurs, but repeated model calls supply simulation and a GPT-4 answerer supplies environmental responses.

## Novelty and closest predecessors

Information-theoretic question scoring and lookahead replace unconstrained self-evaluation in tree-style prompting. The underlying information-gain identity is standard for deterministic partitions; the sharpened reward is a heuristic and is not identical to Shannon information gain.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 GPT-4 closed-set Common SR 50.5→71.2, DX 91.3→97.0, FloDial 43.7→67.3. Open-set Common SR 48.6→55.3 while mean dialogue length slightly increases 17.1→17.4; improvement is not universal efficiency. Table 2 equal-token comparisons give pruned UoT 4.7K tokens and 20Q SR 48.1 versus adapted ToT 4.5K/33.8; full UoT 9.2K/54.4. Appendix F human audits only 10% of simulated records, finding intermediate feedback accuracy 92.9–94.4%, so simulator truth is imperfect. §5 explicitly acknowledges overlapping/noisy hypotheses and incomplete elimination. The printed Eqs. 5–6 omit entropy minus signs; Appendix A restores the intended negative entropy but an intermediate displayed algebra step drops probability factors. The final binary-entropy identity is valid under deterministic disjoint partitions; these typesetting inconsistencies should not be treated as an invalid identity.

## Implementation and reproducibility

INSPECTED CODE: src/uot/uot.py:85–155 partitions candidate sets, implements the sharpened binary-entropy reward, and supports average or maximum future-child aggregation. The executable reward has the correct entropy minus signs absent from the printed equation. However, avg_expected multiplies its child average by a branch probability and expected_reward multiplies again by that branch probability (115–147); paper-equivalent expected-utility interpretation should be checked before using this implementation as a calibrated decision rule. idiv_reward also assigns reward 1 when the two partition cardinalities differ by one (103–107). These are deterministic code semantics, not evidence of benchmark impact. README includes API-dependent closed models and an open-set possibility-update protocol. No API interaction or model output was executed.

Inspected release reference: [https://github.com/zhiyuanhubj/UoT at e08f17ff4ceec0f734da66e04d7d35df04ae6460](https://github.com/zhiyuanhubj/UoT/tree/e08f17ff4ceec0f734da66e04d7d35df04ae6460). File-level inspection is restricted to the paths listed below.

- README.md
- src/uot/uot.py:85–155

## Specific bottlenecks

**uot-elimination: Hard hypothesis deletion is unreliable when answers are noisy, ambiguous, or compatible with several hypotheses.**

Conditions: A symptom is absent despite a disease, or LLM simulated partitions are wrong. Cause: Yes/no sets are assumed disjoint and exhaustive; uncertain likelihoods are replaced by elimination. Evidence: AUTHOR ACKNOWLEDGEMENT: §5; PAPER EXPERIMENT: Appendix F imperfect feedback.

Type: methodological. Status: Still unresolved in the reviewed literature. Confidence: High within the stated experimental scope. Follow-ups: ['KnowNo', 'AIUTA']

**uot-search: Tree lookahead spends many model calls and optimizes a heuristic average continuation.**

Conditions: Larger candidate spaces, richer answer types, deeper planning. Cause: Branching grows with questions and answers, and the backed-up reward averages future choices. Evidence: METHOD: §2.5; PAPER EXPERIMENT: Table 2 and Appendix E.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['OmniAgent']

## Sources and limitations

- [Official implementation](https://github.com/zhiyuanhubj/UoT)
- Task prompts and source reward logic reviewed. The simulator answerer knows the hidden target and is prompted to answer truthfully; no clinical validity or navigation performance is established. No project execution.
- Associated material: Embedded appendices A–I partly reviewed and selected prompts, precise pages listed above
