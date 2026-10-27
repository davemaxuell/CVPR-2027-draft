# OmniAgent

Native Active Perception as Reasoning for Omni-Modal Understanding

Zhenghao Xing; Ruiyang Xu; Yuxuan Wang; Jinzheng He; Ziyang Ma; Qize Yang; Yunfei Chu; Jin Xu; Junyang Lin; Chi-Wing Fu; Pheng-Ann Heng. ICML 2026; PMLR 306:140133–140152. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [OmniAgent.pdf](../../Paper/OmniAgent.pdf). Pages reviewed: ['1', '3–9', '13–20'].

## Problem and operating conditions

Answer questions about long videos by selectively retrieving frames, audio or clips. The environment is a media archive with arbitrary temporal access, not a mobile robot or an unknown navigation scene.

## Mechanism, training, and inference

Qwen2.5-Omni-7B follows an Observation/Thought/Action loop. At each turn it reads the previous transient media plus persistent textual history, writes a textual observation and reasoning record, and calls frame/audio/clip retrieval or answer. Previous raw media are removed while timestamps and textual summaries persist. Agentic SFT uses 58,000 successful teacher trajectories from five training datasets, filtered by task outcome and GPT-4o coherence score at least 3/5. Two SFT epochs use batch 64, lr 1e-5, 16 A100s. RL uses harder queries under 300 seconds, group eight, 150 steps, batch 256, lr 1e-6, 64 A100s. TAURA normalizes trajectory rewards and multiplies each turn advantage by its mean token entropy divided by group mean turn entropy. A clipped policy objective uses these rescaled advantages, with no KL or entropy regularizer. At inference context is 64K; the prompt requests confidence at least 0.9 before answering but forces a best answer at the final step.

## Novelty and closest predecessors

Native multimodal active retrieval plus textual memory consolidation and entropy-weighted RL extends agentic video reasoning. Active perception is established in EmbodiedQA/IQA/Explore-EQA; this method instead exploits nonphysical media access. Entropy-correlated decision forks are not causal information gain, and prompt confidence is not calibrated uncertainty.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 VSI-Bench 35.5→48.4 and LVBench 43.0→50.5 versus the base model, with additional SFT/RL data and inference. Table 4 LVBench agentic SFT 48.7, vanilla GRPO 49.8, TAURA 50.5; DailyOmni 63.3/62.2/64.8. Appendix C’s 79.2% high-entropy fork statistic uses Gemini-labeled top forks on successful traces, without a causal intervention or failure-sample control. Table 7 on 100 LVBench cases: base 34.8 seconds/41% accuracy; OmniAgent 66.8 seconds/51%, despite fewer frames than a 72B comparator. Memory media size is bounded per turn, but textual history and serial latency grow with turns. DEDUCTION: mean turn weights equal one, yet unequal token counts, signed advantages and gradient directions mean this does not prove preservation of original gradient magnitude. No benchmark or gradient test was executed here.

## Implementation and reproducibility

INSPECTED CODE: verl/trainer/ppo/core_algos.py:950–989 computes each trajectory’s masked mean token entropy, divides it by the group mean entropy and broadcasts detached weights over tokens. ray_trainer.py:1392–1409 stores these weights under entropy_weights when the corresponding configuration is enabled; a distinct soft-weight option exists and must not be confused with TAURA. Mean-one trajectory weights do not prove unchanged token-weighted gradients or constant optimizer-step magnitude. The release supplies video action parsing/environment, rewards, multimodal rollout and SFT/RL framework code. The source snapshot is newer maintained code with optional reward/weight variants; paper-equivalent configuration must be frozen. No models, video inference or training were run. This is video-tool reasoning, not a physical navigation benchmark.

Inspected release reference: [https://github.com/harryhsing/OmniAgent at 0ce93ec0b9f6de9cc22057327d90b0b33914849f](https://github.com/harryhsing/OmniAgent/tree/0ce93ec0b9f6de9cc22057327d90b0b33914849f). File-level inspection is restricted to the paths listed below.

- verl/trainer/ppo/core_algos.py:950–989
- verl/trainer/ppo/ray_trainer.py:1380–1418
- agent_system/environments/env_package/video_env.py: inspected action/reward entry-point definitions

## Specific bottlenecks

**omni-memory: Purging raw media makes later reasoning dependent on the faithfulness and completeness of textual observations.**

Conditions: An important detail was omitted or misdescribed, or a later question changes which information matters. Cause: Only selected textual summaries survive; recovering evidence requires another archive query. Evidence: METHOD: Appendix B.4 and prompt warning; DEDUCTION for irrecoverable physical observations.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['VSI-Bench', 'Explore until Confident']

**omni-credit: Entropy weighting is a useful heuristic but is not established causal credit assignment.**

Conditions: High-entropy filler or confidently wrong pivotal decisions. Cause: Outcome-level advantages are redistributed by entropy rather than an intervention on action value. Evidence: PAPER EXPERIMENT: Table 4; limited correlation audit Appendix C.

Type: methodological. Status: Uncertain because evidence is insufficient. Confidence: High within the stated experimental scope. Follow-ups: ['UoT']

## Sources and limitations

- [Author manuscript v2, 17 July 2026](https://arxiv.org/abs/2606.19341)
- [Official implementation](https://github.com/harryhsing/OmniAgent)
- [Official ICML proceedings; title, authors, year and pages verified](https://proceedings.mlr.press/v306/xing26g.html)
- Official ICML publication verified; source inspection is restricted to documented paths. Historical model/configuration equivalence and causal benefit of entropy weighting remain untested. No model execution.
- Associated material: Embedded appendices A–D, pp. 13–20
