# Active Visual Exploration

Active Visual Information Gathering for Vision-Language Navigation

Hanqing Wang; Wenguan Wang; Tianmin Shu; Wei Liang; Jianbing Shen. ECCV 2020. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [Active Visual Exploration.pdf](../../Paper/Active%20Visual%20Exploration.pdf). Pages reviewed: ['1–16'].

## Problem and operating conditions

An instruction follower can move briefly to obtain useful visual evidence before committing to its next navigation direction. This requires explicit accounting of exploratory motion, including motion that reveals an already cached view.

## Mechanism, training, and inference

An EnvDrop-style LSTM navigator encodes panoramic ResNet-152 and orientation features into 512-dimensional visual tokens and attends to 512-dimensional instruction representations. A nested exploration policy chooses a direction or exploration STOP. An exploration LSTM and knowledge LSTM summarize newly attended neighbors, producing residual updates to candidate-direction features. Multi-step exploration may inspect several directions before the navigator acts. Late action uses cached observations and executes only necessary paths to new views in the static navigation graph. Supervised shortest-path navigation and exploration initialization precede actor-critic training. Navigation rewards use goal-distance progress and terminal rewards; exploration rewards measure change in downstream navigation reward minus a 0.1 step cost. Separate critics are used. Curriculum increases maximum exploration from one to six steps, with four selected; 176,000 speaker-generated examples augment training. Exploration at inference is physical lookahead, not a learned world-model rollout.

## Novelty and closest predecessors

Learns when and where to gather new visual evidence, rather than only attending to a fixed panorama. It precedes ULN’s uncertainty-triggered physical branches and explicitly includes trajectory cost. Late action is a cache/execution optimization under a static graph, not a simulator-free imagined transition model.

## Results and independent verdict

PAPER EXPERIMENT: Table 3 unseen baseline SR/SPL/TL 0.52/0.48/10.7; naive exploration 0.54/0.16/35.7; learned one-step 0.55/0.42/13.7; four-step 0.58/0.40/20.6; six-step 0.58/0.38/23.6. Table 1 test SR increases EnvDrop 0.51→0.60 while SPL falls 0.47→0.41 and TL rises 11.7→21.6. These are reliability/efficiency tradeoffs, not metric-wise dominance. Page 14 reports exploration at 15.3% of navigation steps, correcting 65.2% of initially wrong actions but corrupting 10.7% of initially correct actions. Pre-exploration with synthetic-data fine-tuning and beam search are separate settings: the latter has TL 204.4 and SPL 0.05 despite SR 0.70.

## Implementation and reproducibility

INSPECTED CODE: active/agent.py:293–343 constructs training reward/labels by comparing action outcomes with simulator goal distances and restoring the simulation state. exploration:346–415 maintains already-explored candidate masks, an exploration policy and a default four-step lookahead; greedy evaluation is explicit. These oracle distances are training supervision, not an independently calibrated confidence estimate. late_action_taking:1893–1923 removes repeated visits and reconstructs connections through stored graph paths; this supports the paper’s separate delayed-execution protocol and should not be mixed with online physical exploration costs. The train path clips gradients and steps encoder/decoder/explorer optimizers. No project source was executed, and an exhaustive replay accounting audit was not performed.

Inspected release reference: [https://github.com/HanqingWangAI/Active_VLN at d5dabd5fe6127bcfec023b90f14a4ba5ac671f9b](https://github.com/HanqingWangAI/Active_VLN/tree/d5dabd5fe6127bcfec023b90f14a4ba5ac671f9b). File-level inspection is restricted to the paths listed below.

- active/agent.py:293–415,1724–1794,1893–1923

## Specific bottlenecks

**activevig-cost: Exploration improves success while consuming motion and sometimes corrupting a correct choice.**

Conditions: Ambiguous landmarks or long exploration branches, especially when the initial policy action was already correct. Cause: Information value is learned through a downstream reward proxy, while new evidence can mislead the policy and incurs physical travel. Evidence: PAPER EXPERIMENT: Table 3 and p. 14 action-change analysis.

Type: methodological. Status: Partially addressed. Confidence: High within the specified evidence scope. Follow-ups: ['ULN', 'Meta-Explore', 'Explore until Confident']

## Sources and limitations

- [Supplied main paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670307.pdf)
- [Official implementation](https://github.com/HanqingWangAI/Active_VLN)
- Main methods/results and late-action source reviewed; official ECVA page and pinned tree expose no separate appendix. Exact physical trajectory accounting and the printed RL sign conventions were not independently reproduced. No benchmark executed.
