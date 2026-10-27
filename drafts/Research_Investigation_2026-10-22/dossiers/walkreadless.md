# NAP

Walk and Read Less: Improving the Efficiency of Vision-and-Language Navigation via Tuning-Free Multimodal Token Pruning

Wenda Qin; Andrea Burns; Bryan A. Plummer; Margrit Betke. EMNLP 2025, pp. 23556–23570. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [NAP.pdf](../../Paper/NAP.pdf). Pages reviewed: ['1', '3–9', '13–15'].

## Problem and operating conditions

Removing visual/text tokens can lower per-step compute while increasing navigation length and removing critical action options. NAP changes inference of existing trained HAMT, DUET, and GOAT agents without additional parameter tuning.

## Mechanism, training, and inference

Background pruning retains all navigable action-view tokens and removes low column-summed attention background tokens at each visual layer. Backtracking pruning retains only top-scored unvisited graph frontier nodes from earlier steps, reducing both graph tokens and available recovery destinations. Vocabulary priority pruning first builds an offline training-lexicon dictionary using Llama 3 relevance judgments; at each language layer it preferentially removes dictionary-irrelevant tokens, with attention ranking filling or reducing the pruning quota. The LLM is not called during navigation. The cost metric sums instruction encoder FLOPs once and visual/cross-modal FLOPs over decision steps using thop. CE uses ETPNav and its waypoint predictor. No new loss or fine-tuning is required, but the base policy, feature encoders and waypoint predictor are pretrained.

## Novelty and closest predecessors

Task-specific foreground protection and graph-action pruning improve the accuracy/compute frontier relative to generic token pruning. The vocabulary is a context-independent relevance heuristic, not instruction entailment or semantic-preservation certification.

## Results and independent verdict

PAPER EXPERIMENT: Table 2 GOAT at roughly 50% tokens has R2R SR 73.8 versus full 78.1 and FLOPs 55.6%; RxR 63.9 versus 69.6 and FLOPs 49.3%; REVERIE 49.6 versus 53.8 and FLOPs 48.3%. The gain is over other pruned baselines, not full-model accuracy. Table 3 RxR SPL 61.8→54.3, steps 8.2→9.0. CE Table 7 full 56.9 SR/8.8 steps→52.7/9.6 with FLOPs 62.4%; pruning does not always shorten motion. Appendix F Table 13 calls one and flight irrelevant in a stair instruction: this is a documented prompt output and an untested numerical-language risk, not measured failure frequency. FLOPs do not establish wall-clock latency on target hardware. Appendix Table 11 has different step counts from main Table 2, limiting direct cross-table interpretation.

## Implementation and reproducibility

INSPECTED CODE: The release is explicitly labelled an old, uncleaned version in README; final hyperparameters and VLN-CE code remain TODO. The inference rollout wraps language and panorama computation in no_grad and uses a persistent list of removed candidate viewpoints (map_nav_src/r2r/agent.py:476–571). BTP removes the earliest encountered unvisited nodes beyond its threshold from model input (agent.py:134–184). BGP normally preserves action-candidate features and ranks only background features; its action_included ablation can prune actions (models/pruning/bgp.py:107–206). VPP prioritizes tokens using a context-free irrelevant-token ID set, then restores the surviving tokens’ original order (models/pruning/vpp.py:1–99). These are tuning-free operations on an already trained navigator, not an untrained model. The main entry point runs evaluation and computes environment metrics. README’s admitted release mismatch and unreleased final hyperparameters limit paper-equivalent reproduction; no timing or FLOP experiment was executed.

Inspected release reference: [https://github.com/wdqin/VLN-NAP at e6eededcda9d4e1fb504da892176746af8c6ce89](https://github.com/wdqin/VLN-NAP/tree/e6eededcda9d4e1fb504da892176746af8c6ce89). File-level inspection is restricted to the paths listed below.

- README.md
- map_nav_src/r2r/agent.py:134–205,476–571
- map_nav_src/models/pruning/vpp.py:1–99
- map_nav_src/models/pruning/bgp.py:107–206
- map_nav_src/r2r/main_nav.py:190–240

## Specific bottlenecks

**nap-semantics: Context-independent word pruning can remove quantities or relational words that distinguish routes.**

Conditions: Ordinal, count, or context-sensitive instructions under tight token budgets. Cause: A vocabulary-level relevance label ignores sentence composition; attention importance is not semantic necessity. Evidence: AUTHOR EXAMPLE: Appendix F Table 13 labels one and flight irrelevant; DEDUCTION for route effects.

Type: methodological. Status: Uncertain because evidence is insufficient. Confidence: High within the stated experimental scope. Follow-ups: ['NavNuances', 'GELA']

**nap-recovery: Deleting graph frontier options trades recovery access for compute and sometimes still increases motion.**

Conditions: The needed corrective route is a low-attention frontier or pruned observations cause uncertainty. Cause: BTP changes the action set, not only an internal representation. Evidence: PAPER EXPERIMENT: Tables 3 and 7.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['Meta-Explore', 'ULN']

## Sources and limitations

- [Supplied main paper](https://aclanthology.org/2025.emnlp-main.1202.pdf)
- [Official EMNLP implementation](https://github.com/wdqin/VLN-NAP)
- Pruning source reviewed, but the release itself labels hyperparameters/code as uncleaned and does not establish final paper configuration equivalence. No benchmark executed.
- Associated material: Embedded appendices A–F, pp. 13–15
