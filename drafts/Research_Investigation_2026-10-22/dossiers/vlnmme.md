# VLN-MME

VLN-MME: Diagnosing MLLMs as Navigation Agents

Xunyi Zhao; Gengze Zhou; Qi Wu. ACL 2026, pp. 28207–28231. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [VLN-MME.pdf](../../Paper/VLN-MME.pdf). Pages reviewed: ['3–9', '12–25'].

## Problem and operating conditions

Compare zero-shot MLLM backbones and prompt-based memory/reasoning designs using cached views instead of live rendering. The tasks cover R2R, REVERIE navigation without object grounding, and graph-adapted MP3D ObjectNav; physical control and multilingual/interactive dialogue are excluded.

## Mechanism, training, and inference

A Model/Agent/Task interface and Runner load cached cardinal images, connectivity and caption metadata. Four 90-degree views are rearranged as left/front/right/back after heading quantization to the nearest cardinal direction. GPT-5 produces marker captions; BLIP-2 plus GPT-5 supplies viewpoint summaries. Two memory families use textual action summaries or an accumulating textual topological graph. Each has direct, CoT, reflection, and combined prompts. A parser extracts action identifiers and retries invalid outputs. No model parameter updates occur. Stratified subsets preserve scene/path distributions; evaluation uses standard endpoint and trajectory metrics. The cost comparison removes rendering and continuous waypoint prediction, so its speedup is framework-specific.

## Novelty and closest predecessors

A modular simulator-free diagnostic evaluation and agent ablations, not a new continual-learning method. It makes failure inspection practical but keeps graph connectivity, cached views and generated caption inputs, which differ from physical sensors and from raw-video-only policies.

## Results and independent verdict

PAPER EXPERIMENT: Table 2 Qwen2.5-VL-7B text-summary R2R SR 27.5→21.0 with CoT; Gemini baseline 41.0→32.5. This is a result for specified prompts/models, not a theorem against reasoning. Appendix E analyzes 200 Qwen map-memory trajectories: 148 failures, including 17 generation errors and 131 navigation errors, of which 106 loop; 42/52 successes also loop. Only two are described as perfect. On 25 selected hard negatives, Qwen3VL-assisted navigation reaches 52% SR versus selected baseline 0%; few-shot failure examples reach 16%. Assistance changes available information and policy, so this does not causally establish perception is adequate and only planning fails. Appendix Table 6 contradicts a universal 2–3-point proxy claim: NaviLLM REVERIE SR 45→34 and SPL 37→27 on the subset. Appendix D.4 describes reflection revision, while D.8 says reflection/decision metadata do not directly alter control flow; source behavior must arbitrate.

## Implementation and reproducibility

INSPECTED CODE: src/agents/baseline_agent.py:41–145 parses an Action field, gives numeric action matches priority, maps direction words to the first matching available candidate, checks navigability and handles Stop. Therefore textual formatting, option multiplicity and parsing are part of the measured agent pipeline. src/utils/eval.py implements graph nDTW with three-metre normalization and endpoint success. The repository exposes separate baseline, CoT, reflection and MapGPT agents; these are API/model inference strategies rather than a VLN-MME training method. Dataset construction and caption/oracle condition parity still require frozen annotations and configurations. No model/API/simulator execution or full cross-agent behavioral equivalence test was performed.

Inspected release reference: [https://github.com/billzhao1030/VLN-MME at 7d37bd6d1763c28a382201c1634cf93e7cff4a2b](https://github.com/billzhao1030/VLN-MME/tree/7d37bd6d1763c28a382201c1634cf93e7cff4a2b). File-level inspection is restricted to the paths listed below.

- src/agents/baseline_agent.py:1–145
- src/utils/eval.py:5–40
- Pinned agent strategy tree

## Specific bottlenecks

**vlnmme-history: Provided textual history and topology do not reliably prevent repeated wrong actions.**

Conditions: Qwen map-memory agents on the inspected 200-episode subset and tested zero-shot prompt designs. Cause: Models may fail to use history, interpret regions/verticality, or map recognition to actions; the evidence does not isolate these causes. Evidence: PAPER EXPERIMENT: Appendix E, Figures 13–14.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['M-Track', 'Meta-Explore', 'NavNuances']

**vlnmme-proxy: A small benchmark subset is not uniformly a close numerical proxy for every policy.**

Conditions: Comparing methods whose errors concentrate on selected scene/path categories. Cause: Stratification on observed attributes does not control all policy-specific failure distributions. Evidence: PAPER EXPERIMENT: Appendix Table 6 REVERIE NaviLLM difference.

Type: insufficiently tested. Status: Uncertain because evidence is insufficient. Confidence: High within the stated experimental scope. Follow-ups: ['NavNuances']

## Sources and limitations

- [Supplied main paper](https://aclanthology.org/2026.acl-long.1300.pdf)
- [Official proceedings](https://aclanthology.org/2026.acl-long.1300/)
- [Official implementation](https://github.com/billzhao1030/VLN-MME)
- Embedded appendix and diagnostic source reviewed; Figures 15–19 assessed through captions, without independent trajectory relabeling. No benchmark executed.
- Associated material: Embedded appendices A–F, pp. 12–25
