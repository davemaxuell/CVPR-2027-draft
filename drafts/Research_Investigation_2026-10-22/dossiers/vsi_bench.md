# VSI-Bench

Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

Jihan Yang; Shusheng Yang; Anjali W. Gupta; Rilyn Han; Li Fei-Fei; Saining Xie. CVPR 2025. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [VSI-Bench.pdf](../../Paper/VSI-Bench.pdf). Pages reviewed: ['1–8'].

## Problem and operating conditions

Evaluate spatial reasoning from indoor video, including counting, dimensions, distances, directions, route planning, and appearance order. This is offline video question answering, not an interactive navigation policy evaluation.

## Mechanism, training, and inference

VSI-Bench derives questions from ScanNet, ScanNet++, and ARKit scene reconstructions and videos. Most questions use 3-D annotations and templates, with human route-planning annotations and iterative human filtering. Four tasks require numeric answers (object count, absolute distance, object size, room size); four use multiple-choice answers (relative distance, relative direction, route planning, appearance order). Numeric scoring averages relative-error thresholds from 0.5 through 0.95; multiple-choice scoring extracts answer letters/text. Fifteen pretrained models are evaluated zero-shot with greedy decoding. Additional controlled prompt experiments use CoT, tree-style reasoning, and five-sample self-consistency. Grid-map prompting asks models to place objects in a 10 by 10 or 20 by 20 map; supplied ground-truth maps separate perception/memory errors from map-use errors.

SUPPLEMENT CHECK: pp. 1–3 select 150 ARKitScenes, 50 ScanNet++ and 88 ScanNet videos; suppress rare/small objects and ambiguous geometry; and construct route-plan questions as filling turn directions in human-specified routes. Post-answer self-explanations are elicited after a prediction, unlike pre-answer CoT. Error attribution covers 163 selected errors with fractional assignment to overlapping categories; this is an interpretive diagnostic, not a causal intervention on internal computation. Cognitive maps use a 10 by 10 grid; all category names are supplied for map visualization, while the relative-distance intervention supplies only question-mentioned categories.

## Novelty and closest predecessors

Provides a video spatial reasoning benchmark and tests whether explicit language reasoning or spatial representations improve answers. The map intervention is stronger evidence about usable spatial information than an attention picture, but the generated map is an external prediction rather than direct measurement of a model’s latent world state.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 full benchmark Gemini-1.5-Pro 45.4 and GPT-4o 34.0, compared with a frequent-answer baseline 34.0; seven of twelve open models fall below that baseline. Human 79.2 is measured on a 400-question subset (50 per task), where Gemini scores 48.8, so human/full-set comparisons must preserve denominators. On 163 wrong subset answers, a self-explanation-based audit attributes 71% to spatial issues; this categorization is not a causal latent-mechanism test. CoT/tree prompting decreases mean performance by roughly four points in tested configurations and self-consistency by 1.1, without establishing that language reasoning always harms navigation. Table 3 relative-distance accuracy 46 baseline→56 predicted 10-grid→66 ground-truth 10-grid; a ground-truth 20-grid reaches 78 while predicted 20-grid is 54. Map quality, resolution, and downstream reasoning therefore interact.

SUPPLEMENT CHECK: Table 6 reports relative-distance accuracy 36→42 for LLaVA-Video-72B and 40→32 for 7B with cognitive maps; its nearby “8%” description does not match the first row difference. Table 8 gives video-first 48.8 versus question-first 46.3, despite a reversed sentence in the prose. Tables, not the conflicting prose, determine those comparisons. Humans can replay videos without a time limit; a second 560-question human subset was selected to match model full-set performance. Open models receive 32 frames, GPT-4o 16, Gemini native video (Table 9). These protocol differences and selection procedures constrain cross-model and human comparisons. None is an executed navigation result.

## Implementation and reproducibility

INSPECTED CODE: lmms_eval/tasks/vsibench/utils.py:48–180 resolves scene videos, builds numeric or multiple-choice prompts, extracts the first response word, computes case-insensitive exact-match for multiple choice and mean relative-accuracy thresholds for numeric answers, averages each task, averages direction difficulty bins, then averages task scores. This is an eight-task macro-average, not navigation success or accuracy over one pooled item list. The repository contains a modified lmms-eval harness and model adapters, not a new model-training algorithm. README warns the released dataset was refined and scores may differ from the paper. Pin both dataset revision and model/prompt/video-sampling settings for comparisons. Models, videos and API inference were not executed.

Inspected release reference: [https://github.com/vision-x-nyu/thinking-in-space at 51e089c3ae69b9435e9489058610f5b3964c56a8](https://github.com/vision-x-nyu/thinking-in-space/tree/51e089c3ae69b9435e9489058610f5b3964c56a8). File-level inspection is restricted to the paths listed below.

- README.md
- lmms_eval/tasks/vsibench/utils.py:48–180

## Specific bottlenecks

**vsi-spatial: Generated spatial representations are inaccurate enough to limit the benefit of explicit map reasoning.**

Conditions: Distant object pairs, high-resolution maps, and geometric questions beyond local adjacency. Cause: Perception, temporal memory, and spatial reconstruction errors enter the map before reasoning. Evidence: PAPER EXPERIMENT: Table 3 and local-versus-distant map analysis.

Type: methodological. Status: Partially addressed. Confidence: High within the specified evidence scope. Follow-ups: ['OmniAgent']

**vsi-transfer: Offline spatial QA accuracy does not establish closed-loop navigation reliability.**

Conditions: A policy must select views, execute movements, and recover from observation or action errors. Cause: The benchmark supplies a fixed video and evaluates an answer rather than interactive control. Evidence: DEDUCTION from benchmark protocol.

Type: insufficiently tested. Status: Uncertain because evidence is insufficient. Confidence: High within the specified evidence scope. Follow-ups: ['EmbodiedQA', 'IQA', 'VLN-MME']

## Sources and limitations

- [Supplied main paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.html)
- [Official supplementary or complete author version; reviewed scope listed in associated material](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Yang_Thinking_in_Space_CVPR_2025_supplemental.pdf)
- Official 16-page supplement and evaluator source reviewed; figures interpreted through captions/text, not independent visual relabeling. Subset selection, frame budgets and changing dataset version constrain comparisons. No navigation benchmark executed.
- Associated material: {"file": "evidence/04/supplements/vsi_supp.txt", "pages_read": "1–16", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
