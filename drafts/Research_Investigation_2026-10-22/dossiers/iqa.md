# IQA

IQA: Visual Question Answering in Interactive Environments

Daniel Gordon; Aniruddha Kembhavi; Mohammad Rastegari; Joseph Redmon; Dieter Fox; Ali Farhadi. CVPR 2018. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [IQA.pdf](../../Paper/IQA.pdf). Pages reviewed: ['2–8'].

## Problem and operating conditions

Answer existence, counting, and spatial-relation questions by navigating and opening/closing containers in AI2-THOR. The floor is discretized, localization is supplied, and objects must be visible and close enough for interaction. IQUAD has 76,800 training and 1,920 test configurations across 25 and five kitchen rooms.

## Mechanism, training, and inference

HIMN factorizes a high-level GRU planner and low-level Navigator, Scanner, Detector, Manipulator, and Answerer. A 32-action high-level policy selects relative waypoints in a 5×5 grid, view rotations, interactions, or answer. Semantic spatial memory stores detection probabilities, occupancy, coverage, and visit intent. An egocentric spatial GRU updates only the current local window in a global tensor; other locations persist. A* replans over predicted occupancy while a supervised navigation head predicts feasibility/termination. YOLOv3 fine-tuned on training scenes and a depth network project objects into the map; scanning and manipulation controllers have fixed environment-defined behavior. The answerer convolves the full map with a tiled question vector, pools, and predicts the answer. Separate pretraining uses oracle planners/detections/partial maps where appropriate, followed by joint learning. A3C trains the planner with answer, exploration coverage, invalid-action and time rewards; an auxiliary supervised loss predicts validity of all actions.

## Novelty and closest predecessors

Extends embodied QA to actions that change visibility and object state, and introduces local writes into persistent global semantic memory. Memory and control factorization are central; there is no VLM backbone or zero-training claim.

## Results and independent verdict

PAPER EXPERIMENT: Table 2 HIMN with YOLO reaches 68.47% existence, 30.43% counting, 58.67% spatial accuracy versus chance 50/25/50. Episodes average 318/926/516 actions, much longer than the small human sample. Table 3 oracle detection raises existence to 86.56 and spatial to 70.94; oracle navigation further raises counting to 48.44. Removing question input to the planner or the auxiliary validity loss returns performance near chance. Thus semantics and affordance supervision both matter. Table 5 explicitly separates novel arrangements in seen rooms from five unseen rooms. §5.5 acknowledges that the 2-D map cannot distinguish inside from on top and exploration revisits are inefficient.

## Implementation and reproducibility

INSPECTED CODE: networks/qa_planner_network.py constructs question-conditioned memory crops and answer/planning heads. answer_loss:220–243 uses sigmoid cross-entropy for existence/spatial answers and softmax cross-entropy for counting; move_loss:245–252 adds a ten-weighted sigmoid loss for all possible moves when enabled. This confirms auxiliary action-validity supervision in addition to question answering and RL, not an instruction-language-only learner. The release includes TensorFlow/A3C training, supervised map/navigation training, AI2-THOR interaction, question generation and evaluation entry points. Their existence is not a claim each simulator transition was audited. No simulator, training or evaluation was executed; legacy THOR/vision-model assets are prerequisites.

Inspected release reference: [https://github.com/danielgordon10/thor-iqa-cvpr-2018 at 38358569422a93fa42a782f3022554672507a08a](https://github.com/danielgordon10/thor-iqa-cvpr-2018/tree/38358569422a93fa42a782f3022554672507a08a). File-level inspection is restricted to the paths listed below.

- networks/qa_planner_network.py:50–79,114–189,220–253
- Pinned tree: training/evaluation and question-generation entry points

## Specific bottlenecks

**iqa-height: A 2-D semantic map aliases vertical containment and support relations.**

Conditions: Objects inside versus above the same container footprint. Cause: Map projection removes height and explicit relations. Evidence: AUTHOR ACKNOWLEDGEMENT: §5.5.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['VSI-Bench', '3-D map methods']

**iqa-count: Accurate global counting requires reliable persistent detections and efficient coverage.**

Conditions: Long searches with repeated object observations and hidden containers. Cause: Perception errors and inefficient exploration accumulate before answering. Evidence: PAPER EXPERIMENT: Tables 2–3 counting and oracle gaps.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['Explore until Confident']

## Sources and limitations

- [Supplied main paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Gordon_IQA_Visual_Question_CVPR_2018_paper.html)
- Official CVPR page and complete arXiv v3 expose no separate supplement; main §4 and source provide the implementation details. AI2-THOR assets were not initialized, and no physical robot evidence is supplied by these experiments.
