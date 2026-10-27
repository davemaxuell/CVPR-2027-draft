# M-Track

One Step at a Time: Long-Horizon Vision-and-Language Navigation with Milestones

Chan Hee Song; Jihyung Kil; Tai-Yu Pan; Brian M. Sadler; Wei-Lun Chao; Yu Su. CVPR 2022. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [M-Track.pdf](../../Paper/M-Track.pdf). Pages reviewed: ['3–8'].

## Problem and operating conditions

Long ALFRED household tasks fail when an agent skips a subtask or interacts with the wrong object. M-Track explicitly tracks completion of supplied low-level instructions, in addition to a high-level goal.

## Mechanism, training, and inference

A BERT-CRF trained on simulator-derived labels extracts milestone type and object targets. A Mask R-CNN predicts 116 object classes, masks and binary reachability within 1.5 m. WordNet WUP similarity grounds target names to reachable detections. Navigation milestones require a reachable target; interaction milestones additionally require the action mask to overlap it above IoU 0.5. A checklist tracks multiple target objects. The base LSTM or recurrent VLN-BERT sees the high-level goal and only the current low-level sentence, advancing after milestone completion. Before executing an interaction, proactive checking scans the top five predicted actions for a matching target, otherwise chooses navigation. Training combines action imitation, object selection and A2C rewards for distance, interaction correctness, visibility and final success. The VLN-BERT adaptation uses eight panoramic views and object features plus a pointer action head.

## Novelty and closest predecessors

Explicit grounded progress checks and proactive action rejection, rather than implicit attention or a learned binary switch. It is model-agnostic but the evaluated checker assumes object-centered ALFRED milestones and provided instruction decomposition.

## Results and independent verdict

PAPER EXPERIMENT: Table 2 unseen LSTM-L SR/PLWSR 8.70/4.05→13.28/6.25; VLN-BERT-L 12.23/5.60→16.29/7.66. Table 3 applying M-Track only at test improves validation SR 10.35→13.29, while training and testing with it reaches 17.29. Table 4 ground-truth milestones reach 24.38, leaving extraction/grounding room. §6 explicitly states procedural instructions such as turning twice need action-history checks, implicit milestones require inference, and multiple valid subtask orders call for softer constraints. These limitations are direct prior-art statements for any proposed obligation/history tracker.

## Implementation and reproducibility

The main paper was reviewed by assignment 04; the completed independent runtime audit by assignment 05 is incorporated below.

INSPECTED CODE BY ASSIGNMENT 05: Official M-Track commit a7eaf93fc8631c9d1e7bbef437cc858b5b60f36a is materially incomplete. milestone_checker.py:7–30 implements substring matching and reachable flags, with no caller outside its own example. The inspected runtime does not establish paper-equivalent WordNet matching, interaction IoU, persistent checklist or proactive top-five rejection. agent.py:102–104 retains a target-tagger TODO; launcher files point at missing r2r_src/train.py. Assignment 05’s own AST-only test finds a syntax error in env.py line 53 (evidence/05/mtrack_static_results.json); no fetched code was imported. CRF tagging, reachability BCE and CE rollout components are present. See evidence/05/code_audit.md for complete source ranges. This is an independently attributed static audit, not an executed navigation result.

Inspected release reference: [https://github.com/chanhee-luke/M-Track at a7eaf93fc8631c9d1e7bbef437cc858b5b60f36a](https://github.com/chanhee-luke/M-Track/tree/a7eaf93fc8631c9d1e7bbef437cc858b5b60f36a). File-level inspection is restricted to the paths listed below.

- Assignment 05: mtrack_src/milestone_checker.py:1–30; agent.py:96–115,236–345; env.py:53,106–165; train.py:1–90,250–268; eval.py:17–89
- Assignment 05: milestone.py:18–95; milestone/src/model/transformer_crf.py:14–43; vision/alfred_head.py:31–123; vnn.py:156–203; run launchers

## Specific bottlenecks

**mtrack-procedure: Object reachability and interaction checks do not certify procedural action-history requirements.**

Conditions: Turn twice, preserve order, or satisfy an implicit prerequisite that is not an object-state condition. Cause: Current milestones are type/target tuples and the checker primarily observes current objects. Evidence: AUTHOR ACKNOWLEDGEMENT: §6.

Type: methodological. Status: Still unresolved in the reviewed literature. Confidence: High within the stated experimental scope. Follow-ups: ['NavNuances', 'Landmark-RxR']

**mtrack-hard: Hard sequential milestones can block valid alternative task orders or propagate a wrong target extraction.**

Conditions: Multiple valid plans or implicit/ambiguous instructions. Cause: Only the current sentence is exposed and all target checks must complete before advancing. Evidence: AUTHOR ACKNOWLEDGEMENT: §6; PAPER EXPERIMENT: Table 4 GT gap.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['FGR2R', 'BabyWalk']

## Sources and limitations

- [Author full version](https://arxiv.org/abs/2202.07028)
- [Official implementation](https://github.com/chanhee-luke/M-Track)
- Official runtime is materially incomplete and has compile/launcher defects; the independently inspected files do not establish the paper-equivalent WordNet/IoU/top-five milestone checker. No project execution.
