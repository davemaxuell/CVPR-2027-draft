# EmbodiedQA

Embodied Question Answering

Abhishek Das; Samyak Datta; Georgia Gkioxari; Stefan Lee; Devi Parikh; Dhruv Batra. CVPR 2018. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [EmbodiedQA.pdf](../../Paper/EmbodiedQA.pdf). Pages reviewed: ['3–8'].

## Problem and operating conditions

An agent must move through a previously unseen House3D home to obtain visual evidence and answer a question. EQA v1 deliberately restricts questions to one target object, enabling shortest-path demonstrations. It is active visual QA, not natural-language route following.

## Mechanism, training, and inference

EQA v1 filters SUNCG scenes and generates functional-program questions, excluding answer distributions with normalized entropy below 0.5 and questions seen in fewer than four environments. It has 4,246/506/529 train/validation/test questions in disjoint 643/67/57 homes. Four evaluated types are location, color, color-in-room, and spatial preposition. A four-block CNN is pretrained for RGB reconstruction, semantic segmentation, and depth. Separate two-layer 128-dimensional question LSTMs serve navigation and answering. PACMAN uses a recurrent planner choosing forward/left/right/stop and an MLP controller repeating an action up to five times before returning control. Actions move at most 0.25 m or turn 9 degrees. The answer head attends over only the last five frames and predicts one of 172 answers. Supervised navigation cross-entropy trains 15 epochs under a distance curriculum; QA trains 50 epochs. REINFORCE then fine-tunes the navigator with the answerer frozen, using a terminal correct-answer reward of 5 and forward-distance shaping coefficient 0.005. Evaluation starts 10, 30, or 50 actions before the goal and caps episodes at 100 actions.

SUPPLEMENT CHECK: pp. 1–2 describe functional-program question generation, uniqueness constraints and entropy/count filtering; nine generator templates do not mean nine types were used in every main experiment. Page 3 pretrains visual features with semantic segmentation, depth and reconstruction, using Adam at 1e-3, batch 20 and five epochs over 100,000 images. The QA component attends to the final five frames. Page 4 contrasts human versus shortest-path views: worse answer rank on human routes is evidence of view-distribution mismatch, not worse human navigation ability.

## Novelty and closest predecessors

Introduces embodied active QA and a planner/controller temporal hierarchy, combining existing CNN/RNN, imitation, and policy-gradient ingredients. The main distinction from passive VQA is control over observations; unlike IQA, object manipulation is not required.

## Results and independent verdict

PAPER EXPERIMENT: Table 1 at T−30, imitation PACMAN ends 1.50 m from target with QA mean rank 3.13; RL ends 2.19 m away with improved mean rank 2.99 and stops in only 32% of episodes. Authors explicitly observe overshooting. The shortest-path QA oracle has mean rank 3.21, showing the closest endpoint is not necessarily the most informative view. The question-conditioned LSTM ends at 2.23 m, supporting the hierarchy for navigation, but the small QA mean-rank changes should not be presented as high accuracy. Human navigation uses a subset and a different answer output, so there is no directly comparable human mean rank.

## Implementation and reproducibility

INSPECTED CODE: training/models.py:1100–1173 runs a question-conditioned recurrent planner and gathers its hidden state for the controller, which takes image, action embedding and planner state; planner_step/controller_step separate inference calls. train_eqa.py:683–725 evaluates the last five images with the answerer, appends answer-accuracy reward, discounts returns by 0.99, subtracts a running reward baseline and forms planner/controller policy losses. README warns the released shortest paths were updated because the earlier grid paths were inaccurate, so data version must be fixed for replication. The archived repository requires the House3D/SUNCG assets and older PyTorch APIs. None of these external assets or training loops was executed; current physical-world reproducibility is not established by a code snapshot.

Inspected release reference: [https://github.com/facebookresearch/EmbodiedQA at 306fd6ef3064b5b9a3bd2d3f06ac96a09c481d9d](https://github.com/facebookresearch/EmbodiedQA/tree/306fd6ef3064b5b9a3bd2d3f06ac96a09c481d9d). File-level inspection is restricted to the paths listed below.

- README.md
- training/models.py:1100–1173
- training/train_eqa.py:683–725

## Specific bottlenecks

**eqa-stop: Answer-seeking reinforcement can improve visual evidence while overshooting the target and failing to stop.**

Conditions: A useful object view is encountered before the final movement decision. Cause: The navigator is rewarded for answer correctness and progress while the answerer can attend to recent frames; terminal location and evidence quality differ. Evidence: AUTHOR ACKNOWLEDGEMENT and PAPER EXPERIMENT: Table 1 and §5.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['Explore until Confident', 'IQA']

**eqa-memory: Five-frame answering context cannot preserve arbitrary evidence collected much earlier.**

Conditions: Questions requiring distant-object comparison or long search. Cause: The answerer discards all but the last five frames; evaluated v1 questions have one target. Evidence: METHOD: §4; broader question types are proposed rather than evaluated.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['IQA', 'VSI-Bench', 'OmniAgent']

## Sources and limitations

- [Supplied main paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Das_Embodied_Question_Answering_CVPR_2018_paper.html)
- [Official project](https://embodiedqa.org/)
- [Official supplementary or complete author version; reviewed scope listed in associated material](https://openaccess.thecvf.com/content_cvpr_2018/Supplemental/0052-supp.pdf)
- Official supplement and archived source reviewed. House3D/SUNCG assets and checkpoints were not downloaded or initialized; current asset access was not tested. No navigation or QA benchmark executed.
- Associated material: {"file": "evidence/04/supplements/embodiedqa_supp.txt", "pages_read": "1–10", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
