# Mind the Error

Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation

Francesco Taioli; Stefano Rosa; Alberto Castellini; Lorenzo Natale; Alessio Del Bue; Alessandro Farinelli; Marco Cristani; Yiming Wang. IROS 2024; official author repository identifies the accepted paper.. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [Mind the Error.pdf](../../Paper/Mind%20the%20Error.pdf). Pages reviewed: ['2–7'].

## Problem and operating conditions

Detect whether an instruction contains an erroneous direction, room, or object reference, and localize the suspect words after a navigator has executed a trajectory. This is a diagnostic task with a fixed BEVBert navigator, rather than an online language repair policy.

## Mechanism, training, and inference

R2RIE-CE creates paired original/corrupted instructions using direction antonyms and room/object substitutions, with ConceptNet and manual constraints for more plausible replacements. Instruction BERT embeddings use up to 80 tokens of width 768. CLIP visual features from selected trajectory views, positional embeddings, and a CLS token feed four layers of cross-attention with trajectory queries and instruction keys/values, followed by self-attention and feed-forward blocks. Binary error classification uses the CLS output; a separate token head predicts error positions. Training combines binary error cross-entropy and token localization cross-entropy. The 9,000-step AdamW run trains on all error categories without commonsense filtering; the reported checkpoint is selected by validation-unseen AUC on the commonsense all-error set. Inference consumes the completed trajectory, not just the current view.

## Novelty and closest predecessors

Introduces instruction-error detection and localization as distinct from instruction underspecification in ULN and language-following diagnostics in NavNuances. It operationalizes instruction/trajectory inconsistency, but does not identify whether an inconsistency was caused by the instruction or by the fixed navigator.

## Results and independent verdict

PAPER EXPERIMENT: Table II reports AUC/average token distance of 0.58/8.13 for directions, 0.80/7.73 for rooms, 0.74/9.21 for objects, 0.91/7.34 for room plus object, and 0.94/6.14 for all errors; mean 0.79/7.71. Object localization is worse than the CLIP baseline token distance 8.76. The large all-error AUC therefore should not be presented as accurate single-word repair or strong directional-error recognition. On natural R2R-CE annotations, 25 of 1,839 instructions are flagged and eight confirmed; on RxR-CE, 118 of 3,669 are flagged and ten confirmed, including mesh-related issues. These selected-case confirmations are not an estimated corpus-wide error rate. The causal ambiguity between language error and policy error remains unresolved.

## Implementation and reproducibility

INSPECTED CODE: The BEV-based trajectory matcher uses a learned trajectory CLS token, positional encoding and crossmodal encoder, with binary detection and an 80-position localization head (vlnce_baselines/models/bev/vilmodel_cmt.py:687–742). ss_trainer_BEV.py:1755–1789 explicitly says navigation imitation loss is not used to fine-tune the navigator in this phase, and combines binary cross-entropy with per-error token cross-entropy. Ground-truth error sensors supply supervision; evaluation uses sigmoid/ROC-AUC and top-k token predictions. The tokenizer maximum and trajectory representation constrain the scope of localization; this is a trajectory/instruction inconsistency detector, not a causal proof that the language rather than behavior is wrong. README pins Habitat0.1.7 and links model/data resources. Source was inspected, not imported or executed; no detector quality was independently reproduced.

Inspected release reference: [https://github.com/intelligolabs/R2RIE-CE at d330a55a717f4415f34541f8d8514c71c0716804](https://github.com/intelligolabs/R2RIE-CE/tree/d330a55a717f4415f34541f8d8514c71c0716804). File-level inspection is restricted to the paths listed below.

- README.md
- vlnce_baselines/models/bev/vilmodel_cmt.py:687–742
- vlnce_baselines/ss_trainer_BEV.py:1755–1789; inspected optimizer/evaluation references at283–318,875–889,1067–1073,1631–1661

## Specific bottlenecks

**iedl-cause: A trajectory-language mismatch does not identify which input or action is wrong.**

Conditions: The navigator takes a wrong turn while the instruction is correct, or several substitutions leave the route semantically valid. Cause: The detector conditions on one completed policy trajectory rather than an independent ground-truth environmental explanation. Evidence: DEDUCTION from architecture; PAPER EXPERIMENT: weak directional AUC in Table II.

Type: methodological. Status: Still unresolved in the reviewed literature. Confidence: High within the specified evidence scope. Follow-ups: ['NavNuances', 'ULN']

**iedl-online: Detection does not provide an online repair policy.**

Conditions: A mistaken instruction must be diagnosed before a long failed rollout. Cause: The model aggregates the completed trajectory and predicts labels, with no tested action-selection or instruction-editing loop. Evidence: METHOD: architecture and evaluation protocol.

Type: methodological. Status: Uncertain because evidence is insufficient. Confidence: High within the specified evidence scope. Follow-ups: ['VLN-Trans']

## Sources and limitations

- [Official project](https://intelligolabs.github.io/R2RIE-CE/)
- No separate supplement was located through exact-title search and the pinned official tree; no PDF appendix was present there. Synthetic corruption is an operational label, not proof that every modified instruction is semantically false. No benchmark executed.
