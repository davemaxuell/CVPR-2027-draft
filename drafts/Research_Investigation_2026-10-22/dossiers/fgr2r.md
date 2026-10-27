# FGR2R

Sub-Instruction Aware Vision-and-Language Navigation

Yicong Hong; Cristian Rodriguez-Opazo; Qi Wu; Stephen Gould. EMNLP 2020. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [FGR2R.pdf](../../Paper/FGR2R.pdf). Pages reviewed: ['2–9', '11–17'].

## Problem and operating conditions

R2R endpoint success can conceal failure to execute intermediate instructions. The paper introduces human sub-instruction–subtrajectory alignment and a recurrent controller that explicitly tracks a single instruction chunk. The environment is the discretized Matterport3D graph; instructions and route segments are assumed to be monotonically ordered.

## Mechanism, training, and inference

A Stanford dependency-parser heuristic splits instructions at roots, conjunctions, and parataxis, then merges short or action-only fragments (Algorithm 1). The full instruction is encoded by an LSTM before hidden states are partitioned, preserving earlier linguistic context. A chunk-specific attention replaces whole-instruction attention. An LSTM-based shifting head combines the current state, memory, selected-action visual feature, attended chunk, and one-hot count of remaining chunks (Eqs. 4–6). It can advance exactly one chunk and cannot return to an earlier chunk. Action cross-entropy and balanced binary shifting cross-entropy supervise training; student forcing supplies action rollouts while chunk shifts use teacher forcing. Ground-truth shift is positive within 0.5 m of the annotated segment endpoint. The Back-Translation variant applies shifting loss only to its imitation branch. Inference uses predicted shifts and actions. FGR2R adds an average of 3.6 chunks per instruction and 2.4 viewpoints per chunk. Fifty-eight qualified annotators survived a 126-person qualification process. Single-viewpoint chunks are merged during training but retained in validation (Appendix A.1). Models use the original baseline hyperparameters and one Tesla K80.

## Novelty and closest predecessors

Relative to soft whole-instruction attention and progress monitors, the contribution is explicit chunk selection, supervised transitions, and traceable intermediate evaluation. Concurrent BabyWalk uses automatically aligned chunks and curriculum learning, whereas FGR2R uses human alignment. The method is not an instruction-error detector or backward recovery controller.

## Results and independent verdict

PAPER EXPERIMENT: Table 1, PDF p. 7, reports unseen Speaker-Follower SR 29.8→34.7 and nDTW 0.54→0.59; Self-Monitoring SR 40.7→42.4. Back-Translation barely changes, SR 46.6→46.7, while nDTW falls 0.70→0.69. Table 2 shows best shifting F1 only 0.331 and recall 0.209 despite precision 0.806. Thus explicit alignment helps some agents, but correct timing remains weak. Table 3 finds greater endpoint errors for object-specific or condition-dependent chunks; this is descriptive clustering, not a causal isolation of language difficulty. Appendix Table 4 shows R4R Back-Translation SR 31.2→21.5 despite nDTW 0.39→0.48: concatenation leaves bridge viewpoints without instructions and introduces intermediate STOP language. Figure 10 provides a negative case in which perfect chunk shifting still fails object grounding. These results separate progress tracking, grounding, and endpoint success.

## Implementation and reproducibility

INSPECTED CODE: The official pinned release is a dataset and chunker release. source/make_subinstr.py invokes StanfordNLP, processes the three R2R instructions, and writes new_instructions; source/chunking_function.py implements dependency-root/conjunction/parataxis rules with special handling of short turning clauses. README distinguishes manually annotated chunk_view path indices (one-based), potentially singleton rotation/STOP chunks, and test instructions without sub-path labels. Its complete source tree contains the chunker and utilities, not the paper navigator, shift predictor, training objective or evaluator. Those components cannot be verified from this release; their dossier descriptions are paper evidence. No NLP model assets or annotated dataset were executed.

Inspected release reference: [https://github.com/YicongHong/Fine-Grained-R2R at 305ed4cd219d79306bc38650899edc14cfdcfb69](https://github.com/YicongHong/Fine-Grained-R2R/tree/305ed4cd219d79306bc38650899edc14cfdcfb69). File-level inspection is restricted to the paths listed below.

- README.md
- source/make_subinstr.py:1–41
- source/chunking_function.py:1–89

## Specific bottlenecks

**fgr2r-transition: Chunk completion has low recall and cannot reverse an erroneous advance.**

Conditions: Agent deviates from its annotated subpath or must revisit an earlier instruction after backtracking. Cause: Training supplies teacher chunk transitions, whereas inference predicts them; the state machine allows only one forward transition. Evidence: PAPER EXPERIMENT: Table 2 best F1 0.331; AUTHOR ACKNOWLEDGEMENT: Appendix A.1 leaves backward/multi-step shifts to future work.

Type: methodological. Status: Partially addressed. Confidence: High within the reported discretized R2R protocol. Follow-ups: ['BabyWalk', 'Landmark-RxR', 'GELA', 'VLN-Trans', 'M-TRACK']

## Sources and limitations

- [Official proceedings metadata and paper](https://aclanthology.org/2020.emnlp-main.271/)
- [Author-linked release](https://github.com/YicongHong/Fine-Grained-R2R)
- The pinned official release contains dataset/chunking utilities but no paper-equivalent navigator source. No benchmark executed.
- Associated material: Appendix A and qualitative examples embedded in supplied PDF, pp. 11–17
