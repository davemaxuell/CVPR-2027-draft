# Landmark-RxR

Landmark-RxR: Solving Vision-and-Language Navigation with Fine-Grained Alignment Supervision

Keji He; Yan Huang; Qi Wu; Jianhua Yang; Dong An; Shuanglin Sima; Liang Wang. NeurIPS 2021. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [Landmark-RxR.pdf](../../Paper/Landmark-RxR.pdf). Pages reviewed: ['1', '3–9'].

## Problem and operating conditions

Coarse instruction/trajectory pairs weakly supervise individual landmarks; trajectory metrics can be dominated by where a difficult segment occurs. Landmark-RxR supplies human sub-instruction and sub-trajectory boundaries and a diagnostic reset protocol.

## Mechanism, training, and inference

Thirty annotators spend about 2,700 hours producing 166,740 sub-instructions and 46,645 sub-trajectories from English RxR. Each chunk has independent meaning, a changed position and clear endpoint, with cross-annotator verification. Sub-pairs, recombined adjacent pairs, full English RxR and R2R train an RCM-style encoder-decoder with ResNet panoramas and GloVe-300. Student-forcing imitation runs 100K steps, then policy-gradient training 20K, batch 100, lr 1e-4. Soft focal reward weights DTW costs at two sampled annotated critical landmarks by ten before exponentiation; hard focal reward sums +1/−1 for whether any predicted point visits each selected landmark within 3 m. The latter is order-insensitive. Diagnostic evaluation allows ten steps per landmark, then resets the agent to a failed annotated landmark and gives only the unfinished instruction suffix. This is oracle intervention, not autonomous recovery.

SUPPLEMENT CHECK: pp. 1–3 specify an RCM-like recurrent baseline and chunk granularity controls. The annotation datasheet, pp. 5–8, describes 30 annotators and about 2,700 hours, with some original English RxR instruction errors corrected. Its claim contrasting FGR2R as heuristic should be read narrowly: FGR2R uses heuristic linguistic segmentation but its released chunk-to-viewpoint alignment includes human annotation. Human landmark alignment and oracle-reset evaluation remain distinct from autonomous recovery.

## Novelty and closest predecessors

Human alignment supervision and landmark-focused rewards complement global route similarity. The reset metrics isolate local competence and must be reported separately from ordinary complete-route navigation. Hard landmark reward precedes any proposal simply rewarding intermediate landmark coverage.

## Results and independent verdict

PAPER EXPERIMENT: Table 3 all-granularity imitation SR/nDTW 28.4/26.7; nDTW RL 29.3/43.2; soft focal 33.7/41.0; hard focal 34.9/36.6 on English RxR unseen. Thus SR and global alignment trade off. Table 2 adding fine-grained RxR pairs to R2R raises SR 33.0→40.1 while adding full pairs alone yields 33.1. Extra annotations, chunk lengths, and data volume all change; this is not a pure architecture comparison. Table 4 reset-based loss number improves 6,256→4,951 with hard focal reward, but those reset counts do not measure autonomous recovery capability. The report’s attention illustration is supportive qualitative evidence, not a causal grounding proof.

## Implementation and reproducibility

Pinned official repository contains README and data links, with a supplementary PDF link; no model, optimizer, inference or evaluation source files are present in the inspected tree. Data-release provenance is established, but the paper’s code-availability statement is not fulfilled by model source at this inspected commit. The official nine-page supplement was reviewed, including model details and the annotation datasheet.

Inspected release reference: [https://github.com/hekj/Landmark-RxR at 6c9c6f5a0313f7c799c5a55538c9540829469031](https://github.com/hekj/Landmark-RxR/tree/6c9c6f5a0313f7c799c5a55538c9540829469031). File-level inspection is restricted to the paths listed below.


## Specific bottlenecks

**landmark-order: Landmark visitation alone does not enforce ordered or procedural instruction constraints.**

Conditions: A path visits the right objects in the wrong order or repeats an action incorrectly. Cause: Hard reward uses a minimum distance over all predicted points independently for each landmark. Evidence: DEDUCTION from Eqs. 3–4.

Type: methodological. Status: Partially addressed. Confidence: High within the stated experimental scope. Follow-ups: ['M-Track', 'NavNuances']

**landmark-reset: Reset-based local success does not quantify an agent’s ability to recover by itself.**

Conditions: A route fails before a later annotated segment. Cause: Evaluation supplies the correct landmark location and remaining instruction after failure. Evidence: METHOD: §4.

Type: insufficiently tested. Status: Uncertain because evidence is insufficient. Confidence: High within the stated experimental scope. Follow-ups: ['Meta-Explore', 'ULN']

## Sources and limitations

- [Official main paper](https://proceedings.neurips.cc/paper/2021/file/0602940f23884f782058efac46f64b0f-Paper.pdf)
- [Official dataset release](https://github.com/hekj/Landmark-RxR)
- [Official supplementary or complete author version; reviewed scope listed in associated material](https://proceedings.neurips.cc/paper/2021/file/0602940f23884f782058efac46f64b0f-Supplemental.pdf)
- Repository tree/README inspection is not model-code inspection; no released model source found at pinned commit.
- Associated material: {"file": "evidence/04/supplements/landmark_rxr_supp.txt", "pages_read": "1–9", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
