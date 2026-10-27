# RxR

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding**. Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, Jason Baldridge. EMNLP 2020.

Primary paper: [Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding](https://arxiv.org/abs/2010.07954). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 9, 12, 19, 20, 21].

Associated files: RxR.pdf (Main PDF; 21 pages; SHA256 8a964d95106ff267f934acf705a6b0be304a193f84745fe0ad55f51133d61bd8).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

RxR reduces R2R shortest-path and length biases and provides multilingual dense grounding. It contains 126,069 spoken and transcribed instructions in English, Hindi and Telugu over 16,522 paths, plus guide and follower pose traces. It directly tests whether agents follow route constraints instead of only reaching the goal.

## 3. Mechanism, training, and inference

Sections 3–4, PDF pp 3–6: sample a simple sequence of rooms, then find the shortest panorama path constrained to that sequence. Greedy selection favors coverage and paths that are not globally shortest. In total, 44.5% of paths are not globally shortest; the average is 14.9m and 27.4% longer than the shortest route. Spoken instruction timestamps are aligned through ASR/transcript dynamic time warping to six-degree-of-freedom poses. Human followers validate instructions and supply alternative paths. Primary metrics nDTW and sDTW measure path adherence because endpoint success alone is insufficient. The baseline in Section 5, p 7 uses multilingual BERT word features, a CNN text encoder, an LSTM policy and EfficientNet-B4 image features. Training mixes 50% behavioral cloning and 50% policy-gradient rollouts with incremental nDTW reward. The grounding auxiliary loss is L(z, m)=log Σi exp(zi)−log Σi mi exp(zi), equal to the negative log of attention mass on observed features, where binary mi labels regions viewed by the human. An analogous mask applies to words. Training uses 100k steps, batch 32 and Adam at 1e−4. Inference uses the full instruction and panorama; human traces supply supervision rather than test inputs.

## 4. Novelty and evidence

Multilingual spoken instructions, dense pose alignment, varied paths that need not be shortest, and human alternative paths jointly strengthen the benchmark. The dataset offers supervision for language progress, but its own attention-supervision results are mixed. Annotation availability does not imply solved grounding.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 4, p 6: following the correct first step and then going straight achieves 27.2% SR on R2R versus 13.7% on RxR, supporting reduced shortcut bias. Table 5, p 8: combining guide and follower paths improves monolingual training; joint multilingual training and automatic translations can hurt. Visual attention supervision has mixed effects and textual attention masking did not improve performance. Table 6 shows weak direct transfer across datasets despite shared buildings. VERDICT: RxR is an essential control for route-faithfulness proposals. Report languages and guide/follower training conditions, and distinguish English subsets from the full multilingual benchmark.

Table 7, PDF p 9, trains/evaluates unimodal ablations rather than merely deleting a modality at test time: multimodal unseen SR 22.8 averages above the language-only and vision-only models; direction words can still guide actions using graph geometry without visual content. Appendix A/p 12 explains 247 annotators, separate Guide/Follower workers, pilot feedback and retraining below 80% success. The datasheet pp 19–21 confirms building-level splits inherited from MP3D/R2R and external simulator dependence. This collection protocol supports quality but does not establish immunity to all annotation or graph biases.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/google-research-datasets/RxR](https://github.com/google-research-datasets/RxR); inspected snapshot **d22df5940a0486a42b703cd445937210ce21e9f5**. Snapshot metadata and source are under `repositories/01/rxr`.

The official data repository is pinned. The release README, formats and available assets were inspected. The paper implements its baseline in VALAN; inspecting the data repository does not establish inspection of VALAN model code. Dataset alignment and metric definitions are direct primary evidence.

Official dataset documentation and schemas were inspected at the pinned RxR commit. VALAN is a related public framework, but its README lists R2R/CVDN/Touchdown rather than verifying the exact RxR multilingual baseline release; this assignment does not mislabel it as an inspected RxR implementation. The paper’s original annotation tool release was prospective; a current primary-source search locates google-research/pangea as the released Guide/Follower collection toolkit, but that toolkit was not executed.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [README.md](https://github.com/google-research-datasets/RxR/blob/d22df5940a0486a42b703cd445937210ce21e9f5/README.md) — Guide/Follower schemas; Pose Traces; Text Features; Changelog: Data release documents dense alignments, missing word timestamps in a small subset, and seven corrected training pose-trace archives. This is data-interface inspection, not baseline model execution.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **rxr-B1 — Partially addressed**. Dense human alignment is partial and imperfect; simple attention constraints do not consistently improve policy performance. Evidence: Section 5 grounding-supervision experiments and Table 5. Type: supervision. Follow-ups: GELA language alignment, MARVAL, SkillNav. Confidence: high.
- **rxr-B2 — Partially addressed**. Multilingual and cross-dataset distribution differences persist despite more data and shared scenes. Evidence: Tables 5–6. Type: generalization. Follow-ups: MARVAL multilingual Marky data, NavRAG diversification. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/google-research-datasets/RxR; https://github.com/google-research/valan
