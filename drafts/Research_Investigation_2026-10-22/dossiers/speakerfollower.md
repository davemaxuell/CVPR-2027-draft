# Speaker-Follower

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Speaker-Follower Models for Vision-and-Language Navigation**. Daniel Fried, Ronghang Hu, Volkan Cirik, Anna Rohrbach, Jacob Andreas, Louis-Philippe Morency, Taylor Berg-Kirkpatrick, Kate Saenko, Dan Klein, Trevor Darrell. NeurIPS 2018.

Primary paper: [Speaker-Follower Models for Vision-and-Language Navigation](https://proceedings.neurips.cc/paper_files/paper/2018/file/6a81681a7af700c6385d36577ebec359-Paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: Speaker-Follower.pdf (Main PDF; 12 pages; SHA256 bc6b4a9ef46cc71797a890497c60c5089e30bdba4ce59f26af4b5a5f0ef16e5b).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Human navigation instructions are scarce and an instruction-conditioned follower may choose routes whose visual content does not explain the instruction. A learned reverse-direction speaker supplies both additional training instructions and a route-compatibility score.

## 3. Mechanism, training, and inference

PDF pp 3–6: train an LSTM speaker PS(d|r) on human route–description pairs; generate descriptions for 178k sampled shortest routes in training environments; train a follower PF(r|d) on augmented data then fine-tune on human data. The follower attends over language and 36-view panoramic features and selects neighboring viewpoints or STOP. A state-factored search collects 40 completed candidate routes. Final selection is argmaxr PS(d|r)^λ PF(r|d)^(1−λ), equivalently λlogPS+(1−λ)logPF; λ is near 1. The inspected code evaluates 0 and 0.95 weights and can standardize scores. State-factored search merges candidates by physical state despite history-dependent LSTM states; therefore it is approximate. The physical traversal needed to obtain candidate observations must count in trajectory length.

Section 4.1, PDF pp 6–7, specifies student forcing: sample follower actions and supervise the shortest-path action from the resulting state. Speaker maximum-likelihood training uses token cross entropy. The follower trains for 50k iterations on generated instructions and then 20k on human instructions; image ResNet features are fixed. The reported inference mixture uses λ=0.95.

## 4. Novelty and evidence

The same reverse instruction-generation model is used for both data augmentation and pragmatic inference. Panorama-level neighbor actions also change the action abstraction relative to the original low-level R2R baseline. These are separable sources of improvement, so the method is not merely a speaker loss added to an otherwise identical baseline.

## 5. Results and independent verdict

PAPER EXPERIMENT: Table 1, PDF p 7, gives validation-unseen SR 19.9 for the baseline, 24.6 after speaker data augmentation, 39.3 with augmentation plus pragmatic inference, and 54.6 for all three components including panoramic actions. The same candidate-search comparison gives 30.5 SR using follower scores and 39.3 using the speaker, which isolates reranking more directly than the full system comparison. Table 2, p 8, reports test SR 53.5 for both the clean selected route and the challenge submission, but trajectory length changes from 11.63m to 1257.38m when physical search traversal is included; OSR changes from 63.9 to 96.0. These are separate resource protocols. VERDICT: augmentation, pragmatic reranking and action abstraction each contribute. Speaker likelihood is learned compatibility rather than semantic entailment or calibrated completion confidence, and search gains carry substantial movement cost.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/ronghanghu/speaker_follower](https://github.com/ronghanghu/speaker_follower); inspected snapshot **e4dcdf923ecacf7893397bd9821377e71e32a75a**. Snapshot metadata and source are under `repositories/01/speakerfollower`.

Inspected follower rollout/CE, speaker token likelihood, augmentation entrypoint, and rational reranking. The reranker explicitly constructs physically traversed trajectories when configured, so clean candidate paths and executed exploration paths must not be conflated.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [tasks/R2R/speaker.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/speaker.py) — lines 179–182: Speaker score sums token log likelihoods and trains with token NLL.
- [tasks/R2R/follower.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/follower.py) — rollout lines 430–538; state-factored search: Follower trains on action CE and maintains recurrent state during rollout/search.
- [tasks/R2R/rational_follower.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/rational_follower.py) — lines 85–106, 131–150: Physical path reconstruction is distinct from weighted speaker/follower candidate reranking; tested speaker weights include0.95.
- [tasks/R2R/data_augmentation_from_speaker.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/data_augmentation_from_speaker.py) — entry_point lines 35–83: Speaker-generated and scored candidates form augmentation output.
- [tasks/R2R/eval.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/eval.py) — 56–138: Scores endpoint/oracle distances and full trajectory length; candidate reranking and search cost should not be collapsed into greedy performance.
- [tasks/R2R/train.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/train.py) — 68–138;174–233: Training driver configures data augmentation, training split, vocabulary and validation separately.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **speakerfollower-B1 — Partially addressed**. A high reverse-language likelihood can favor common descriptions without establishing exact route semantics; same-scene synthetic routes do not add environmental diversity. Evidence: Speaker likelihood objective and training-scene augmentation §3. Type: data/methodological. Follow-ups: ScaleVLN, SRDF, AirBERT. Confidence: medium.
- **speakerfollower-B2 — Partially addressed**. Candidate search observes and traverses extra paths, creating a distinct resource protocol. Evidence: §3 search; rational_follower physical trajectory reconstruction. Type: inference cost. Follow-ups: FAST, DUET. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/Kelym/FAST; https://github.com/wz0919/VLN-SRDF
