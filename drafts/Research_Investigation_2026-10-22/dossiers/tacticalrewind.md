# Tactical Rewind

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Tactical Rewind: Self-Correction via Backtracking in Vision-And-Language Navigation**. Liyiming Ke, Xiujun Li, Yonatan Bisk, Ari Holtzman, Zhe Gan, Jingjing Liu, Jianfeng Gao, Yejin Choi, Siddhartha Srinivasa. CVPR 2019.

Primary paper: [Tactical Rewind: Self-Correction via Backtracking in Vision-And-Language Navigation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ke_Tactical_Rewind_Self-Correction_via_Backtracking_in_Vision-And-Language_Navigation_CVPR_2019_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: Tactical Rewind.pdf (Main PDF; 9 pages; SHA256 5aa8e1faafe07ad3307128199b7c4f3a5977b5de450941ab20eb3a57b72bc9cc).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Local navigation policies need a systematic means to leave an unproductive branch and compare plausible goal locations without committing permanently to the first STOP. FAST treats frontier choice and final goal ranking as distinct decisions.

## 3. Mechanism, training, and inference

PDF §2/Algorithm 1 pp 3–5: maintain frontier queue QF of partial routes/actions scored by exploration heuristic L, and candidate queue QC of possible completed routes scored by goal function G. Exploitation follows the local best action until confusion/loops; exploration selects the best frontier. A visited transition graph supports physical backtracking. L may aggregate raw action logits rather than log-softmax; G combines progress, speaker likelihood and path-score statistics through a trained reranker. Short search stops on a local stopping criterion; long search runs a fixed expansion budget. Every visited location can become a final goal candidate. Raw-logit accumulation avoids one softmax-normalization artifact discussed in the paper, but is not theoretically calibrated: adding an arbitrary state-dependent constant to all logits preserves the action distribution while changing summed-logit route ranking. This is a mathematical invariance counterexample, not an executed benchmark result.

## 4. Novelty and evidence

Explicit separation of exploration and goal selection, organized backtracking, and plug-in search over existing followers. It directly anticipates proposals to decouple movement from stopping or rerank visited goal candidates; a new contribution must go beyond that abstraction.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 short search test SR 54/SPL 41/TL 22.08m; long search 61/3/196.53m. Self-monitoring beam search has 61 SR/2 SPL/373.09m. Table 2 plug-in validation improvements for Speaker-Follower 37→43 SR and self-monitoring 47→56 SR demonstrate some policy independence. VERDICT: improved search efficiency relative to a particular beam baseline, with a substantial SR–movement tradeoff; no proof of global score comparability.

Table 4, PDF p 8, reranks a fixed pool produced by 40 frontier expansions: all combined goal signals achieve unseen SR 64.03 versus 58.75 for mean log probability, 43.68 for speaker score alone and oracle 90.20. This is direct prior art for a learned visited-goal verifier using several global signals. Table 3/p 7 also finds raw-logit sums and log-probability sums close (56.66 versus 56.23 SR); it does not establish softmax normalization as the dominant source of error.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/Kelym/FAST](https://github.com/Kelym/FAST); inspected snapshot **c4f1dabb388a16d96a388d4d6c3a309047bc41e3**. Snapshot metadata and source are under `repositories/01/tacticalrewind`.

Inspected frontier expansion, raw-logit/log-prob options, separate ending queue, final candidate reranker, and physical return path. This is concrete released prior art for goal-candidate return behavior.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [follower.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/follower.py) — lines 844–917: Tracks logit/log-prob summaries; selectively expands topK actions; routes STOP candidates to ending_queue separately from frontier nodes.
- [follower.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/follower.py) — lines 966–986: Goal reranker uses length/logit/log-prob/progress/speaker features, then realistic_jumping returns through the visited graph.
- [follower.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/follower.py) — lines 542–551 and658–666: Progress MSE and action CE training remain separate from search heuristics.
- [run_search.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/run_search.py) — search setup and sweep entrypoints: Search settings are explicit experiment choices and must be matched in comparisons.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **tacticalrewind-B1 — Partially addressed**. Higher success consumes far more movement; a fixed expansion budget is not adaptive value-of-information control. Evidence: Table 1 short/long comparison. Type: inference cost. Follow-ups: DUET global single-run policy. Confidence: high.
- **tacticalrewind-B2 — Still unresolved in reviewed evidence; broader frontier audit required**. Raw-logit heuristic is sensitive to arbitrary state-dependent offsets even when local action probabilities are unchanged. Evidence: L definition §2; softmax invariance deduction; no empirical claim. Type: score identifiability. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/cshizhe/VLN-DUET
