# Assignment 10 — fresh source and search ledger

Cutoff/access date: **2026-10-23**. Independent review of proposal v1 SHA256 `4CBC57E6781D83BF3C3A88F9030C54EFBD39A45B8B98535EB68D5DD163D77F9F`; subsequent frozen v2 recheck SHA256 `B67DF5A386C4FFAB02672D11A561F7C6025E09A99172E80FEE08A52B0CBC6305`; final v3 delta recheck SHA256 `2E4A46C636502D105422951C5AFE12DD7731C65D2F3C6733EC69643394000909`. Searches were executed afresh. Initial batches preceded the saved independent initial assessment and earlier audit conclusions; later searches continued after selected technical audits were consulted. Search results from aggregators served only as leads; substantive technical claims below use primary sources. This was a targeted novelty review, not an exhaustive absence proof.

## Queries executed

| Batch | Exact queries | Useful outcome |
|---|---|---|
| 1 | `"instruction-conditioned" "world model" "value" navigation ranking 2026`; `"World4Scorer" "Monotone Cost Ranking"`; `"world model" "measured" "utility" "frozen" scorer navigation` | Fresh discovery lead for GUI decision-value paper, PiJEPA; direct opens of World4Scorer and MCR. |
| 2 | `"Beyond Next-Screen Prediction" "arxiv"`; `"world model" "frozen" "value" "ranking" navigation instruction 2026 site:arxiv.org`; `"Iterative Value-Aware Model Learning"` | Primary arXiv identifier for GUI paper; official NeurIPS IterVAML. |
| 3 | `"Value Gradient weighted Model-Based Reinforcement Learning" pdf`; `"instruction" "world model" "value-aware" navigation`; `"semantic" "geometric" "action" "world model" navigation ranking`; `"measured" "outcome" "frozen" "world model" navigation scorer` | Primary VaGraM mirror and author page; NavWM, policy-shaped prediction and other screening leads. |
| 4 | `"world model" "outcome" "contrast" "instruction" navigation ranking 2026`; `"instruction" "value gradients" "world model"`; `"semantic" "geometric" "instruction" "ranking" navigation`; `"Policy-shaped prediction: avoiding distractions"` | RIWANav and Grounded World Model leads; official NeurIPS policy-shaped-prediction paper. |

Broad search batches returned mixed topics. No absent hit was treated as evidence that the exact proposal is unprecedented. No recommendation is based on aggregator performance summaries.

## Primary sources inspected

| Source / primary URL | Version / publication status established | Precise inspected scope |
|---|---|---|
| [World4Scorer full text](https://arxiv.org/html/2609.36438v1), [metadata](https://arxiv.org/abs/2609.36438) | v1, 29 Sep 2026; preprint, no venue verified | §§3.2–3.4, Appendix C.4 including Eqs.11–13/local qualification, D.1 intervention controls, D.2 frozen LeWM experiment/scorer architecture and labels. HTML line ranges approximately 592–649, 652–691. |
| [MCR full text](https://arxiv.org/html/2608.09073v1), [metadata](https://arxiv.org/abs/2608.09073) | v1, 10 Aug 2026; preprint, no venue verified | §3.3.1 Eqs.3–5; §3.3.2 Eqs.6–7; §3.3.3 and Appendix C negative ACT interpretation; §4.1 task/data distinction. HTML lines 144–191. |
| [Beyond Next-Screen Prediction](https://arxiv.org/html/2610.24372v1), [metadata](https://arxiv.org/abs/2610.24372) | v1, 17 Oct 2026; preprint, no venue verified | §3.1 Eq.1; §3.2 Eq.2 labels; §3.3 Eq.3 decision value; §§5.3, 6.1, 6.4 distinctions between matched offline comparisons and unmatched online lookahead. HTML lines 100–119, 152–183. |
| [VAML proceedings](https://proceedings.mlr.press/v54/farahmand17a.html), [PDF](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf) | AISTATS 2017, PMLR54:1486–1494 | Value-aware objective, PDF p.4 Eqs.6–7; robust value-class formulation. |
| [IterVAML proceedings](https://papers.nips.cc/paper_files/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html), [PDF](https://papers.nips.cc/paper/2018/file/7a2347d96752880e3d58d72e9813cc14-Paper.pdf) | NeurIPS 2018 | §3 Eq.11, PDF pp.4–5; Algorithm1, p.6; theoretical guarantees have their own assumptions. |
| [VaGraM PDF](https://arxiv.org/pdf/2204.01464), [author lab page](https://www.pair.toronto.edu/blog/2022/vagram-voelcker/) | Paper header and author page identify ICLR2022; OpenReview direct access challenged | §2.2 value-aware formulation; §§3.1–3.2 Eqs.3–7, PDF pp.4–5, real-state gradient and diagonal weighting. |
| [CVAML proceedings](https://proceedings.mlr.press/v267/voelcker25a.html), [full text](https://arxiv.org/html/2505.22772v2) | ICML2025, PMLR267 | §4.2 Eq.3 and Propositions2–3; stochastic model variance and correction. |
| [SeekVLN](https://arxiv.org/html/2609.37353v1), [metadata](https://arxiv.org/abs/2609.37353) | v1, 29 Sep2026; preprint, no venue verified | §3.3 Eqs.9–13; cloned state/history/context, same-policy finite-horizon branches, geodesic-progress reward, policy/value updates. |
| [VISTAv2](https://arxiv.org/html/2512.00041v1), [metadata](https://arxiv.org/abs/2512.00041) | v1; metadata displays 14 Nov2025; preprint, no venue verified | §§3.1–3.3; RGB/depth/odometry, action-conditioned futures, instruction scorer, value-map fusion and search. Date recorded as displayed, not inferred from identifier. |
| [LAW proceedings page](https://aclanthology.org/2021.emnlp-main.328/), [PDF](https://aclanthology.org/2021.emnlp-main.328.pdf) | EMNLP2021 proceedings | Abstract/§1 and §3 motivation: shortest-goal versus language-aligned supervision. |
| [RIWANav](https://arxiv.org/html/2610.08640v1) | v1, 6 Oct2026; preprint, no venue verified | §III-B Eq.3 policy update; §III-C Eqs.4–7 expert-motion consistency, prioritized replay and native video training. HTML lines 120–139. |
| [PiJEPA](https://arxiv.org/html/2603.25981v1), [metadata](https://arxiv.org/abs/2603.25981) | v1, 26 Mar2026; preprint, no venue verified | §3.4, especially Conditioning paragraph and Algorithm1: language in policy prior, goal-image latent planning cost. |
| [Grounded World Model](https://arxiv.org/html/2604.11751v1) | v1, 13 Apr2026; preprint, no venue verified | §2.2 Eq.3 future embedding/instruction similarity; WISER task scope. |
| [Policy-shaped prediction](https://papers.nips.cc/paper_files/paper/2024/file/17af43527227c5c96db0f8d4c6aadc4e-Paper-Conference.pdf) | NeurIPS2024 proceedings URL verified; targeted indexed §2.1 excerpt, not a full-paper audit | Visual-domain task-gradient weighting lead. Not used to assert exact equivalence with v1. |
| [NavWM metadata](https://arxiv.org/abs/2606.24101) | Submitted 23 Jun2026; abstract-level screening only | Broad unified latent reasoning/action/generation; no exact-loss conclusion. |

## Access limits and independence

- OpenReview forum and PDF for VaGraM returned a browser-verification challenge. The primary arXiv PDF and author lab page supplied full-text and publication-status evidence instead.
- An aggregator claimed the GUI paper was updated 20 October. Primary arXiv metadata lists only v1. A deliberate v2 HTML attempt returned 404; all conclusions use v1.
- No retrieved source was reproduced in a simulator. Performance claims remain the authors' reported evidence, never project results. Newer preprints limit novelty without establishing reproducibility.
- No weights, software packages, model training, simulator rollout or external model call was requested or performed. Browser queries and ordinary file inspection supplied evidence.
- Initial verdict was saved in [10_initial_assessment.md](../assignments/10_initial_assessment.md) before earlier audit conclusions. Later inspected A4/A5 relevant source comparisons, A6 feasibility and raw logs, A7 experiment protocol/register, and relevant original scorer/config/labels code. Did not read A8 synthesis, the main report, or A9.
