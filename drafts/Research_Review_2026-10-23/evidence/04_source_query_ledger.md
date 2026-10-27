# Assignment 4 source and query ledger

Research date/cutoff: 2026-10-23, Asia/Seoul. Native browser search/open tools, GitHub connector reads, and local standard-library HTTP retrieval were used. No external model API or scientific experiment was run. Parent-provided paper names were leads only; claims in the report were checked against primary text.

## Local evidence and integrity

- Original design: `../../Research Approach/CE_World_Model_Approach.md` and `../../Research Approach/Scorer_Design.md`, read in full across bounded reads.
- Original reference implementation: `Scorer/scorer.py` candidate evaluation, losses and margin selection, and complete `Scorer/labels.py`; no modifications.
- Archived full sources: `a4_sources/*.html` and derived text files. `a4_collect.py` is the retrieval/parser helper. `download_manifest.json` and `extra_manifest.json` contain primary URLs, byte counts and SHA-256 digests.
- Four fresh pinned official code files were fetched through the GitHub connector and saved alongside sources. They were not executed. Exact repository commits/paths appear in the assignment.
- Initial retrieval using the default MSYS Python failed certificate validation for every URL. Retried with installed Windows CPython 3.14, preserving normal certificate verification; all 15 full HTML sources succeeded. No TLS-verification bypass was used.
- Full-paper availability does not mean every reference or mathematical proof was independently verified. Inspection concentrated on mechanisms, objectives, inference, diagnostic experiments and relevant supplements. Source methods are reported as authors' descriptions unless separately marked inspected code.

## Primary source record

| ID / archived basename | Primary full text and status source | Inspection scope and qualification |
|---|---|---|
| P01 `dreamwalker` | https://arxiv.org/html/2308.07498v1 ; https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.html | Main §3.1–3.2; Table 2/Fig.4; Appendix I distance/evaluator curriculum and planning. ICCV 2023 verified. |
| P02 `hnr` | https://arxiv.org/html/2404.01943v1 ; https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Lookahead_Exploration_with_Neural_Radiance_Representation_for_Continuous_Vision-Language_Navigation_CVPR_2024_paper.html | §3.2–3.3, Table 3, Appendix A.2/C and fresh rendering code. CVPR 2024 verified. |
| P03 `dino_wm` | https://arxiv.org/html/2411.04983v2 ; https://proceedings.mlr.press/v267/zhou25t.html | §3.1–3.2; appendix structure; training config inspected. ICML 2025 verified; do not mislabel the earlier ICLR review copy as accepted publication. |
| P04 `nwm` | https://arxiv.org/html/2412.03572v2 ; https://openaccess.thecvf.com/content/CVPR2025/papers/Bar_Navigation_World_Models_CVPR_2025_paper.pdf | §3.1–3.3/4.4 and planning code. CVPR 2025 verified. |
| P05 `streamvln` | https://arxiv.org/html/2507.05240v1 ; https://github.com/InternRobotics/StreamVLN | §3 and appendix framework; agent code. Preprint verified. ICRA 2026 is reported by official repository; proceedings not separately inspected. |
| P06 `vistav2` | https://arxiv.org/html/2512.00041v1 ; https://arxiv.org/abs/2512.00041 | §3.1–3.4, Algorithms 1–2; §4.1–4.3. Preprint only verified. Metadata says submitted 2025-11-14 despite December-format identifier. No separately accessible appendix/code was established. |
| P07 `seekvln` | https://arxiv.org/html/2609.37353v1 ; https://arxiv.org/abs/2609.37353 | §3.1–3.3, C.2/D.2 and real deployment appendix. Preprint 2026-09-29; no venue found/verified. |
| P08 `futurenav` | https://arxiv.org/html/2606.30367v1 ; https://arxiv.org/abs/2606.30367 | §3.2–3.4, training-vs-inference pathway. Preprint 2026-06-29 verified; typeset title block additionally says August 24, 2026. Version metadata governs citation. |
| P09 `cast` | https://arxiv.org/html/2508.13446v2 ; https://prism.robotics.princeton.edu/publications/contributor/shah-dhruv-54 | §IV–V and counterfactual data mechanism. v2 2026-06-08; author's institution lists RA-L 2026. Publisher record was not independently inspected. |
| P10 `worldmap` | https://arxiv.org/html/2604.07957v1 ; https://arxiv.org/abs/2604.07957 | Teacher/student and task distinction screened from full source; no code. Preprint 2026-04-09 verified. Less detailed inspection than closest sources P06/P07/P14. |
| P11 `dreamnav` | https://arxiv.org/html/2509.11197 ; https://arxiv.org/abs/2509.11197 ; https://yixfeng.github.io/ | §III-C–E and §IV-A. Preprint 2025-09-14 verified; author homepage reports ICRA 2026. Proceedings not separately checked. |
| P12 `navmorph` | https://arxiv.org/html/2506.23468 ; https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.pdf | §3.1–3.4, especially Eq.5–6 loss and working modes. Full HTML includes supplement. ICCV 2025 verified. |
| P13 `cvaml` | https://arxiv.org/html/2505.22772 ; https://proceedings.mlr.press/v267/voelcker25a.html | §3–4 and §6 auxiliary losses; appendix coverage identified, proofs not independently rederived. ICML 2025 verified. |
| P14 `world4scorer` | https://arxiv.org/html/2609.36438v1 ; https://arxiv.org/abs/2609.36438 | §3.1–3.4 Eqs.1–7, Appendix D.2 read closely. Abstract title includes “World4Scorer”; HTML title omits prefix. Preprint 2026-09-29 verified; no venue confirmed. Repository redirected from GuobaPei/World4Scorer to World4Scorer/World4Scorer; implementation not audited. |
| P15 `mcr` | https://arxiv.org/html/2608.09073v1 ; https://arxiv.org/abs/2608.09073 | §3.3–3.4 Eqs.3–8 and experiment setup; appendices B/C available. Preprint 2026-08-10 verified; no venue confirmed. |
| P16 VAML | https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf ; https://proceedings.mlr.press/v54/farahmand17a.html | Primary paper accessible, formulation/motivation followed from CVAML citation. AISTATS 2017 verified. |
| P17 TD-MPC | https://proceedings.mlr.press/v162/hansen22a.html | Official publication abstract inspected. Full PDF browser retrieval failed at 12,435,898-byte content-size limit. ICML 2022 verified; detailed independent implementation comparison not claimed. |

## Official-code record

| Saved file | Exact URL | Inspection |
|---|---|---|
| `dino_train.yaml` | https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/train.yaml | `train_encoder: False`, `train_predictor: True` at lines 68–69 in fresh connector content. |
| `nwm_planning_eval.py` | https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/planning_eval.py | LPIPS objective, repeated stochastic evaluation, candidate sorting; lines 201, 242–265. |
| `streamvln_agent.py` | https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/streamvln/streamvln_agent.py | Generation/cache/action parsing around 244–258. |
| `hnr_nerf.py` | https://github.com/MrZihan/HNR-VLN/blob/6c935c1f6f0d715b1c6bef28724bdfabb48b159e/NeRF/models/etp/nerf.py | Feature alpha/transmittance computation, around 182–195. |

## Search log

Queries were submitted in batches to the native web search tool. Search results and third-party summaries were used only for discovery, followed by primary URLs above. No-hit searches are not evidence of novelty.

1. `DREAMWALKER vision language navigation world model paper`
2. `HNR hierarchical neural representation hallucination navigation vision language predictive`
3. `vision language navigation future prediction reranking world model 2026`
4. `DREAMWALKER vision language navigation arxiv`
5. `DINO-WM Navigation World Models NWM paper`
6. `vision language navigation counterfactual outcome value critic reranking future 2026`
7. `"Value-Aware Model Learning" Voelcker 2025 CVAML`
8. `"StreamVLN" ICCV 2025 arxiv`
9. `"DINO-WM" ICLR 2025 accepted`
10. `"Task Informed Dynamics Learning" visual model value aware TD-MPC`
11. `"DINO-WM" site:proceedings.mlr.press`
12. `"VISTAv2" conference code`
13. `"Seek Before You Move" github conference`
14. `"CAST: Counterfactual" ICRA RSS 2026`
15. `"DINO-WM" site:proceedings.mlr.press/v267`
16. `"Task-Oriented Model Learning" value aware ranking world model decision focused 2025`
17. `"Lookahead Exploration" "CVPR" 2024`
18. `"NavMorph" ICCV 2025`
19. `Value-Aware Model Learning Farahmand 2017 proceedings`
20. `Temporal Difference Learning for Model Predictive Control ICML 2022 paper`
21. `Value Gradient weighted Model Based Reinforcement Learning ICLR VaGraM official`
22. `vision language navigation value successor "NavQ"`
23. `"Look Before You Leap" vision language navigation 2018 paper`
24. `"MapDream: Task-Driven" proceedings mlr 2026`
25. `"DreamNav" ICRA 2026 accepted`

Direct primary URL opens added parent-discovered VISTAv2, SeekVLN, FutureNav, CAST, WorldMAP, World4Scorer and MCR. Publication metadata for the recent arXiv sources was explicitly opened, rather than inferred from identifier year or a secondary leaderboard.

## Citation chains and remaining leads

- DREAMWALKER cites earlier imagined VLN planning and Pathdreamer; HNR directly responds to generated-RGB lookahead. [Look Before You Leap, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Wang_Look_Before_You_ECCV_2018_paper.html) was located but not fully audited in this bounded assignment.
- CVAML points to VAML, IterVAML, MuZero, TD-MPC and VaGraM. The [VaGraM OpenReview PDF](https://openreview.net/pdf?id=4-D6CZkRXxI) was blocked by a browser verification challenge. Search located arXiv:2204.01464, but no full alternative read was completed. Do not cite the author's older CV as definitive venue year; search located an ICLR 2022 publication copy.
- [MapDream, ICML 2026](https://proceedings.mlr.press/v306/lian26c.html) has a verified proceedings entry and relevant task-shaped mapping premise. Only abstract/publication screening was completed; no contribution-level verdict is based on its full method.
- Parent-discovered **Beyond Euclidean Proximity**, arXiv:2605.22164, remains a handoff lead for Assignment 5; this worker did not independently inspect it.
- SimpleVLN arXiv:2610.23203 and ReInsVLN arXiv:2610.26545 are being checked by Assignment 1; they are not primary evidence borrowed into this audit.
- There was no time-bounded evidence that all relevant 2026 preprints were exhausted. The report therefore rejects broad novelty claims where overlap is demonstrated and leaves exact-intersection originality uncertain.

## Unresolved tests

The assignment report specifies candidate headroom, matched direct value, actual-future ceiling, candidate reassignment, valid language/history interventions, same-family objective comparisons, and full pipeline cost. These are proposed experiments; none are claimed executed.
