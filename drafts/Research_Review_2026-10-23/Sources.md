# Source index and evidence boundaries

Search cutoff **23 October 2026**. This is a targeted scientific and novelty audit, not an exhaustive systematic review. Full query histories, precise inspected scope, downloaded-source hashes and unsuccessful retrievals are recorded in [A1](assignments/01_problem_baselines.md), [A2](assignments/02_formulation.md), [A4 source ledger](evidence/04_source_query_ledger.md), [A5](assignments/05_crossfield.md), [A9](assignments/09_correctness.md), [A10](assignments/10_novelty_viability.md), and [coordinator searches](evidence/coordinator_search_log.md). Source availability does not imply that every linked version or repository was executed.

| Source | Publication status established in this run | Principal locators and use |
|---|---|---|
| [DREAMWALKER](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DREAMWALKER_Mental_Planning_for_Continuous_Vision-Language_Navigation_ICCV_2023_paper.pdf) | ICCV2023 proceedings | §§3.3–4.7, Table2/Figure4; future/evaluator interventions, privileged-distance caveat |
| [HNR](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Lookahead_Exploration_with_Neural_Radiance_Representation_for_Continuous_Vision-Language_Navigation_CVPR_2024_paper.pdf) | CVPR2024 proceedings | §§3.2–3.3/4.3, Table3; oracle feature comparison, model-design rather than confirmed weight-sharing control |
| [HNR supplement](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wang_Lookahead_Exploration_with_CVPR_2024_supplemental.pdf) | Proceedings supplement | A.1–A.2; training/data/resource context |
| [StreamVLN](https://arxiv.org/html/2507.05240v2) | ICRA2026 acceptance stated in author metadata/repository; inspected arXiv v2 | §§III-D/IV, TablesIII–IV; history/data training and action-schedule controls |
| [DINO-WM](https://proceedings.mlr.press/v267/zhou25t.html) | ICML2025 proceedings | [Full text](https://arxiv.org/html/2411.04983v2), §3/AppendixA.4–A.5; frozen visual encoder, latent dynamics, MPC |
| [NWM](https://openaccess.thecvf.com/content/CVPR2025/papers/Bar_Navigation_World_Models_CVPR_2025_paper.pdf) | CVPR2025 proceedings | [Full text](https://arxiv.org/html/2412.03572v2), §3/4.4/Appendix7; CDiT, image-goal reranking |
| [VISTAv2](https://arxiv.org/html/2512.00041v1) | Preprint; no venue verified | §§3.1–3.3; frozen-stack imagined candidate scoring. Empirical protocol/supervision ambiguities retained |
| [SeekVLN](https://arxiv.org/html/2609.37353v1) | Preprint,29 Sep2026 | §3.3 Eqs9–13/AppendixC.2,D.2; matched simulator/context branches and benefit-based credit |
| [World4Scorer](https://arxiv.org/html/2609.36438v1) | Preprint,29 Sep2026 | §§3.2–3.4; AppendicesC.4,D.1,D.2; candidate outcome supervision, future anchoring and frozen-world-model scorer |
| [Monotone Cost Ranking](https://arxiv.org/html/2608.09073v1) | Preprint,10 Aug2026 | §3.3.2 Eqs6–7/AppendixB.3; ranking-aware navigation dynamics |
| [FutureNav](https://arxiv.org/html/2606.30367v1) | Preprint,29 Jun2026 | §§3.3–4.3, Tables2–3; training-only predictive objective rival |
| [SimpleVLN](https://arxiv.org/html/2610.23203v1) | Preprint,16 Oct2026 | §§3–4, Tables1–4; architecture-preserving navigation baselines; release/settings differences in A1 |
| [ReInsVLN](https://arxiv.org/html/2610.26545v1) | Preprint,19 Oct2026 | §III-A–C/TableI; history-use alternative, selected samples and reference-derived reminders |
| [NavMorph](https://openaccess.thecvf.com/content/ICCV2025/papers/Yao_NavMorph_A_Self-Evolving_World_Model_for_Vision-and-Language_Navigation_in_Continuous_ICCV_2025_paper.pdf) | ICCV2025 proceedings | [Full text](https://arxiv.org/html/2506.23468v2), §3 and §§8–9; latent world/history/navigation objectives |
| [DreamNav](https://arxiv.org/html/2509.11197v1) | Preprint inspected; author page reports ICRA2026, proceedings not checked | §III-C–E; language-based imagined-trajectory comparison |
| [WorldMAP](https://arxiv.org/html/2604.07957v1) | Preprint,9 Apr2026 | §III-B–C/AppendixB; training-only teacher/student alternative |
| [CAST](https://arxiv.org/html/2508.13446v2) | Preprint inspected; author lab reports RA-L2026 | §§IV–V, Eqs1–2; counterfactual language/action supervision, distinct from actual branch outcomes |
| [VAML](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf) | AISTATS2017 proceedings | §§2–4; task/value-relevant model fitting precedent |
| [VaGraM](https://openreview.net/pdf?id=4-D6CZkRXxI) | ICLR2022 | Value-gradient weighting and off-support-value rationale; exact coverage in A2/A5 |
| [CVAML](https://proceedings.mlr.press/v267/voelcker25a.html) | ICML2025 proceedings | [Full text](https://arxiv.org/html/2505.22772v2), §4.2 Eq3/Proposition3; stochastic model-loss calibration |
| [DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf) | AISTATS2011 proceedings | §3 Algorithm3.1; learner-occupancy data aggregation |
| [AggreVaTe](https://arxiv.org/html/1406.5979) | 2014 technical report/preprint verified | §2; cost-to-go-aware learning and aggregation |
| [Policy improvement](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) | Author-hosted 2018 Sutton–Barto textbook draft | §4.2 Eqs4.6–4.9. A2 reports direct PDF timeout and primary indexed-text verification; our finite-horizon proof is separately supplied |
| [IterVAML](https://papers.nips.cc/paper/2018/file/7a2347d96752880e3d58d72e9813cc14-Paper.pdf) | NeurIPS2018 proceedings | Eq11, Algorithm1; fixed/current-value fitting lineage |
| [Value equivalence](https://proceedings.neurips.cc/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf) | NeurIPS2020 proceedings | §3 Definition1; Bellman equivalence is stronger than action ranking |
| [Task-based model learning](https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf) | NeurIPS2017 proceedings | §§3.1–3.3 Eqs3–7; decision-focused precedent |
| [Melding the Data–Decisions Pipeline](https://ojs.aaai.org/index.php/AAAI/article/view/3982) | AAAI2019 venue record | Problem formulation and continuous relaxations; decision-focused learning |
| [Future-trajectory distillation](https://github.com/neitzal/learning-to-distill-trajectories) | ICLR2021 indexed paper and author repository | README/task transfer; direct OpenReview PDF challenged, so limited mechanism coverage |
| [Privileged Foresight Distillation](https://arxiv.org/html/2604.25859v2) | Preprint, v1 28 Apr/v2 2 May2026 | §§3.2–3.4 Eqs4–10; privileged future correction, future-free deployment |
| [CEM proposal audit](https://arxiv.org/html/2610.00921v1) | Preprint,1 Oct2026 | §§3–4,6; scoring/proposal confounding; exact four candidates avoid this particular loop |
| [TRM](https://arxiv.org/html/2605.22164v2) | Preprint, v1 21 May/v2 31 Aug2026 | Method Eqs5–6, Table4; temporal cost supervision and horizon matching |
| [Beyond Next-Screen Prediction](https://arxiv.org/html/2610.24372v1) | Preprint,17 Oct2026; only v1 verified, no venue | §3.1 Eq1, §3.3, §5.3, §6.4; matched direct/prediction decision value and offline/online distinction. Fresh A10 finding, independently opened by coordinator |
| [RIWANav](https://arxiv.org/html/2610.08640v1) | Preprint,6 Oct2026; no venue verified | §III-C Eqs4–7; geometric-consistency/novelty-weighted native dynamics, distinct from measured frozen-critic continuation gaps |
| [LAW](https://aclanthology.org/2021.emnlp-main.328.pdf) | EMNLP2021 proceedings | Abstract, §§1/3; shortest-goal supervision can conflict with CE instructions, limiting novelty of obvious goal-progress reversals |
| [PiJEPA](https://arxiv.org/html/2603.25981v1) | Preprint,26 Mar2026 | §3.4 conditioning; language in policy prior, goal image still required by planning cost |
| [Grounded World Model](https://arxiv.org/html/2604.11751v1) | Preprint,13 Apr2026 | §2.2 Eq3; language alignment of predicted outcomes, distinct from measured CE return supervision |

Selected official implementation pins: DINO-WM `0a9492fa12044b852ae9e001cc74604b79c8bb0c`; NWM `3f6cd8e70d6f2d1e2b9684acff510710135f0f41`; StreamVLN `e48f6ff7e9201d93aae003e8f64b04c00cec13bc`; HNR `6c935c1f6f0d715b1c6bef28724bdfabb48b159e`; SimpleVLN `89b72a59c8f7feb805c0918e43dd5bfbdd7c48cc`; FutureNav `ef4cb4358a0aeb7afed60ad79f7c8e85668a48f2`. Exact files/lines, source hashes, and code-inspection versus execution distinctions are in A1/A4/A6.

The ten supplied local files are individually versioned in [material_inventory.json](material_inventory.json). Their claims are reconstructed in [original_specification.md](original_specification.md). Prior linked investigations were read as context and leads; this run does not claim to repeat the older 131-paper review. Recent preprints constrain novelty even when their effectiveness is not independently reproduced.

The coordinator additionally opened the full GUI paper and metadata on 23 October after A10's fresh search; §§3.1–3.3 and5.3 were directly checked. Its reference-match supervision differs from actual CE continuations, so its empirical conclusion is not transplanted to CE. A10's report records additional GWM/PiJEPA/RIWANav screening and exact inspected scope. No searched-but-unread lead supports a substantive verdict.
