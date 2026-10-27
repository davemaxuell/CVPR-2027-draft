# Shared evidence register

Evidence types remain distinct: **SPEC** supplied design; **PUB** author-reported published/preprint result; **CODE** inspected implementation; **EXEC** check run in this review; **DERIVED** analysis under stated assumptions; **HYP** untested prediction. Multiple workers agreeing is not an additional evidence type.

| ID | Type | Finding / relevance | Source and scope | Does not establish |
|---|---|---|---|---|
| E01 | SPEC | Four primitive H1 futures, frozen policy, outcome utility, staged trials | [Original reconstruction](original_specification.md), [ten-file hashes](material_inventory.json) | CE effectiveness or novelty |
| E02 | PUB | Earlier CE planners respond to oracle future/evaluation interventions | [A1](assignments/01_problem_baselines.md), DREAMWALKER Table2/Fig4 and HNR Table3 | Headroom on these small policies; same frozen HNR weights; deployable oracle |
| E03 | PUB+CODE | Schedule, memory and training alternatives can explain gains | A1 StreamVLN/SimpleVLN/FutureNav/ReInsVLN dossiers and pins | Cross-paper aggregate score comparability |
| E04 | PUB | Imagined candidate scoring and real-branch credit already have close precedents | [A4](assignments/04_direct_novelty.md), VISTAv2§3, SeekVLN§3.3, World4ScorerD.2, MCR§3.3.2 | Exact CE intersection is implemented elsewhere; empirical replication |
| E05 | CODE | Current scorer detaches future inputs and has no WM training path | original scorer.py observe/score/scorer_loss; A2/A4/A6 | A future decision-aware loss is already implemented |
| E06 | DERIVED | Full-history finite-horizon Q_ref improvements telescope; δ≥2ε controls bounded-error overrides | [A2](assignments/02_formulation.md), A3 independent challenge | An empirical margin supplies uniform error bounds |
| E07 | EXEC | 500 depth4 synthetic trees,7500 history checks,79 margin comparisons pass | [A2 code](evidence/02_formal_checks.py), [results](evidence/02_formal_checks.json) | CE metrics or navigation performance; numerical proof of general theorem |
| E08 | DERIVED | Generated futures add no new current-episode observation conditional on full inputs/fixed weights | A2 conditional-independence argument | No benefit from representation, priors, transfer or finite computation |
| E09 | DERIVED | Pairwise squared gaps are centered regression; weighted gaps a graph-Laplacian quadratic | A2, A5 independently checked | A new generic ranking-learning principle |
| E10 | CODE+EXEC | Feature reference works synthetically, but actual CE/WM/backbone adapters remain missing | A6 copied checks and resource audit | Full pipeline fits GPU or tracks language progress |
| E11 | CODE | Inspected default DINO-WM path uses14×14 features, unlike supplied scorer16×16 | A6 pinned vworld_model.py/config analysis | Every unselected checkpoint has that same grid |
| E12 | EXEC | Probed host has RTX4070Laptop8188MiB but Python3.13torch2.9CPU and no Habitat | A6 current environment probes | All environments on host are CPU-only or GPU hardware unusable |
| E13 | DERIVED+EXEC | Correctly distributed one-sample imagination can degrade decision value; noisy sample-max can inflate oracle headroom | A3 utility-compatible counterexamples and numerical record | Actual CE has these rates or all sampled-future methods fail |
| E14 | HYP | A grounded utility-gap loss improves over native prediction and direct-Q | A2 candidates; A8 screening and final v3 review complete, CE test absent | A distinct novel contribution or expected numerical gain |
| E15 | DERIVED+EXEC | Centered loss equals pair-sum/K²; gradient route, STOP/frozen-parameter behavior and 6,561 greedy/margin bound cases checked | [A9](assignments/09_correctness.md), [code](evidence/09_correctness_checks.py), [results](evidence/09_correctness_checks.json) | Full49-token integration, useful critic gradients or CE results |
| E16 | PUB | Matched direct-versus-predicted decision-value protocol already explicit in GUI literature | Beyond Next-Screen Predictionv1 §§3.1–3.3/5.3; fresh A10 search, coordinator primary verification | Same empirical conclusion in CE, or exact measured-return adaptation already implemented |
| E17 | DERIVED | Q_ref is policy-relative, nDTW/success not clause ground truth; turn ties are not strict proxy reversals | A9/A10 independent reviews; proposal v3 claim boundaries | A semantic understanding finding from value mismatch alone |
| E18 | CODE+DERIVED | B-fitted frozen critic transmits B-label information into A-adapted dynamics | Proposed v2/v3 computational graph and A7 data contract | Held-out leakage when all A/B are permitted training scenes; strict A/B independence |

All CE collection, policy training, world-model adaptation, real-image feature inference, unseen navigation evaluation and GPU pipeline profiling remain **not executed** unless separately added to the experiment register with actual logs. Access limits are kept in each source ledger. Final independent evaluations record the exact versions in [reconciliation](review_reconciliation.md); they do not certify the entire project or establish empirical gains. E14 remains a hypothesis after synthesis.
