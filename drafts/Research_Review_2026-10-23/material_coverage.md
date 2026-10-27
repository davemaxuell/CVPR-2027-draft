# Supplied material coverage

The supplied folder contains ten files. Exact absolute paths, byte lengths, modification times and SHA256 versions are in [material_inventory.json](material_inventory.json). All ten were inspected; [package verification](evidence/package_verification.json) compares current hashes with intake. Original files and the installed skill remain unchanged.

| Supplied file | Reconstruction role | Primary ownership / inspection |
|---|---|---|
| README.md | Entry and current scope | Coordinator,A3,A6 |
| CE_World_Model_Approach.md | Objective, task, trials, training/data/evaluation | Coordinator,A1–A7; source verification A9/A10 |
| Scorer_Design.md | Scorer, information, losses, pilot | Coordinator,A2–A7,A9/A10 |
| Backbone_Trials.json | Three locked model families/revisions | Coordinator,A3,A6 |
| World_Model_Trials.json | Four families, open weights, access | Coordinator,A3,A6 |
| Scorer/config.json | Tensor/hyperparameter contract | Coordinator,A2,A3,A6 |
| Scorer/scorer.py | Actual detach, memory, STOP, heads, selection | Coordinator,A2,A3,A6,A7,A9 |
| Scorer/labels.py | Utility arithmetic | Coordinator,A2,A3,A6,A7 |
| Scorer/check_design.py | Synthetic reference test scope | Coordinator,A3,A6; copied execution A6 |
| Scorer/check_results.json | Supplied result distinct from fresh run | Coordinator,A3,A6 |

Earlier linked `Research_Investigation_2026-10-22` reports/reviews and literature/repository records supplied context and leads, with fresh primary checks for substantive current claims. They are not treated as approved design, current empirical results, or a repeated131-paper systematic review. Newly inspected paper versions/publication status, repository commits, precise locators and assignment ownership are in [Sources](Sources.md) and individual evidence ledgers. Downloaded primary snapshots are evidence copies, not reproduced experiments.

Material gaps: no supplied current CE branch tape, trained navigation adapter, selected executable WM checkpoint, real-image integration, full GPU profile or CE benchmark result. Scoped probes did not demonstrate assets/working Habitat; other environments/locations may exist. NWM metadata indicates gated weights, but account authorization was not tested. Incomplete releases and inaccessible endpoints are recorded in the relevant dossiers with their consequences.
