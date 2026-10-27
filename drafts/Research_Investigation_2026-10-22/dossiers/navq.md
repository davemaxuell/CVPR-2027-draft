# NavQ

## Bibliography and reviewed material

**NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation**. Peiran Xu; Xicheng Gong; Yadong Mu. ICCV 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/ICCV2025/html/Xu_NavQ_Learning_a_Q-Model_for_Foresighted_Vision-and-Language_Navigation_ICCV_2025_paper.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- NavQ.pdf; Main PDF; SHA256 `3d41994c0f586eda436c86f79d801279823ef75f0302dcf2657ac0fd6d412708`; 15 pages.

## Problem, task and assumptions

Discrete goal-oriented VLN on REVERIE/SOON. Natural language specifies an object goal, so conventional reward-labelled Q-learning needs scarce instructions. Seeks task-agnostic future semantics from unlabeled scene graphs.

## Method, training and inference

Section 3 pp. 3–6: uses DUET global/local graph encoders plus a future branch. Q (T,a) is a vector of discounted future caption features, not scalar expected reward: R (A)+γEπQ (T∪{A},a′), Eq. 6. Targets sum node-caption embeddings weighted by future visitation probability and γ^t, Eq. 8. Rollouts disallow revisits and are constrained to shortest continuations to make actions discriminative. A4-layer Transformer processes interleaved historical node/action tokens and candidate directions; masked-autoencoder warmup precedes MSE target regression. The Q-model is frozen for subsequent agent training. A4-layer graph future encoder cross-attends to instruction features; its output augments DUET fusion. Separate heads predict traversed distance from global features and remaining distance from future features. DUET auxiliary pretraining plus these distance tasks precedes online DAgger finetuning. Optional 1351 HM3D/Gibson scenes train Q without speaker instructions. Stage batch/LR/iterations:128/1 e−5/30 k,32/5 e−5/100k,4/1 e−5/20 k; single 3090 reported.

## Novelty and closest predecessors

Successor-feature-like prediction of text embeddings and a goal-conditioned future graph branch avoid RGB or recurrent test-time rollouts. A*-inspired auxiliary heads are not an admissible A* heuristic or optimality proof. Closest provided competitors are DUET, HNR, VLN-SIG and semantic prior methods.

## Results and independent verdict

PAPER EXPERIMENT: Table 3 p. 7 matched reproduced baseline SR 48.14/SPL 33.38/RGSPL 21.05; FE only 48.20/33.92/23.14; Q only 48.48/32.22/21.86; full 53.22/38.89/27.12; oracle Q+FE59.27/47.04/31.62. Main-table original DUET differs in visual backbone, acknowledged p. 7; use reproduced baseline for attribution. REVERIE test SR 53.29 vs reportedDUET52.51; substantial val gains shrink on test. SOON test SR 38.59 vs 33.44 but GOAT40.50, so not universally best. Table 4, p. 8: gamma 0/0.3/0.5/0.7 gives SR 51.12/51.95/53.22/50.89. Longer successor aggregation is useful only up to a point in this experiment. The source constructs targets on one stored shortest-path tree, not all equal shortest paths.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/woyut/NavQ_ICCV25. Snapshot commit: `9465fe43c9c1583bd7ba390b6be30996ea4f5a5c`. Retrieval: 2026-10-22.

INSPECTED CODE: `code/Q_pretrain/GPT_VL.py:125–240` forms visual/text/action sequences and predicts 512-dimensional future features with squared-error supervision, rather than a scalar reward Q-function. `code/Q_pretrain/dataset.py:180–244` explicitly constructs the target from stored shortest paths: immediate caption features plus gamma-discounted downstream features weighted by branch probabilities. Only one stored shortest path per destination is used, resolving ties operationally without proving graph-theoretic uniqueness. Earlier dataset code averages 36 views into 12 and optionally adds ScaleVLN scenes; `train.py` contains optimizer and MSE updates. This supports the successor-feature interpretation and identifies a concrete data/architecture confound. The audit covers Q-model construction, supervision, training and dataset preparation; it does not certify every downstream navigation/evaluation branch or replicate the main tables.

Inspected files: `code/Q_pretrain/GPT_VL.py`, `code/Q_pretrain/dataset.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Averaging future semantics can erase rare goal evidence; targets are policy/distribution dependent, not probabilities of correct navigation. The p. 5 uniqueness claim for shortest continuation requires tie-breaking/unique shortest paths: a diamond graph permits two equal first actions to one node. Shortest-route assumptions also do not transfer directly to route-faithful R2R or continuous collision constraints. Feature-oracle gap remains substantial. Extra-scene benefit does not establish architecture-only gain.

Relevant follow-up relationship: UNeMo jointly predicts task-conditioned future features; SC²-WM adds online adaptation in CE. Neither is a directly matched resolution of NavQ's aggregated-semantic ambiguity.

## Access and uncertainty record

Full text is preserved under `corpus/navq/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/navq/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-navq-aggregation

**Still unresolved in the reviewed literature — Methodological.** Successor semantic averaging can erase rare or route-specific evidence.

Conditions: Long discounted continuations, multiple equal shortest routes, or route-faithful instructions. Cause: A fixed-dimensional weighted semantic target summarizes future captions under a selected continuation rather than representing all branch-specific consequences.

Evidence: PAPER EXPERIMENT: Table 4 p.8 gamma 0/.3/.5/.7 SR 51.12/51.95/53.22/50.89; Table 3 p.7 predicted full SR 53.22 versus oracle 59.27. INSPECTED CODE: dataset uses one stored shortest path per target.

Follow-up assessment: UNeMo adds task-conditioned future features and SC²-WM works in CE, but neither supplies a matched remedy for aggregated-semantic ambiguity. Confidence: Medium.

### 02-navq-uniqueness

**Still unresolved in the reviewed literature — Methodological.** Shortest continuation is not mathematically unique without a tie-breaking or graph assumption.

Conditions: Graphs containing equal-length shortest paths to the same destination. Cause: Distinct first actions can share a shortest destination path length; the implementation chooses one stored path operationally.

Evidence: DEDUCTION/EXECUTED TEST: diamond-graph counterexample in evidence/02/logic_check_results.json; source dataset target construction, lines 180–244 at the dossier commit.

Follow-up assessment: The released target construction resolves operational labeling via a stored path, but does not establish the printed general uniqueness claim. Confidence: High: the unqualified uniqueness claim has a counterexample; stored-path labeling remains well defined.

Independent isolated check (**EXECUTED TEST**): An isolated diamond graph has two equally short first actions to the same destination, refuting unqualified shortest-continuation uniqueness. The released dataset still defines operational targets through a stored shortest path. Setup and output: `evidence/02/logic_checks.py` and `evidence/02/logic_check_results.json`.
