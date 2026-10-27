# PlaNet

## Bibliography and reviewed material

**Learning Latent Dynamics for Planning from Pixels**. Danijar Hafner; Timothy Lillicrap; Ian Fischer; Ruben Villegas; David Ha; Honglak Lee; James Davidson. ICML 2019 (2019). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1-6. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- PlaNet.pdf; Main PDF; SHA256 `1c3aee51ab99649e970815a7f8bccae8c7711addbfc8d50c4853a0d2e742f26a`; 11 pages.

## Problem, task and assumptions

Continuous control from 64×64 images and rewards, without state access during acting. Learns a latent simulator from online data and plans actions through it.

## Method, training and inference

Sections 2–4: RSSM deterministic history h_t=f (h_(t−1),s_(t−1),a_(t−1)) and Gaussian stochastic state s_t; posterior conditions on current encoded image, prior on history. Pixel/reward likelihood and prior-posterior KL form variational objective (Eq. 3). Optional latent overshooting compares multi-step transition priors with observed posteriors (Eqs.5–7), stopping posterior gradients for distances>1. Crucially p. 5 states final RSSM does not require overshooting. MPC/CEM samples Gaussian action sequences, predicts latent rewards, selects elites/refits distribution for several iterations, executes the first action and replans from the new posterior. No pixel decoding is needed inside planning, and there is no learned actor/value network. Random seed experience is followed by planner-collected experience and repeated model updates.

## Novelty and closest predecessors

Combines deterministic and stochastic recurrent state with online latent MPC; extends latent generative dynamics beyond one-step reconstruction. Overshooting is a methodological contribution but should not be falsely described as essential to every reported RSSM result.

## Results and independent verdict

Figures 4–5 evaluate sixDMC tasks and show importance of combining deterministic memory and stochastic states. Reported sample-efficiency comparison with model-free agents uses different compute and interaction budgets. These experiments support compact latent planning on their suite, not calibrated model probabilities or VLN performance.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/google-research/planet. Snapshot commit: `c04226b6db136f5269625378cd6a0aa875a92842`. Retrieval: 2026-10-22.

INSPECTED CODE: `planet/models/rssm.py:96–139` implements recurrent belief, Gaussian prior, and observation-conditioned posterior; decoder features concatenate stochastic state and belief. `planet/training/define_model.py:52–78` connects posterior/prior inference to configured objectives and active episode collection. `planet/scripts/configs.py:180–198` exposes KL/free-nats, optional overshooting, reward-loss scaling and Adam settings; overshooting is not enabled merely because its code exists. `planet/control/planning.py:24–60` samples action sequences, rolls out latent dynamics, scores predicted reward, and refits CEM elites. `planet/control/mpc_agent.py:57–83` executes only the first selected action and updates from the next real observation. `planet/tools/preprocess.py:22–30` reduces images to five bits by default and adds dequantization noise. The source provides training and control interfaces; this audit does not execute them or verify historical checkpoint compatibility.

Inspected files: `planet/control/mpc_agent.py`, `planet/control/planning.py`, `planet/models/rssm.py`, `planet/scripts/configs.py`, `planet/scripts/objectives.py`, `planet/tools/preprocess.py`, `planet/training/define_model.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Planning cost scales with candidate sequences×horizon×iterations. A single stochastic rollout per candidate is a noisy score, and selecting high predicted reward can exploit model errors. Longer imagined horizons compound model bias; limited observation history can alias hidden states.

Relevant follow-up relationship: Dreamer amortizes planning into an imagined actor; DINO-WM removes reward and pixel-decoder dependence through pretrained features; NavMorph applies RSSM to language navigation. All shift tradeoffs rather than proving model correctness.

## Access and uncertainty record

Full text is preserved under `corpus/planet/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/planet/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-planet-online_search

**Shifted into another cost or limitation — Methodological.** Receding-horizon latent CEM requires repeated candidate rollouts at each action decision.

Conditions: Latency-limited control with long horizons or many candidate sequences. Cause: Compute grows with candidate count, horizon and optimizer iterations; one sampled stochastic rollout per candidate can also produce noisy ranking.

Evidence: AUTHOR METHOD: latent CEM procedure and inspected planner source; Figures 4–5 test performance, not a universal latency bound.

Follow-up assessment: Dreamer amortizes search into a learned actor/value model. Online cost is reduced by accepting training cost and approximation error. Confidence: Medium.

### 02-planet-model_bias

**Still unresolved in the reviewed literature — Methodological.** Selecting high predicted reward can exploit transition and reward-model errors.

Conditions: Long rollouts or actions poorly supported by real experience. Cause: The same learned model both predicts outcomes and supplies the objective optimized by search.

Evidence: DEDUCTION: RSSM/reward planning pipeline; World Models supplies direct virtual-versus-real exploitation evidence, while PlaNet tests do not establish general calibration.

Follow-up assessment: Dreamer changes policy optimization and CVAML corrects a different value-aware estimator bias; neither is proof of physically correct unsupported rollouts. Confidence: Medium.
