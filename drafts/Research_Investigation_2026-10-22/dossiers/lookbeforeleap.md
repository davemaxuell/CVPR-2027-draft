# Look Before You Leap

## Bibliography and reviewed material

**Look Before You Leap: Bridging Model-Free and Model-Based Reinforcement Learning for Planned-Ahead Vision-and-Language Navigation**. Xin Wang; Wenhan Xiong; Hongmin Wang; William Yang Wang. ECCV 2018 (2018). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content_ECCV_2018/papers/Xin_Wang_Look_Before_You_ECCV_2018_paper.pdf. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,4-12. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Look Before You Leap.pdf; Main PDF; SHA256 `e320deeb42c2dfbe018d5733b4c6f5f297dcab78c0b159a94058e02eb74c74e9`; 17 pages.

## Problem, task and assumptions

R2R instruction navigation with six low-level actions and RGB features. Learns visual/reward predictions to augment a model-free policy rather than relying only on history-conditioned action selection.

## Method, training and inference

Sections 3–4 pp. 4–9: instructionLSTM and ResNet152 image features feed model-free policy. Two MLP environment models predict next visual feature and progress reward (Eqs.2–3). For each of five non-stop first actions, a depth-two imagined trajectory is encoded by anLSTM; these rollout features are concatenated with model-free context before final policy scoring. Train environment predictors first by feature/reward MSE using 95%teacher and 5%random actions, then freeze them. Policy learning mixes supervised imitation and REINFORCE, with annealed teacher weight. Reward is geodesic distance reduction D (s)−D (s')(Eq. 10), requiring simulator supervision during training. At inference short feature rollouts are repeated at each real step. Algorithm 1's printed annealing expression uses exp (iter/T) despite 'decrease'; the intended negative sign cannot be assumed without source confirmation.

## Novelty and closest predecessors

Early navigation-specific hybrid of imagination-augmented decision making and imitation/RL, prior to later pixel/NeRF/latent systems. Imagined features directly affect inference but do not form a persistent global map. 

## Results and independent verdict

Table 1 p. 12 unseen SR 24.6 versus cross-entropy 21.3 and model-free 21.5; test 25.3 versus 22.1/23.1. Gains establish proof of concept under this early action interface, not modern baseline competitiveness. Model and policy ablations matter more than cross-era score comparisons.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: No verified official implementation located in the recorded search.. Snapshot commit: `not available`. Retrieval: 2026-10-22.

The original ECCV paper was reviewed; no verified official repository implementing its complete lookahead policy was located. Related later code must not be substituted without labeling it.

Inspected files: No implementation files inspected; access scope is stated above. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Teacher-heavy environment data poorly covers the policy's mistakes; a frozen predictor cannot correct subsequent shifts. Fixed short horizons and deterministic features obscure multimodality. Additional real observations correct history but there is no explicit model-risk estimate or test-time residual update.

Relevant follow-up relationship: DREAMWALKER replaces feature-only branches with explicit future panoramic imagery; HNR predicts features efficiently in continuous navigation. Neither should erase this earlier model-based inference prior art.

## Access and uncertainty record

Full text is preserved under `corpus/lookbeforeleap/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/lookbeforeleap/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-lookbeforeleap-frozen_model

**Partially addressed — Methodological.** Teacher-heavy model data and frozen deterministic short-horizon prediction poorly cover policy errors.

Conditions: Actions and scenes outside the environment-model training distribution. Cause: The predictor is trained before policy use and is not corrected online; deterministic features compress multiple possible outcomes.

Evidence: AUTHOR METHOD and DEDUCTION: two-stage training/rollout sections; Table 1 p.12 unseen SR 24.6 versus model-free 21.5 establishes early utility but not shift robustness.

Follow-up assessment: DREAMWALKER/HNR strengthen predictions; NavMorph and SC²-WM add adaptation. SC²-WM specifically supplies delayed real-observation residual updates, but on a different architecture and CE protocol. Confidence: Medium.
