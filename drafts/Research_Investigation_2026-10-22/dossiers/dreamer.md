# Dreamer

## Bibliography and reviewed material

**Dream to Control: Learning Behaviors by Latent Imagination**. Danijar Hafner; Timothy Lillicrap; Jimmy Ba; Mohammad Norouzi. ICLR 2020 (2020). Research cutoff: **2026-10-22**.

Primary publication/source: https://arxiv.org/pdf/1912.01603v3. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-6. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Dreamer.pdf; Main PDF; SHA256 `318891d59ba0065f57eb0e735735700e5eb528f53bf08b81c773437385586c64`; 20 pages.

## Problem, task and assumptions

Learn continuous-control policies from images and scalar rewards with limited real interaction. It is reward-driven control, not language-conditioned navigation.

## Method, training and inference

Section 3 pp. 3–5: recurrent state-space model has deterministic history and stochastic latent state; observation posterior is fit to a transition prior. World-model training uses pixel/reward likelihood and KL (Eqs.9–10), with contrastive/reward-only variants examined. From posterior states, imagination generates trajectories using the learned actor and transition model. Lambda returns (Eq. 6) combine truncated reward sequences with bootstrapped values. Actor maximizes these returns through differentiable/reparameterized dynamics (Eq. 7); critic regresses stop-gradient lambda targets (Eq. 8). Dynamics parameters are held fixed during behavior updates, while gradients propagate through their operations into actor actions. At deployment real images update the posterior and the actor selects an action; there is no online CEM tree search. Data collection, model fitting and imagined actor/critic learning alternate.

## Novelty and closest predecessors

Relative to PlaNet's per-step optimization, amortizes planning into an actor trained by analytic gradients through multi-step latent dynamics and value estimates. Compared with World Models, learning is joint/iterative and differentiable rather than CMA-ES over a small controller.

## Results and independent verdict

Figure 6 p. 6 reports mean score 823 after 5 M environment steps across 20 DMC tasks, compared with PlaNet 332 and D4PG786 after 100 M; five seeds. The comparison supports sample efficiency in this suite, not equal compute or language navigation performance. Representation/return-horizon ablations show contributions, but no formal off-distribution guarantee.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/danijar/dreamer. Snapshot commit: `56d4d444dfd0582b0e79dab80aebbea74c0ce40d`. Retrieval: 2026-10-22.

INSPECTED CODE: `dreamer.py:145–275` optimizes image/reward likelihoods and KL with free nats, then actor and value objectives over imagined latent trajectories using lambda returns and separate optimizers. `models.py:25–110` contains recurrent deterministic state plus diagonal-Gaussian stochastic posterior/prior; standard deviation is softplus plus 0.1. `dreamer.py:136–145` updates the posterior using a real observation before actor inference. `wrappers.py:1–80` supplies DeepMind Control images and action/reward handling; `dreamer.py:329–340,441–449` supplies normalization and evaluation/training separation. The model supports gradients through latent dynamics for actor optimization. This is inspected TensorFlow source, not an executed control experiment; historical dependencies and assets remain prerequisites.

Inspected files: `dreamer.py`, `models.py`, `wrappers.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

A policy can exploit model errors outside collected support; differentiable long rollouts can amplify model and value bias. Reward reconstruction can omit task-relevant details when future goals change. The learned policy removes online search cost but shifts cost into imagined training and an imperfect value model.

Relevant follow-up relationship: DreamerV2/V3 extend representation/scaling; NavMorph imports RSSM dynamics into VLN. CVAML later shows value-aware stochastic-model loss pitfalls; these are related but not a claim that original Dreamer uses every analyzed loss.

## Access and uncertainty record

Full text is preserved under `corpus/dreamer/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/dreamer/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-dreamer-planning_cost

**Shifted into another cost or limitation — Methodological.** Avoiding online candidate search requires learning an actor and value model on imagined trajectories.

Conditions: Tasks where latency at action selection matters but training compute is available. Cause: Dreamer amortizes PlaNet-style latent planning through actor/value optimization rather than solving the search anew at deployment.

Evidence: AUTHOR METHOD and PAPER EXPERIMENT: Sections 3–4 and Figure 6, PDF pp.3–6; mean score 823 after 5M steps on 20 DMC tasks. This does not establish equal compute.

Follow-up assessment: Dreamer addresses PlaNet online search cost; later Dreamer versions extend scaling. The cost moves into behavior learning and value approximation. Confidence: Medium.

### 02-dreamer-model_bias

**Still unresolved in the reviewed literature — Methodological.** Imagined actor improvement can exploit model or value errors outside collected support.

Conditions: Long imagined trajectories or policy updates into poorly observed regions. Cause: Policy gradients optimize returns supplied by learned dynamics and bootstrapped values; neither provides a general off-distribution correctness guarantee.

Evidence: DEDUCTION: Dreamer latent rollout/value objectives; World Models provides a direct exploitation example, while Dreamer benchmark gains do not isolate this failure mode.

Follow-up assessment: Iterative data collection and later Dreamer models mitigate distribution mismatch, but no reviewed evidence resolves all out-of-support goal-conditioned navigation errors. Confidence: Medium.
