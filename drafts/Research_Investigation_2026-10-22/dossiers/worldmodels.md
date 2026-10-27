# World Models

## Bibliography and reviewed material

**Recurrent World Models Facilitate Policy Evolution**. David Ha; Jürgen Schmidhuber. NeurIPS 2018; related World Models arXiv version (2018). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.neurips.cc/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-6. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- World Models.pdf; Main PDF; SHA256 `3f12aec600996c893b9d8f4610f302d2604f28a880d98abf6e7883c46cb0d4b3`; 13 pages.

## Problem, task and assumptions

Learn compact visual models and evolve a small controller; test whether a controller can learn in a hallucinated simulator and transfer to a real environment. CarRacing and VizDoom are not instruction-navigation benchmarks.

## Method, training and inference

Sections 2–4: VAE maps pixels to z; MDN-RNN predicts a Gaussian-mixture distribution over next z from current z, action and recurrent h. A small linear controller maps concatenated[z,h] to actions and is optimized by CMA-ES with episodic reward. VAE/RNN are pretrained separately on 10,000 random rollouts without controller reward. In CarRacing, controller evolution evaluates real environment trajectories using compact features. In VizDoom, a learned termination predictor and MDN-RNN form a simulator in which the controller is evolved before real transfer. Sampling temperature controls stochasticity. It is not per-step online imagination search and the two experiments use different training deployment arrangements.

## Novelty and closest predecessors

Demonstrates compact model-based representation and entirely dreamed controller training with transfer, using separable perception/memory/control. The architecture is simple enough to expose exploitation of the learned simulator directly.

## Results and independent verdict

Table 1 p. 4 CarRacing fullVMC906±21 versus vision-only 632±251 over 100 trials. Table 2 p. 4 Doom at temperature .1: virtual 2086 versus actual 193 survival steps; at 1.15 virtual 918 versus actual 1092. Section 4.2 p. 6 explicitly documents controller exploitation of model deficiencies. This is concrete evidence against treating visually plausible simulated success as physical success.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/hardmaru/WorldModelsExperiments. Snapshot commit: `fd982b9691a941b52c6addbde29bc801ca6202c8`. Retrieval: 2026-10-22.

INSPECTED CODE: the article website links the actual author implementation, `hardmaru/WorldModelsExperiments`, inspected at the separate immutable snapshot recorded here. `carracing/vae/vae.py:14,64–87` confirms a 32-dimensional VAE with reconstruction MSE plus KL floor and Adam. `carracing/rnn/rnn.py:39–45,139–168` confirms 256 recurrent units, five Gaussian mixtures and negative log likelihood of next latents conditioned on actions. `carracing/series.py:35–39` normalizes 64×64 RGB and samples VAE latents. `carracing/model.py:98–114,154–239` implements the controller and episode evaluation; `carracing/train.py:89–93,354–360` implements CMA-ES and evaluation. `doomrnn/doomrnn.py:24,624–655` samples the learned simulator with temperature 1.25 and learned termination. This release setting differs from the paper's highlighted 1.15 setting; reproduce the intended configuration explicitly. No historical TensorFlow/Gym code or model checkpoint was executed.

Inspected files: `carracing/model.py`, `carracing/rnn/rnn.py`, `carracing/series.py`, `carracing/train.py`, `carracing/vae/vae.py`, `doomrnn/doomrnn.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Policy-induced distribution shift exposes world-model inaccuracies; low-temperature models can be exploited. Increasing temperature is a useful empirical heuristic, not calibrated uncertainty or a general robustness proof. Random pretraining may omit important actions/outcomes; the small controller also limits expressiveness.

Relevant follow-up relationship: PlaNet/Dreamer iteratively collect policy experience and learn stronger latent behavior; CVAML clarifies value-aware model objectives. The exploitation bottleneck survives architecture changes and requires evidence from actual outcomes.

## Access and uncertainty record

Full text is preserved under `corpus/worldmodels/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/worldmodels_implementation/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-worldmodels-exploitation

**Partially addressed — Methodological.** A controller can achieve high imagined reward by exploiting a poorly supported world model.

Conditions: Low-temperature learned Doom dynamics and policy-induced distribution shift. Cause: Random-policy training undercovers controller-induced states, and the controller searches weaknesses of the learned transition/reward process.

Evidence: PAPER EXPERIMENT: Table 2 p.4 temperature .1 gives virtual/real survival 2086/193; at 1.15, 918/1092. Section 4.2 p.6 explicitly discusses exploitation.

Follow-up assessment: PlaNet/Dreamer collect policy experience iteratively and use stronger latent training. This mitigates mismatch but does not prove robustness of unsupported imagined trajectories. Confidence: Medium.
