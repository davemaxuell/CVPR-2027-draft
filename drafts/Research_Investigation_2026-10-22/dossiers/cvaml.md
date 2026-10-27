# CVAML

## Bibliography and reviewed material

**Calibrated Value-Aware Model Learning with Probabilistic Environment Models**. Claas A. Voelcker; Anastasiia Pedan; Arash Ahmadian; Romina Abachi; Igor Gilitschenski; Amir-Massoud Farahmand. ICML 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.mlr.press/v267/voelcker25a.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1-9,13-14,18-21,23. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- CVAML.pdf; Main PDF; SHA256 `61f0d5c22c30470bf9fcf84d0111d3fb195866684c6f88a2364dc53ba77babf1`; 24 pages.

## Problem, task and assumptions

Model-based RL with stochastic transitions: should a learned model match full transition distributions or only value-relevant expectations? A sample-based value loss can unintentionally suppress transition variance. This is adjacent theory/continuous control, not a VLN algorithm or confidence-calibration method.

## Method, training and inference

Sections 3–5: an (m,b)-VAML objective compares model value after m imagined transitions with a b-step bootstrapped real target. For independent real/model samples, Eq. 3 decomposes sample squared error into desired squared expected-value error plus model-value variance and target variance. The model variance term changes the optimum. CVAML subtracts an estimated model-value variance, requiring multiple model samples, to recover the intended expectation objective; finite-sample normalization must match the source implementation. The value learner is separately trained with ordinary TD (m=0) because stochastic MuZero-style imagined-target regression can fail even with perfect dynamics. Eq. 7 explains why latent feature MSE targets E[φ(s')] while value matching targets E[V (φ(s'))]; they commute for all distributions only with affine V. Proposition 5 gives existence of deterministic value-equivalent models under compact connected metrizable state space, continuous kernel/value and bijective latent mapping. It neither proves learnability nor guarantees arbitrary downstream rewards or policies. Training/inference otherwise inherit MuZero-style model-based RL agents; no picture generation is required. Appendix A.3, pp. 18–20, proves existence by continuity on a compact connected latent image and a measurable selection attaining the expected value. This supplies a measurable value-equivalent mapping for one fixed value function, not a continuous neural realization or a model preserving all language-conditioned values.

## Novelty and closest predecessors

Identifies a statistical bias in sampled value-aware model losses and supplies a correction, then separates model learning from value learning. Distinct from making visual predictions more realistic or estimating epistemic uncertainty. Closest predecessors VAML, MuZero and TD-MPC are explicitly analyzed rather than simply outscored.

## Results and independent verdict

Figures 2–3 pp. 6–8 evaluate 1000 randomly generated Garnet MDPs with bootstrap intervals and seven challenging DMC dog/humanoid tasks with 30 seeds. Improvements depend on setting; not a universal dominance theorem. The finite-MDP experiments test the identified bias more directly than aggregate DMC return. No R2R/RxR or physical navigation validation is provided. Figure 4, p. 9 explicitly aggregates 30 seeds per environment. Appendix Table 2, p. 21 specifies latent width 512, batch 128, discount 0.99, actor/critic/model learning rates 3e-4, encoder rate 1e-4, one-step model rollout, and one/four model samples. Appendix C.4, p. 23 shows the BYOL-style auxiliary loss is essential in its humanoid experiments: objective calibration alone is insufficient.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/adaptive-agents-lab/CVAML. Snapshot commit: `e522ef376992475e57982bccf489338e8ff0be7a`. Retrieval: 2026-10-22.

INSPECTED CODE: `mad_td/training_util/model_loss_functions.py:120–171` evaluates sampled model next states using the critic, subtracts a variance correction, and minimizes squared mismatch of sample-mean values. Lines 156–158 use population sample variance (`ddof=0`) divided by N. Under the restricted assumption of independent samples and a fixed value function, an unbiased correction would use unbiased sample variance divided by N; this implementation leaves a sigma²/N² residual. Its action may depend on the first sampled next state, so the simple independent-sample derivation is not a general diagnosis of training behavior. Lines 390–425 show objective dispatch and target-critic use. `config/algo/corrected_itervaml.yaml` uses four stochastic samples and retains likelihood-related model losses. `mad_td/trainers/mad_td_model.py:40–95` initializes encoder, actor, critic and latent-model train states from replay data. `mad_td/utils/eval_utils.py:25–50` evaluates on-policy one-step tuples using saved physics state. No JAX training or numerical reproduction of paper performance was executed; the estimator discrepancy is a scoped mathematical deduction requiring controlled comparison.

Inspected files: `config/algo/corrected_itervaml.yaml`, `mad_td/trainers/mad_td_model.py`, `mad_td/training_util/model_loss_functions.py`, `mad_td/utils/eval_utils.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Correction requires extra samples and can increase estimator variance; value equivalence is policy/reward dependent, so preserving current V can discard information needed after a language-goal change. Existence conditions do not establish an implementable globally correct navigation model. Calibration here means objective correction, not action-risk reliability or conformal coverage.

Relevant follow-up relationship: DINO-WM and WoMAP use feature/dynamics losses; NavQ predicts successor semantics. A proposal about goal-dependent value calibration must distinguish CVAML's known sample-bias correction from a new navigation-specific contribution.

## Access and uncertainty record

Full text is preserved under `corpus/cvaml/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/cvaml/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-cvaml-estimator

**Still unresolved in the reviewed literature — Implementation-specific.** Finite-sample variance correction can retain bias and raise sampling costs.

Conditions: Stochastic transitions with a finite number of next-state samples. Cause: The inspected estimator uses population variance divided by N; with iid samples and fixed value, that differs from unbiased sample variance divided by N. Sample-dependent action selection adds further dependencies.

Evidence: INSPECTED CODE: CVAML loss discussion and immutable links in 02_repository_audit.md. EXECUTED TEST: evidence/02/logic_check_results.json enumerates 16 N=4 ±1 samples; corrected mean loss 0.0625 versus approximately zero with unbiased variance. This is a restricted arithmetic counterexample, not a benchmark result.

Follow-up assessment: No inspected follow-up supplies a correction for this exact estimator and dependency structure. Confidence: High for the stated iid fixed-value counterexample; uncertain for benchmark impact.

### 02-cvaml-goal

**Uncertain because evidence is insufficient — Methodological.** Value equivalence for one value function does not preserve all future instruction-dependent distinctions.

Conditions: Changing language goals or rewards after learning a task-specific latent model. Cause: The objective preserves expected values for its selected value function, allowing state distinctions irrelevant to that function to disappear.

Evidence: DEDUCTION: main objective and Appendix proof, PDF pp.18–20; the existence argument assumes a compact connected setting and a suitable value mapping. Appendix C.4 p.23 shows auxiliary BYOL loss remains important.

Follow-up assessment: NavQ predicts goal-dependent semantic successors; WoMAP learns detector-derived rewards. Neither establishes universal goal-change preservation or action-risk calibration. Confidence: Medium.

Independent isolated check (**EXECUTED TEST**): The isolated iid fixed-value finite-sample variance correction check finds expected residual loss 0.0625 for N=4 Rademacher samples; unbiased sample variance correction is approximately zero. This is not a CVAML benchmark run. Setup and output: `evidence/02/logic_checks.py` and `evidence/02/logic_check_results.json`.
