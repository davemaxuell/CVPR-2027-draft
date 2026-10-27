# SC2-WM

## Bibliography and reviewed material

**SC²-WM: A Self-Correcting World Model with Closed-Loop Feedback for Vision-and-Language Navigation in Continuous Environments**. Xuan Yao; Yuze Zhu; Junyu Gao; Zongmeng Wang; Changsheng Xu. ICML 2026 (2026). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.mlr.press/v306/yao26u.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1-8,13-18. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: SC2-WM (arXiv).pdf: all 18 pages compared at the extracted-token level. Only the first-page arXiv stamp differs from the main PDF; method, tables and supplement text are identical.

- SC2-WM.pdf; Main PDF; SHA256 `e77f42e075484b5b78756bdff7bfecc4222b98d60e47f18247b6521a472a9d7a`; 18 pages.
- SC2-WM (arXiv).pdf; Alternate / duplicate PDF; SHA256 `c616d4c32548ab67086b264e49ad8fd5bfddf84f023af3ccd305c4873342f90b`; 18 pages.

## Problem, task and assumptions

RGB-D VLN-CE under partial observability and scene shift. The agent predicts waypoints, maintains a topological map and selects actions from an instruction plus observation history. Panoramic and monocular configurations have materially different sensory access.

## Method, training and inference

Sections 3.1–3.3, pp. 3–6: visual features first retrieve recent episodic features using attention with relative spatial/time cues and temporal decay; a gate blends current and retrieved representations. An observation-conditioned posterior updates latent s_t. A provisional policy proposes an action, and a Transformer prior predicts its next latent mean. Equation 9 defines Δ_t = predicted s_(t+1) − s_t. A learned residual/gating network ψ(s_t,Δ_t) produces a refined current state, from which the final policy selects the executed action. This Δ is an endogenous predicted state change, not an observed next-state prediction error, and the paper explicitly says it is not a target to minimize. Training combines teacher action cross-entropy on original/refined states (Eq. 11), posterior/prior KL (Eq. 12), and decoded future-feature squared error (Eq. 13). Test-time adaptation compares provisional versus final action distributions by KL, uses a running percentile plus a low-entropy condition, and updates world-model parameters through reconstruction loss (Eq. 14); policy parameters are excluded according to the text. Thus there are separate immediate inference refinement and gradient adaptation mechanisms.

## Novelty and closest predecessors

Relative to NavMorph's forward memory updates and HNR's predicted feature tree, SC²-WM adds provisional-action latent refinement and conditional gradient adaptation. These are combinations of recurrent world modeling, episodic visual calibration and test-time optimization; they do not establish statistical calibration or a causal correctness guarantee.

## Results and independent verdict

PAPER EXPERIMENT: Table 3 p. 8, monocular R2R-CE val-unseen: reproduced base SR 43.77/SPL 29.39; state correction 48.34/33.02; plus visual calibration 49.86/36.05; full 50.90/37.17; fixed-interval adaptation 50.30/36.58; removing reconstruction 48.89/34.58. These ablations support the combined recipe and modest trigger benefit, but do not isolate whether Δ is reliable under confidently wrong dynamics. Table 1 p. 7 HNR panoramic base 61/51 to 66/54, test 58/50 to 62/53. Training is reported on one RTX3090, under 40 hours (p. 8); no runtime independently measured.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/sunrise-yuze/SC2_WM. Snapshot commit: `f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b`. Retrieval: 2026-10-22.

INSPECTED CODE: `vlnce_baselines/models/etp/vilmodel_cmt.py:598–621,934–967` implements a 2304→1536→768 MLP over navigation, mixed-candidate and raw-candidate features. The immediate refinement uses detached reconstructed features and a learned residual/gate. This is not a literal implementation of the paper's Gaussian posterior/prior RSSM equations. Crucially, `vlnce_baselines/ss_trainer_ETP.py:2246–2308` ALSO compares a previous predicted candidate feature with the next real panoramic observation using one minus cosine similarity. This delayed observed residual drives test-time adaptation; a claim that SC²-WM never corrects using observed outcomes is false. It also has drift and percentile-KL controls. The active branch does not show the paper's entropy gate at this location. `TTA_setup_trainable_params` at 1030–1035 enables all global-encoder parameters, including action/cross-modal components, whereas its nearby docstring describes a narrower component. Lines 2325–2371 recompute features and replace graph embeddings after adaptation. The shipped R2R configuration uses one environment, 30,000 iterations, learning rate 1e-5, adaptation learning rate 6e-7 and zero drift-decay coefficient. Its default backtracking option is teleport, so a reproduction must explicitly select and document the continuous-control protocol. README documents assets and the SS-ETP trainer. No checkpoint execution establishes which released branch produced each published table. The code supports delayed residual adaptation, but does not justify treating endogenous immediate refinement as an observed prediction error.

Inspected files: `README.md`, `run_r2r/iter_train.yaml`, `vlnce_baselines/models/etp/vilmodel_cmt.py`, `vlnce_baselines/ss_trainer_ETP.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

The feedback signal can be large for a correct motion and small for an incorrect but stationary latent prediction; low action entropy does not imply correctness. Algorithm 2 p. 6 uses d (predicted s_(t+1)) versus current calibrated v_t, whereas Eq. 13 and prose use next observation v_(t+1). Timing/target discrepancy requires source-level resolution before implementation. Also distinguish dynamic percentile described in text from fixed τ notation in Algorithm 2. These are methodological/reproducibility uncertainties, not demonstrated benchmark failures. The supplement and released code resolve the broad feedback concern: a delayed comparison with actual observations does exist. Remaining uncertainty concerns which parameters/trigger are used, prediction-target alignment and confidence under shift, not the absence of observed residual correction.

Relevant follow-up relationship: NavMorph, HNR, UNeMo; outside-corpus WorldMAP and PiJEPA are relevant current leads, not yet read in this dossier.

## Access and uncertainty record

Full text is preserved under `corpus/sc2wm/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/sc2wm/`. Both supplied 18-page versions include the same substantive supplement. Appendix pp. 15–16 explicitly defines delayed forecast-to-real-observation alignment (FRL); Table 6 gives SR 49.86 without adaptation, 49.48 for entropy minimization, 50.19 asynchronous and 50.90 FRL. Reported runtime rises from 17.78 to 18.92 seconds. Pages 17–18 report 500 physical robot runs (100 trials for each of five configurations), a 1.5 m success criterion and offboard/asynchronous processing; these are author experiments, not reproduced. The immediate endogenous delta is not the same quantity as this delayed residual. Source discrepancies are recorded in the implementation audit.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-sc2wm-feedback_timing

**Resolved within a clearly specified setting — Implementation-specific.** The broad ambiguity about whether correction uses any real next observation is resolved by the released asynchronous TTA branch.

Conditions: Interpreting immediate endogenous latent delta versus delayed deployment adaptation. Cause: Main-paper notation mixes current/next targets; immediate state refinement and delayed TTA use different signals.

Evidence: INSPECTED CODE: ss_trainer_ETP.py:2246–2308 compares prior predicted candidate features with the next real panorama using cosine residual; supplement PDF pp.15–16 describes FRL and Table 6. Immediate Eq.9 delta remains predicted next latent minus current latent.

Follow-up assessment: The paper supplement and official release resolve existence/timing for this inspected branch. This does not certify correctness of all formulas or confidence calibration. Confidence: High for inspected branch only.

### 02-sc2wm-calibration

**Uncertain because evidence is insufficient — Insufficiently tested.** Residual-triggered adaptation is not yet established to improve decisions under confidently wrong dynamics or residual misalignment.

Conditions: Domain shifts, ambiguous action selection, prediction/observation mismatch, or entropy confidently favoring a wrong action. Cause: Internal latent delta is not prediction error; delayed feature residual and percentile thresholds are proxies, and the active code updates a broader global encoder than a world-model-only reading suggests.

Evidence: PAPER EXPERIMENT: Table 3 p.8 full SR 50.90 versus fixed-interval 50.30; supplement Table 6 no adaptation 49.86, entropy 49.48, asynchronous 50.19, FRL 50.90. INSPECTED CODE: TTA_setup_trainable_params:1030–1035 and ss_trainer_ETP.py:2246–2308.

Follow-up assessment: WorldMAP and PiJEPA are current primary-source leads reserved for follow-up audit; no reviewed matched residual-to-action-risk validation resolves this concern. Confidence: Medium.
