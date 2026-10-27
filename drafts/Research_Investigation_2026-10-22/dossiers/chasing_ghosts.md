# Chasing Ghosts

## Bibliography and reviewed material

**Chasing Ghosts: Instruction Following as Bayesian State Tracking**. Peter Anderson; Ayush Shrivastava; Devi Parikh; Dhruv Batra; Stefan Lee. NeurIPS 2019 (2019). Research cutoff: **2026-10-22**.

Primary publication/source: Supplied PDF; see local file and SHA256 in corpus metadata. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Chasing Ghosts.pdf; Main PDF; SHA256 `ac33b9905932abb1ceba992e8cf47f36bcb379877449d0922d58f573f6a29548`; 11 pages.

## Problem, task and assumptions

Follow an R2R route using RGB-D panoramic views, camera calibration and known pose. The 'ghost' is an imagined demonstrator whose instruction-conditioned location is inferred; it is not a generated future camera image or the physical agent's uncertain localization.

## Method, training and inference

Sections 3–4 pp. 3–6: CNN features are projected using depth into a ground-plane map M in R^(C×Y×X), updated by sparse convolutional GRU. An LSTM instruction decoder emits successive action and observation descriptions. A differentiable histogram Bayes filter represents ghost pose (x,y,heading). The motion update convolves the previous belief with instruction/map-dependent normalized kernels (Eqs.3–4); a LingUNet observation likelihood multiplies this prior and the posterior is renormalized (Eq. 5). Six inner filter steps infer the whole instruction trajectory and are rerun whenever the real agent expands its map. A separate MLP scores observed graph nodes using belief neighborhood, distance and visited flag; selecting a visited node stops. Training minimizes KL to demonstrated instruction-trajectory beliefs plus teacher-action cross-entropy, with a 50:50 teacher/policy mixture; visual CNN is frozen. Inference repeatedly maps real views, reinfers the ghost's endpoint and chooses reachable nodes. No online model gradients or RGB synthesis.

## Novelty and closest predecessors

Turns instruction following into structured probabilistic localization of a hypothetical demonstrator, combining geometric mapping, language-conditioned motion and observation models. Relative to LingUNet endpoint prediction, explicitly represents sequence and orientation. It is prior art for multimodal navigation beliefs, not empirical probability calibration.

## Results and independent verdict

Table 1 p. 7: unseen goal localization 44.9% versus LingUNet 32.2%; removing heading yields 39.4%. This supports the filtering representation. Table 2 p. 8: complete navigation unseen SR 35/SPL 31 and test 33/30; the complete agent is weaker than strong contemporary navigation baselines. Better localization does not establish better control. Standard R2R split, no independent run.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/batra-mlp-lab/vln-chasing-ghosts. Snapshot commit: `f819aea21b94d9d3e23d9b6b9264054ee50c007b`. Retrieval: 2026-10-22.

INSPECTED CODE: `tracker/modules/navigators.py:142–189` builds spatial and language-conditioned motion kernels, renormalizes their product, and propagates log belief. Lines 279–286 combine motion with language-conditioned observation likelihood. `tracker/modules/loss.py:20–36` implements KL supervision against target belief maps. `tracker/train_filter.py:181–262` creates target heatmaps from reference coordinates and uses truncated backpropagation; lines 51–103 validate prediction on supervised paths. This confirms that the filter predicts the instruction's ideal agent, not a learned photorealistic transition model. The repository also supplies policy and evaluator modules, but their complete checkpoint-level reproduction and preprocessing dependency chain were not executed; source inspection here is strongest for the learned filter rather than the complete deployed policy.

Inspected files: `tracker/modules/loss.py`, `tracker/modules/navigators.py`, `tracker/train_filter.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Known accurate poses/depth and discretized ground-plane map restrict transfer; fixed six ghost steps compress variable instructions. Bayesian normalization does not validate learned likelihood calibration or independence assumptions. The strong goal-localization versus weaker end-to-end navigation result exposes a planning/control interface bottleneck. Later topological systems improve navigation, but do not directly test whether a persistent multimodal belief helps under ambiguous instructions.

Relevant follow-up relationship: DUET/topological mapping changes the controller and representation; HNR and SALI add predicted future content. A direct matched comparison preserving multimodal belief versus collapsing it remains absent in this reviewed chain.

## Access and uncertainty record

Full text is preserved under `corpus/chasing_ghosts/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/chasing_ghosts/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-chasing_ghosts-control

**Uncertain because evidence is insufficient — Methodological.** Improved instruction-goal localization does not translate into competitive complete navigation.

Conditions: R2R with RGB-D panoramas, supplied poses and discrete reachable graph nodes. Cause: The ghost endpoint belief is consumed by a separate local graph-node controller; localization quality alone does not solve route selection and stopping.

Evidence: PAPER EXPERIMENT: Tables 1–2, PDF pp.7–8: localization 44.9% versus 32.2%, but unseen navigation SR/SPL 35/31. DEDUCTION: interface bottleneck rather than measured causal attribution.

Follow-up assessment: DUET/topological planners change control and representation; HNR/SALI add predicted content. No matched belief-versus-controller intervention was reviewed. Confidence: Medium.

### 02-chasing_ghosts-belief

**Uncertain because evidence is insufficient — Insufficiently tested.** Normalized ghost beliefs have no demonstrated calibration under pose error or ambiguous instructions.

Conditions: Variable-length instructions, incorrect pose/depth or multiple plausible instruction routes. Cause: Learned motion/observation likelihoods, a fixed six-step ghost rollout, and normalized products do not establish valid likelihood independence or coverage.

Evidence: DEDUCTION: Sections 3–4, PDF pp.3–6; tracker/modules/navigators.py:142–189,279–286 and tracker/modules/loss.py:20–36 at the dossier commit.

Follow-up assessment: Later map-based and imagined-view papers in this assignment do not report matched posterior-coverage tests. Confidence: Medium.
