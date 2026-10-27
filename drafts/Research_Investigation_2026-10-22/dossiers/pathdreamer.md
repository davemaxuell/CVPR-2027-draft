# Pathdreamer

## Bibliography and reviewed material

**Pathdreamer: A World Model for Indoor Navigation**. Jing Yu Koh; Honglak Lee; Yinfei Yang; Jason Baldridge; Peter Anderson. ICCV 2021 (2021). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/ICCV2021/papers/Koh_Pathdreamer_A_World_Model_for_Indoor_Navigation_ICCV_2021_paper.pdf. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Pathdreamer.pdf; Main PDF; SHA256 `28fc51b1bfe0ad4909f95d457ffd10ab09bc188d8846f9413777579a3c18ad3b`; 11 pages.

## Problem, task and assumptions

Predict indoor panoramic RGB/depth/semantics at a specified future pose from observed RGB-D, semantic labels and pose history. Downstream planning assumes known future navigation graph geometry.

## Method, training and inference

Section 3 pp. 3–5: accumulate a 3D point cloud and reproject into candidate pose. Stochastic RedNet structure generator inpaints semantics/depth with 32-channel Gaussian latent, conditional prior and ground-truth-conditioned posterior. Semantic CE/depth L1/KL weights 1/100/.5. Stage 2 MultiSPADE generates RGB conditioned on completed structure, RGB guidance and prior frame, using GAN hinge, perceptual and feature-matching losses. Stages train separately; RGB training receives ground-truth structure. About 400 k random 5–8 node MP3D trajectories; structure 512×256, RGB 1024×512. Quantitative inference uses prior mean. Beam search ranks imagined trajectories by instruction compatibility, executes first action and replans from a new real observation.

## Novelty and closest predecessors

Hierarchical geometry-guided stochastic completion replaces repeated-current-view lookahead. Prediction is independent of a specific policy and distinct from auxiliary-training-only semantics.

## Results and independent verdict

Table 1 p. 7 unseen semantic mIoU 78.3 one-step versus 50.8 over 1–6 steps. Table 3 p. 8 horizon 3 navigation SR 50.4/SPL 47.3 versus repetition 40.6/37.7 and oracle future images 59.3/55.8. The ground-truth future navigation graph is provided throughout: visual oracle-gap reduction is not evidence of planning without oracle geometry.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/google-research/pathdreamer. Snapshot commit: `351cc2ff79e079d8e280aff95666e2323eb4ee9b`. Retrieval: 2026-10-22.

INSPECTED CODE: `models/pathdreamer_models.py:193–258` projects RGB/depth/semantics into persistent point-cloud memory; lines 260–377 reproject into a requested pose, complete structure, synthesize RGB and optionally add predictions back into memory. The default `add_preds_to_memory=False` is an explicit provenance choice. `models/point_cloud_models.py:205–220` creates Gaussian prior/posterior modules; `models/pathdreamer_config.py:30–38` sets panoramic dimensions and circular padding. README provides two pretrained checkpoints and an inference Colab. The inspected release contains inference/model tests, but no full training launcher or downstream beam-search navigation implementation in its source listing; therefore the paper's training losses and navigation experiments cannot be fully reproduced from this repository alone. No checkpoint was downloaded or run.

Inspected files: `README.md`, `models/pathdreamer_config.py`, `models/pathdreamer_models.py`, `models/point_cloud_models.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Hallucinated structure compounds with horizon; ground-truth structure during RGB training introduces stage mismatch. Multi-stage high-resolution synthesis costs compute; stochasticity and GAN realism do not imply calibrated uncertainty.

Relevant follow-up relationship: DREAMWALKER predicts future waypoints, HNR renders features cheaply, SALI stores imagined nodes. These partly address geometry/computation/persistence but do not verify hidden-world truth.

## Access and uncertainty record

Full text is preserved under `corpus/pathdreamer/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/pathdreamer/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-pathdreamer-geometry

**Partially addressed — Methodological.** Image-generation gains were evaluated with a privileged future navigation graph.

Conditions: Interpreting navigation results as planning through completely unknown geometry. Cause: The navigation evaluation supplies future graph connectivity while replacing only future visual observations.

Evidence: PAPER EXPERIMENT: Table 3 p.8 horizon-three SR 50.4 versus repetition 40.6 and oracle 59.3; evaluation explicitly uses ground-truth future graph.

Follow-up assessment: DREAMWALKER predicts waypoints and HNR evaluates continuous candidates, partly reducing the oracle-graph assumption while introducing waypoint/geometry prediction error. Confidence: Medium.

### 02-pathdreamer-horizon

**Partially addressed — Methodological.** Generated structure degrades over successive unseen views.

Conditions: One to six synthesized steps into occluded geometry. Cause: Autoregressive structure hallucinations accumulate; training RGB generation with ground-truth structure adds stage mismatch.

Evidence: PAPER EXPERIMENT: Table 1 p.7 unseen semantic mIoU 78.3 one-step versus 50.8 across one to six steps.

Follow-up assessment: HNR reduces pixel-generation cost and SALI preserves imagined nodes; neither validates the truth of long-horizon occluded structure under equal sensing. Confidence: Medium.
