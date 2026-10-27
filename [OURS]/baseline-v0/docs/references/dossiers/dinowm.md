# DINO-WM

## Bibliography and reviewed material

**DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning**. Gaoyue Zhou; Hengkai Pan; Yann LeCun; Lerrel Pinto. ICML 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://dino-wm.github.io/. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-7. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- DINO-WM.pdf; Main PDF; SHA256 `f054a2c03888aa9a50b7747fb99a168f2979cb7fa216b075011b089d45b628db`; 21 pages.

## Problem, task and assumptions

Offline action-labeled visual trajectories and a goal image. Zero-shot means planning toward new goals without reward-specific retraining; the transition model itself is trained. No language-route following is tested.

## Method, training and inference

Section 3 pp. 3–5: frozen DINOv2 encodes an image into N spatial patches of width E, z∈R^(N×E). Embedded actions are concatenated with each patch; a Transformer with frame-causal attention predicts next patch features. Teacher-forced future-feature MSE (Eq. 1) trains the model; optional proprioception is concatenated. An independent pixel decoder can be learned with reconstruction MSE (Eq. 2), but planning does not need it. Inference uses receding-horizon CEM: sample action sequences, roll out latent predictions, score terminal feature MSE to the goal, refit elite action distributions, execute the first action and replan using a real image. Gradient-based planning is evaluated but weaker. Memory and compute scale with patch count, context and sampled rollout count rather than decoded image size.

## Novelty and closest predecessors

Uses frozen dense semantic visual features as a directly predictable state, avoiding reward learning and jointly learned pixel reconstruction; contrasts with PlaNet/Dreamer and global feature embeddings. Spatial patch ablations support the representation choice, rather than establishing universal pretrained-feature sufficiency.

## Results and independent verdict

Table 1 p. 6: Maze/Wall/Reach/PushT SR .98/.96/.92/.90; comparison DreamerV3, retrained without reward,1/1/.64/.30. TD-MPC2 gets 0 in this reward-free setup, which is a task adaptation limitation rather than evidence the original reward-trained algorithm is universally inferior. Deformable Rope/Granular Chamfer distances .41/.26 versus Dreamer 2.49/1.05; sample sizes 50 for first four tasks and 10 for deformables. Table 2 patch versus CLS PushT .90 versus .44 supports spatial representation.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/gaoyuezhou/dino_wm. Snapshot commit: `0a9492fa12044b852ae9e001cc74604b79c8bb0c`. Retrieval: 2026-10-22.

INSPECTED CODE: `models/visual_world_model.py:1–180,190–309` encodes images into DINO patch features, concatenates/replaces action tokens, and trains prediction against detached future embeddings. The optional decoder has a separate reconstruction role. `planning/cem.py:106–118` samples candidate actions, calls world-model rollout and refits elites; it does not require a pixel decoder for planning. `conf/train.yaml` freezes the encoder and uses a three-frame history, one-step prediction and predictor learning rate 5e-4. `planning/evaluator.py:112–163` denormalizes actions and evaluates actual environment rollouts. Model/loss, planning, configuration and evaluator paths were inspected; all task-specific dataset preprocessing and checkpoints were not executed or exhaustively audited.

Inspected files: `conf/train.yaml`, `models/visual_world_model.py`, `planning/cem.py`, `planning/evaluator.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Feature similarity need not equal task success; aliasing and counterfactual coverage can yield planner exploitation. A deterministic feature predictor averages multimodal outcomes. Repeated CEM rollouts impose compute, and no language, dynamic-obstacle or calibrated-confidence guarantee follows. These are methodological deductions from the representation and evaluation scope.

Relevant follow-up relationship: WoMAP adds language-conditioned distilled rewards and VLM action proposals. NWM changes latent representation to video generation. PiJEPA is a current language-conditioned planning lead for assignments 05–06.

## Access and uncertainty record

Full text is preserved under `corpus/dinowm/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/dinowm/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-dinowm-language

**Partially addressed — Methodological.** Image-goal feature planning does not itself provide language-goal reward semantics.

Conditions: Tasks specified by text rather than an available target image. Cause: The planner optimizes distance to a specified visual target in frozen DINO feature space.

Evidence: AUTHOR METHOD: Section 3 and task evaluation, PDF pp.3–6; Table 2 shows patch features help PushT but does not validate language understanding.

Follow-up assessment: WoMAP adds VLM proposals and language-conditioned distilled object rewards, with physical object-reaching tests. This adds detector and VLM errors and is not broad instruction-following validation. Confidence: Medium.

### 02-dinowm-exploitation

**Uncertain because evidence is insufficient — Methodological.** Feature similarity can reward predicted outcomes that are not achievable or task-correct.

Conditions: Out-of-support actions, aliased views or multimodal futures during CEM search. Cause: A deterministic feature predictor and feature-distance goal objective need not preserve all physical constraints; optimizing them can select model errors.

Evidence: DEDUCTION from Section 3 and inspected planner; Table 1 task successes do not test calibrated confidence or adversarial model exploitation.

Follow-up assessment: WoMAP extends the reward interface but retains optimizer/model/reward error; no matched intervention establishes resolution. Confidence: Medium.
