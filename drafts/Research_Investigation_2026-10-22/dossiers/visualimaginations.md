# Visual Imaginations for VLN

## Bibliography and reviewed material

**Do Visual Imaginations Improve Vision-and-Language Navigation Agents?**. Akhil Perincherry; Jacob Krantz; Stefan Lee. CVPR 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/CVPR2025/papers/Perincherry_Do_Visual_Imaginations_Improve_Vision-and-Language_Navigation_Agents_CVPR_2025_paper.pdf. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-7. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Visual Imaginations for VLN.pdf; Main PDF; SHA256 `4e1e240f8237cbdf5446a25aebb29bfbf39a28ea355bc4493f4cd7d8494668d9`; 10 pages.

## Problem, task and assumptions

Add generated visual landmark exemplars to HAMT/DUET R2R inputs. These are text-conditioned prototypes, not predictions of an unseen actual camera state.

## Method, training and inference

Section 3 pp. 3–4: FG-R2R segments yield spaCy noun phrases filtered by a manual exclusion list; SDXL renders real-estate-style landmark imagery. ViT features, type embeddings and a three-layerMLP align images with text; imagination tokens join instruction tokens in HAMT and DUET branches. Auxiliary cosine alignment to noun-phrase embeddings (Eq. 2) and three-stage finetuning train new projections, then backbone at lowLR, then jointly. Deployment consumes cached/generated instruction images without an action-conditioned transition model. p. 5 reports 41,558 training images,1024 square and 3.2 sec/image onH100.

## Novelty and closest predecessors

Explicit visual prototypes aid linguistic grounding, contrasting with VLN-SIG prediction supervision and Pathdreamer environment-conditioned rendering.

## Results and independent verdict

Table 1 p. 5 HAMT unseen 66.24/61.51→67.26/62.02, test 65/60 unchanged; DUET unseen 71.52/60.41→72.12/60.48, test 69/59→71/60. Modest improvements require matched multi-seed assessment. Generator pretraining and preprocessing add data/compute. Table 3, p. 6: removing imaginations at test time still gives SR 66.92 versus baseline 66.24, while correct imaginations give 67.26 and wrong ones 66.24. This directly supports a training regularization component plus a smaller aligned-input benefit. Table 6, p. 7 compares no auxiliary loss (66.75 SR), InfoNCE (67.18) and cosine alignment (67.26); the loss choice has modest effects. Selected attention visualizations explicitly do not establish causality.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/akhilperincherry/VLN-Imagine. Snapshot commit: `31816e52e9913e9993a5ab05cf1e99b56044e84a`. Retrieval: 2026-10-22.

INSPECTED CODE: `VLN-DUET/map_nav_src/models/vilmodel.py:562–655` adds an imagination type embedding and aligns projected image features with mean noun-phrase embeddings by one minus cosine similarity. Lines 1154–1166 concatenate imagination and text tokens for both global/local encoders. `VLN-DUET/map_nav_src/r2r/agent.py:317–382` loads cached instruction-keyed image features and masks; 412–449 constructs alignment inputs and 621–623 adds its weighted auxiliary loss. Thus deployment uses static text-conditioned prototypes rather than predicting the actual next view conditioned on a candidate action. The repository contains broader HAMT/DUET training/evaluation scaffolds; all scripts, generated-image provenance and multi-stage optimizer flags were not exhaustively checked, and no benchmark was run.

Inspected files: `VLN-DUET/map_nav_src/models/vilmodel.py`, `VLN-DUET/map_nav_src/r2r/agent.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Prototype specificity can mislead unusual appearances; parser/manual exclusions restrict coverage. Images add semantic priors but cannot reveal hidden physical facts. Gains may reflect regularization/priming rather than grounded foresight.

Relevant follow-up relationship: MonoDream/FantasyVLN shift prediction into training. Compare prototype grounding separately from actual-state and action-conditioned prediction.

## Access and uncertainty record

Full text is preserved under `corpus/visualimaginations/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/visualimaginations/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-visualimaginations-attribution

**Uncertain because evidence is insufficient — Insufficiently tested.** Generated landmark prototypes may improve representation learning without faithfully predicting the current hidden scene.

Conditions: Unusual object appearances or claims that foresight causes the navigation gain. Cause: Images are semantic prototypes derived from instruction phrases, not action-conditioned scene transitions; training losses remain beneficial when prototypes are removed at test time.

Evidence: PAPER EXPERIMENT: Table 3 p.6 correct/none/wrong imagination SR 67.26/66.92/66.24 versus baseline 66.24; selected attention maps do not establish causality.

Follow-up assessment: MonoDream/FantasyVLN also exploit auxiliary prediction. Separate prototype alignment from actual-state prediction in any comparison; no causal deployment mediation test was reviewed. Confidence: Medium.
