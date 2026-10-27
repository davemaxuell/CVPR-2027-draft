# ImagineNav

## Bibliography and reviewed material

**ImagineNav: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination**. Xinxin Zhao; Wenzhe Cai; Likun Tang; Teng Wang. ICLR 2025 (2025). Research cutoff: **2026-10-22**.

Primary publication/source: Supplied PDF; see local file and SHA256 in corpus metadata. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- ImagineNav.pdf; Main PDF; SHA256 `82a38aefb15ef778b7e6173c57dc1860148da5a5d6ba986dada21391efdc7734`; 15 pages.

## Problem, task and assumptions

Open-vocabulary ObjectNav on HM3D/HSSD using a VLM without object-specific fine-tuning. The complete pipeline still contains trained waypoint and PointNav modules.

## Method, training and inference

Section 3: Where2Imagine uses ResNet18 and human ObjectNav demonstrations to predict displacement/orientation proposals. Six panoramic RGB directions and depth filter candidate views using wall-distance and rotation constraints. PolyOculus novel-view synthesis renders candidates; GPT-4o mini chooses an indexed image for the target object; pretrained VER PointNav executes and replans after arrival. Training includes Habitat-Web demonstrations. No persistent semantic map is maintained. The full system requires depth, despite introductory RGB-only descriptions.

## Novelty and closest predecessors

The VLM chooses among predicted candidate views supported by learned waypoint proposals. This differs from prompting only on currently observed views.

## Results and independent verdict

Table 1, p. 7: HM3D SR 53/SPL 23.8 versus VLFM 52.5/30.4; HSSD 51/24.9. Oracle views raise SR to 62/59. Table 2, p. 8 compares learned versus fixed waypoints on 100 episodes: 56/24.3 versus 49/23.3. That smaller ablation differs from the 200-episode headline setting. Better success is not necessarily better path efficiency.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: No verified official implementation located in the recorded search.. Snapshot commit: `not available`. Retrieval: 2026-10-22.

No verified official implementation of the original paper was located. The later ImagineNav++ project links a repository whose recorded README says code is coming; it is not evidence of an available original implementation.

Inspected files: No implementation files inspected; access scope is stated above. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

NVS domain shift and proposal constraints limit useful branching. Mapless local choices can revisit regions. Open-vocabulary is not training-free, and predictor realism is not calibrated outcome reliability.

Relevant follow-up relationship: ImagineNav++ (arXiv 2512.17435) adds selective foveation and memory: https://200203z.github.io/imaginenav-plus/ . Its recorded repository remains a forthcoming-code page. WMNav instead stores heuristic curiosity scores.

## Access and uncertainty record

Full text is preserved under `corpus/imaginenav/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/imaginenav/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-imaginenav-memory

**Uncertain because evidence is insufficient — Methodological.** Local mapless imagination can revisit regions and spend path length on unproductive choices.

Conditions: Long object-goal searches with repeated visually similar regions. Cause: Limited persistent spatial memory and proposal constraints restrict global exploration history.

Evidence: PAPER EXPERIMENT: Table 1 p.7 HM3D SR/SPL 53/23.8 versus VLFM 52.5/30.4. DEDUCTION: lower efficiency is compatible with, but does not prove, revisitation as its cause.

Follow-up assessment: ImagineNav++ arXiv:2512.17435 adds memory/selective foveation. Its project and forthcoming-code status were inspected, but a matched full implementation/experimental audit is outside this dossier. Confidence: Medium.

### 02-imaginenav-views

**Still unresolved in the reviewed literature — Methodological.** Novel-view prediction errors leave a substantial action-selection gap.

Conditions: Unseen scenes and viewpoints outside the synthesizer training distribution. Cause: The action selector depends on generated candidate views whose appearance/content can be wrong.

Evidence: PAPER EXPERIMENT: Table 1 p.7 oracle views raise SR from 53 to 62 on HM3D and from 51 to 59 on HSSD.

Follow-up assessment: HNR changes representation and WMNav changes the scoring interface; cross-protocol gains do not resolve the matched ImagineNav oracle gap. Confidence: Medium.
