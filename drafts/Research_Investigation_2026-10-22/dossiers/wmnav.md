# WMNav

## Bibliography and reviewed material

**WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation**. Dujun Nie; Xianda Guo; Yiqun Duan; Ruijun Zhang; Long Chen. IROS 2025; author project labels Oral (2025). Research cutoff: **2026-10-22**.

Primary publication/source: https://b0b8k1ng.github.io/WMNav/. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-8. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- WMNav.pdf; Main PDF; SHA256 `aa992ca55b1c5bd53dccc6481cf960a417971aa2b07333eeba4c60a5a1527bc9`; 8 pages.

## Problem, task and assumptions

Training-free task-specific ObjectNav from panoramic RGB-D and pose. Pretrained VLMs provide semantic exploration scores and action choices, not a learned physical transition simulator.

## Method, training and inference

Section III: rotate through six viewpoints, query PredictVLM for 0–10 directional target/exploration scores, project them into a metric curiosity map. Initialize unknown cells 10 and merge M_t=min (M_(t−1),M_nav_t)(Eq. 3). Project map scores back into views and select maximum average direction. PlanVLM uses previous subtask/current view to output next subtask and goal flag (Eq. 4). ReasonVLM selects annotated polar waypoint candidates. Exploration filters visited regions; goal-approaching mode removes length constraints and densely samples goal locations, then uses estimated distance for stop (Eq. 6). Three VLM roles are prompt configurations, without gradient updates. The 'cost' is subtask/flag prompt memory, not a learned reward model.

## Novelty and closest predecessors

Persistent curiosity map plus subtask memory and two-stage waypoint selection, extendingVLMNav-style annotated actions. Uses world-model terminology broadly for semantic prediction and memory; no action-conditioned next-RGB model is present.

## Results and independent verdict

Table I, p. 6: HM3D v0.1 SR 58.1/SPL 31.2; MP3D 45.4/17.2. MP3D SPL is below VLFM 17.5 despite broad superiority wording. Table II, HM3D v0.2: adding the two-stage action proposal raises SR from 70.1 to 72.2 while reducing SPL from 34.7 to 33.3. Table III gives Qwen 3B SR 29.7 versus Gemini 1.5 Pro 58.1, showing substantial backbone dependence. HM3D versions must be compared separately.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/B0B8K1ng/WMNavigation. Snapshot commit: `57b74cee7a00b669733e8ca917f34a8aadfd6fd0`. Retrieval: 2026-10-22.

INSPECTED CODE: `src/WMNav_agent.py:1244–1302` parses VLM scores, updates curiosity with elementwise minimum, and chooses the highest mean direction. There is an important exception: identified goal-mask cells are reset to 10 at lines 1286–1287. Thus ordinary merge updates cannot recover an underestimated exploration cell, but goal identification can override it. `src/WMNav_env.py:356–389` collects 12 views at roughly 30-degree rotations and applies directional masks, then carries subtask text forward. `src/WMNav_agent.py:752–764` uses 0.8 m internal goal stopping; `config/WMNav.yaml:20–43` ships Gemini 2.0 Flash, depth-sensor navigability, 40 macro steps and a 1 m evaluation success threshold. These are not the paper's headline Gemini 1.5 Pro configuration. No task-specific training optimizer or learned transition loss is part of this prompt-based method. API access, Habitat assets and exact model/version settings are required to reproduce it.

Inspected files: `config/WMNav.yaml`, `src/WMNav_agent.py`, `src/WMNav_env.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

DEDUCTION: ordinary minimum merging prevents recovery from a falsely low exploration value when a later estimate rises. However, the released code explicitly resets identified goal-mask cells to 10, so this is a scoped limitation with a recovery exception. Prompt scores are not calibrated probabilities. The printed 1.0 versus 0.1 m stopping threshold is inconsistent, and the inspected code uses 0.8 m internally; exact configuration must be stated.

Relevant follow-up relationship: ImagineNav++ adds memory to visual imagination; SGM/L2 M supply learned map probabilities/uncertainty. Confidence recovery and observation-provenance are specific remaining concerns.

## Access and uncertainty record

Full text is preserved under `corpus/wmnav/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/wmnav/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-wmnav-recovery

**Partially addressed — Implementation-specific.** An ordinary cell that receives a falsely low curiosity score cannot recover merely because a later score rises.

Conditions: Minimum-merge updates to non-goal cells. Cause: The ordinary update takes the minimum of old and new values, creating irreversible pessimism; identified goal-mask cells are an explicit exception reset to 10.

Evidence: INSPECTED CODE: _merge_evalue and update_curiosity_value:1263–1287. EXECUTED TEST: isolated 10→2→9 merge gives 10→2→2; goal reset produces 10. Not a simulator performance test.

Follow-up assessment: Goal-mask reset resolves recovery for those identified cells only. SGM/L2M use different maps, with no matched ordinary-cell recovery intervention reviewed. Confidence: High for update behavior; unknown for aggregate success impact.

### 02-wmnav-backbone

**Uncertain because evidence is insufficient — Methodological.** Heuristic VLM curiosity scores and stopping choices vary substantially with backbone and configuration.

Conditions: Transferring the recipe to smaller VLMs or comparing different HM3D versions. Cause: Prompt responses are not calibrated probabilities; paper and source stopping thresholds differ and models differ in semantic judgment.

Evidence: PAPER EXPERIMENT: Table III p.6 Qwen3B SR 29.7 versus Gemini1.5Pro 58.1; INSPECTED CODE uses 0.8m internally, while printed 1.0/0.1m threshold is inconsistent. Evaluation success threshold is a separate setting.

Follow-up assessment: ImagineNav++ adds memory and SGM/L2M change score sources; none establishes backbone-invariant calibration under a matched protocol. Confidence: Medium.

Independent isolated check (**EXECUTED TEST**): An isolated ordinary minimum-merge update on values 10,2,9 remains at 2; an explicit goal-mask reset returns 10. This tests update arithmetic, not navigation performance. Setup and output: `evidence/02/logic_checks.py` and `evidence/02/logic_check_results.json`.
