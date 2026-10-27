# Visual Planning (VPRL)

## Bibliography and reviewed material

**Visual Planning: Let's Think Only with Images**. Yi Xu; Chengzu Li; Han Zhou; Xingchen Wan; Caiqi Zhang; Anna Korhonen; Ivan Vulić. ICLR 2026 (2026). Research cutoff: **2026-10-22**.

Primary publication/source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/3de60dc2ff7ac8866e805bbf457e2291-Abstract-Conference.html. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-6. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- Visual Planning (VPRL).pdf; Main PDF; SHA256 `07fabcffc8a710f6a8fa19cc56e35db4d98734532dd604158ea34899771c8293`; 34 pages.

## Problem, task and assumptions

Plan in fully visual grid environments (FrozenLake,Maze,MiniBehavior), outputting a sequence of images. This is not photorealistic, partially observed route-following.

## Method, training and inference

Section 2: LVM7 B autoregressively generates next visual states (Eq. 1). Warmup CE uses random-walk prefixes with a randomly sampled valid next-state target (Eq. 2), maintaining transition diversity. Stage 2 samples G next-image candidates and applies GRPO clipped importance ratios plus referenceKL (Eq. 3). A rule-based dynamics interpreter extracts whether the image transition is valid; a progress estimator classifies distance-decreasing, valid-nonprogressing or invalid moves, with rewards 1/0/−5(Eq. 4). Thus hidden symbolic environment knowledge supplies verification/reward even though model outputs are images. Inference generates a full visual plan; actions are implicit and parsed for evaluation.

## Novelty and closest predecessors

Applies RL to image-sequence planning, comparing a visual-only pretrained backbone with textual multimodal reasoning. Not evidence that language is generally unhelpful; the benchmarks are deliberately vision-first and have exact verifiers.

## Results and independent verdict

Table 1, p. 6: VPRL averages EM 80.6/PR 84.9, versus visual SFT 56.1/65.2 and Qwen SFT 53.6/69.9. EM accepts any shortest valid trajectory by parsing environment states rather than requiring exact pixel equality. This comparison isolates the training change within LVM more clearly than it establishes modality superiority across differently pretrained backbones.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/yix8/VisualPlanning. Snapshot commit: `081e67ca6926318623a73b279d45cf0f7dc2b8fa`. Retrieval: 2026-10-22.

INSPECTED CODE: `train_rl_maze.py:79–106` invokes a symbolic action parser with the true layout and distance map, rewarding a distance-decreasing valid transition by 1, a valid nonprogressing transition by 0, and invalid output by -5. Lines 171–210 configure GRPO/LoRA training. `evaluation/visual_planning_evaluator.py:418–432` checks parsed validity and target coordinates at the expected shortest length; MiniBehavior uses task-specific pick/drop logic at 596–602. The release therefore depends on privileged rule-based validation even though its model emits only image tokens. Some imported custom trainer/parser/tokenizer infrastructure is external or absent from the small snapshot; it is not a self-contained reproduction package. No training or planning benchmark was executed.

Inspected files: `evaluation/visual_planning_evaluator.py`, `train_rl_maze.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

Rule-based interpreter and exact progress estimate are privileged; transferring to noisy unknown environments requires learned validators whose errors can be exploited. Warmup random targets encourage but do not prove full transition coverage. Synthetic grid success is not empirical navigation performance.

Relevant follow-up relationship: FantasyVLN/CoT-VLA/WorldVLN use visual reasoning in richer embodied tasks but do not inherit exact grid verifiers. A new visual-planning proposal needs a reliable real-world outcome validator.

## Access and uncertainty record

Full text is preserved under `corpus/visualplanning/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/visualplanning/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-visualplanning-verifier

**Uncertain because evidence is insufficient — Methodological.** Exact grid-state validators do not transfer automatically to noisy embodied outcomes.

Conditions: Replacing synthetic grids with real visual navigation or manipulation. Cause: RL rewards and exact-match/progress scoring use a rule-based interpreter and known transition/task structure; a learned real-world validator can be wrong or exploitable.

Evidence: AUTHOR METHOD and DEDUCTION: interpreter/reward formulation; Table 1 p.6 EM/PR 80.6/84.9 establishes gains within the synthetic domain only.

Follow-up assessment: CoT-VLA/FantasyVLN/WorldVLN use richer tasks without inheriting the exact validator. No reviewed real-world counterpart preserves the same correctness guarantees. Confidence: Medium.
