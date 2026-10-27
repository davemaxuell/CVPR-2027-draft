# FantasyVLN

## Bibliography and reviewed material

**FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-and-Language Navigation**. Jing Zuo; Lingzhou Mu; Fan Jiang; Chengcheng Ma; Mu Xu; Yonggang Qi. CVPR 2026 (2026). Research cutoff: **2026-10-22**.

Primary publication/source: https://openaccess.thecvf.com/content/CVPR2026/papers/Zuo_FantasyVLN_Unified_Multimodal_Chain-of-Thought_Reasoning_for_Vision-and-Language_Navigation_CVPR_2026_paper.pdf. Bibliographic author/title fields were checked against the supplied PDF first page; source labels that depend solely on an author statement are identified in the access record. Main PDF pages actually read for this dossier: 1,3-6. Reference-list-only pages are not claimed as substantive review. Associated material reviewed: No separately supplied associated file; included appendices are identified by the page list.

- FantasyVLN.pdf; Main PDF; SHA256 `592426cc943fd9270393fb848b8230cffbbebcd9e375a11d7031e3660b1c7529`; 10 pages.

## Problem, task and assumptions

Long, multistage LH-VLN with a 7 B VLM. The aim is to benefit from reasoning supervision while using a fast direct action mode at deployment.

## Method, training and inference

The backbone supports four gated modes: direct non-CoT, textual CoT, coarse VAR visual CoT, and combined multimodal CoT. The VAR decoder is frozen. Equations 4–8 alternate direct action cross-entropy with cross-mode alignment. After a non-CoT update, its detached action-token probabilities supervise CoT action outputs, while ordinary token losses supervise the reasoning content. Thus the explicit alignment direction is not simply a CoT teacher distilled into a direct student. Shared backbone updates may still improve the direct mode. Deployment primarily selects non-CoT inference, making visual imagination chiefly a training mechanism; the code audit below confirms the alternating update order.

## Novelty and closest predecessors

Unified reasoning modes and action alignment provide auxiliary learning for a fast direct policy. CoT-VLA instead generates intermediate images during deployment. Auxiliary reasoning supervision and shared-backbone transfer are close prior mechanisms that constrain broad novelty claims.

## Results and independent verdict

Table 1, p. 6: full-task SR 2.44, ISR 11.01, CSR 9.64 and CGT 8.99, versus Aux-Think SR 0.65 and baseline SR 0. Table 2 reports 1.03 actions per second versus CoT-VLA 0.19 (about 5.4 times faster) and WorldVLA 1.02. Table 3 gives visual-only non-CoT ISR 11.19, slightly above the full system's 11.01, although the full system has better complete-task SR . Removing alignment gives SR 0. CoT-VLA and Aux-Think were reimplemented because their original code was unavailable, limiting direct comparability.

**Evidence distinction:** numerical results above are PAPER EXPERIMENT, not independently reproduced. Architecture and intended operation are AUTHOR CLAIM unless an INSPECTED CODE locator below confirms them. Limitations and cross-paper implications below are DEDUCTION, not measured failures unless explicitly identified.

## Official implementation and reproducibility

Repository: https://github.com/Fantasy-AMAP/fantasy-vln. Snapshot commit: `01be27681184b1b878d79ac9fd434f032051427a`. Retrieval: 2026-10-22.

INSPECTED CODE: `trainer.py:145–178,201–217` forms hard losses for reasoning tokens and soft cross-entropy for action tokens using detached non-CoT teacher logits. Lines 571–626 first update the non-CoT branch, then refresh teacher outputs and update CoT branches. This confirms the direction is non-CoT→CoT for the explicit action-alignment loss; shared parameters may still transfer useful representation learning back to the direct mode. `train.py:33–37,100–129` passes alignment options and prepared datasets into the trainer. `lhvln/NavModel/VLMModel.py:170–217` processes history/current images and generates action text through the common VLM. No separate online VAR rollout appears in this inference function. Benchmark metrics/config and all external visual-token assets were not independently executed; code inspection does not establish faithful chain-of-thought reasoning.

Inspected files: `lhvln/NavModel/VLMModel.py`, `train.py`, `trainer.py`. Project code executed: **No**. A fetched source file is not counted as inspected until recorded here. No simulator benchmark, training run or checkpoint evaluation was executed.

## Specific bottlenecks and initial follow-up chain

More than 97% of full tasks still fail. Token alignment does not prove that reasoning is faithful or causally responsible for actions; auxiliary branches may provide representation regularization. Foundation-model priors, scale and training data complicate attribution.

Relevant follow-up relationship: MonoDream disables auxiliary heads at inference. NavForesee and WorldVLN retain predicted structures in action computation. Grouping these papers by actual deployment use is more informative than grouping by the word imagination.

## Access and uncertainty record

Full text is preserved under `corpus/fantasyvln/`; original PDF paths and hashes are in its metadata. Search logs are in `evidence/02/search_*.json`; commit-pinned fetched code and retrieval manifests, where available, are in `repositories/02/fantasyvln/`. Equations/tables were read from page-labelled full-PDF extraction; ambiguous extraction or missing implementation details remain explicitly unresolved. Authors’ claims of causal/general reasoning are not elevated to established mechanisms.

## Structured bottleneck resolution record

These statuses apply only to the reviewed chain and stated conditions; they are not universal unsolved-problem claims. Immutable code links are in [the assignment implementation audit](../assignments/02_repository_audit.md).

### 02-fantasyvln-completion

**Still unresolved in the reviewed literature — Methodological.** Full multi-part navigation success remains very low despite relative gains.

Conditions: The evaluated long complete-task benchmark rather than isolated subgoal success. Cause: A long sequence of grounding/action decisions can fail even when per-subtask completion improves; the experiment does not isolate the causal distribution of these failures.

Evidence: PAPER EXPERIMENT: Table 1 p.6, SR 2.44, ISR 11.01, CSR 9.64; more than 97% of complete tasks fail. No independently reproduced decomposition.

Follow-up assessment: MonoDream/NavForesee/WorldVLN evaluate different task/sensing settings, so their headline successes do not establish a solution on this protocol. Confidence: Medium.

### 02-fantasyvln-faithfulness

**Uncertain because evidence is insufficient — Insufficiently tested.** Action-token alignment does not establish that visual reasoning is faithful or necessary at inference.

Conditions: Attributing improvements to actual reasoning rather than auxiliary representation learning. Cause: Training aligns branches, while direct deployment can act without explicit visual CoT; multiple training changes share credit.

Evidence: PAPER EXPERIMENT: Table 3, visual-only non-CoT ISR 11.19 versus full 11.01; alignment removal SR 0. INSPECTED CODE: trainer ranges cited in dossier confirm non-CoT teacher to CoT action alignment.

Follow-up assessment: MonoDream and VLN-SIG also use auxiliary prediction. No matched counterfactual intervention establishes faithful reasoning across these systems. Confidence: Medium.
