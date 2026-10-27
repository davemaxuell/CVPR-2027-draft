# NaviLLM

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Towards Learning a Generalist Model for Embodied Navigation**. Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, Liwei Wang. CVPR 2024.

Primary paper: [Towards Learning a Generalist Model for Embodied Navigation](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Towards_Learning_a_Generalist_Model_for_Embodied_Navigation_CVPR_2024_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: NaviLLM.pdf (Main PDF; 11 pages; SHA256 b67582a2bcd70981971746abb0468e2e759622012c268c290be30d1cb3f6dbb2); NaviLLM (supplement).pdf (Supplement; 3 pages; SHA256 8f0940a540efa07027ce4a8879f171ff3ac2902cad685a415f040a642a70a5d0).

Main-paper analysis is supplemented by the explicitly listed associated-file review below. Exact SHA256 duplicates were deduplicated and not counted as additional evidence. NaviLLM (supplement).pdf: pages [1, 2, 3]; All supplemental text, prompts, dataset counts and ablations inspected.

## 2. Problem and operating assumptions

Unify instruction navigation, object localization, trajectory summarization, 3D QA and held-out EQA within a multimodal language model. Multi-task transfer is central; a model trained on several embodied datasets is distinct from an untouched general VLM used zero-shot.

## 3. Mechanism, training, and inference

Paper §3–4 pp 3–5: frozen EVA-CLIP-Large 428M gives per-view CLS features; a 2-layer 1024D Transformer fuses views plus angle/GPS information; projections align them to Vicuna-7B. Prompts contain Task, indexed Observation, indexed History and Output Hint. Paper describes generation of view/object IDs, summaries or answers using unified token CE, L=−Σj log pθ(yj|prompt, y<j). History stores selected-view representations. Pretraining 10k steps and multi-task fine-tuning 5k, batch 64, Adam 3e−5, 8 A100 GPUs for about 80 hours; teacher/student forcing alternate. Data includes CVDN, SOON, R2R, REVERIE, ScanQA, augmentation and 23k LLaVA examples. Test uses greedy generation except temperature .01 for SOON/REVERIE. RELEASED-CODE QUALIFICATION: nav_model.py additionally uses global graph candidate embeddings and visited masks, merges local embeddings, permutes candidates, then applies a classifier head to LLM CLS hidden states for navigation logits. Thus the release contains DUET-derived graph infrastructure and the action path is not simply unrestricted textual ID generation.

The released implementation differs materially from the paper’s uniform generation description: train.py 229 creates CrossEntropyLoss(ignore_index, reduction=sum), and tasks/agents/mp3d_agent.py 734–756 applies it to navigation candidate logits and teacher action indices. Separate QA and summarization branches apply language-model generation losses at 865 and 901. Therefore, the inspected released navigation branch is an LLM-conditioned graph classifier with separate generative tasks, not solely next-token generation of a view ID.

## 4. Novelty and evidence

Schema-unified embodied multi-task adaptation of an LLM, with transfer to held-out tasks. The contribution is broader than a stronger R2R policy. Paper-versus-release differences should be resolved before using it as a supposedly graph-free VLM baseline.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 CVDN test 7.90 is goal progress in meters, despite its caption incorrectly saying SPL for CVDN; SOON 26.26 SPL, R2R 60 SPL, REVERIE 32.33 SPL. Table 2 ScanQA test exact-match 26.3. Table 4 p 7 held-out EQA navigation 47.78 SR/35.60 SPL, answer 44.5%; EQA dataset was manually filtered for invalid endpoints (§4.4), so release/filter matching matters. Held-out task comparisons support transfer, while random-LLM initialization and multi-task ablations show dependence on language pretraining/data. VERDICT: useful generalist baseline, not universal dominance over specialized models.

Supplement pp 1–3 gives task history caps (15 for R2R/REVERIE, 30 for CVDN, 20 for SOON), excludes 849 EQA examples, and supplies candidate(0)=STOP prompts. Its SOON count is 27.8k in Table 3 versus 2780 in the text, an apparent factor-ten reporting discrepancy. Schema ablations aggregate heterogeneous metric units; that aggregate should not be interpreted as a calibrated universal navigation score.

Table 5, PDF p 8, shows that extra pretraining does not improve every task: relative to multi-task training with a pretrained LLM, adding the pretraining stage increases R2R SPL 58→59 but lowers REVERIE SPL 36.63→35.68 and ScanQA exact match 23.3→22.1. Generalist data mixing has measurable tradeoffs.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/LaVi-Lab/NaviLLM](https://github.com/LaVi-Lab/NaviLLM); inspected snapshot **e069f46ed98affb221d58715a785613622e11145**. Snapshot metadata and source are under `repositories/01/navillm`.

Inspected scene encoder, global/local navigation embeddings, prompt construction interface, classifier output, task agents and loss-return paths. The release reuses graph structures beyond the simplified paper diagram. No checkpoint/runtime validation was performed.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [models/nav_model.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/models/nav_model.py) — forward_navigation129–247: Builds global graph embeddings, masks visited nodes, merges local views, permutes candidates, applies out_head to LLM CLS states and returns fused logits.
- [models/nav_model.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/models/nav_model.py) — summarization251–344;object grounding407 onward: Other tasks route through language-model losses/heads; training objectives depend on task mode.
- [models/image_embedding.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/models/image_embedding.py) — multi-view embedding classes: Encodes visual and geometric inputs before the language model.
- [tasks/agents/mp3d_agent.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/tasks/agents/mp3d_agent.py) — rollout/teacher-action paths: Graph navigation and supervision sit outside the LLM module.
- [tasks/agents/mp3d_agent.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/tasks/agents/mp3d_agent.py) — 734–756;845–904: Navigation criterion over candidate logits is backpropagated separately from QA/summarization token-generation loss.
- [train.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/train.py) — 207–231: Constructs summed categorical cross entropy with ignored padding targets; driver uses task-specific agents.
- [tasks/datasets/r2r.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/tasks/datasets/r2r.py) — 10–60;108–129: Expands instructions into samples, truncates encodings and prepares fine-grained labels; evaluation flattens graph subpaths and uses final/nearest geodesic errors, path length and SPL.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **navillm-B1 — Partially addressed**. Shared multi-task training does not eliminate action grounding failures; broad transfer and specialized endpoint accuracy trade off. Evidence: Tables 1,3,4. Type: methodological. Follow-ups: RoomTour3D action-enriched data, GA-VLN geometry tokens. Confidence: high.
- **navillm-B2 — Uncertain because evidence is insufficient**. Released action implementation differs from the paper’s generative description; matching architecture/checkpoint/protocol is a reproducibility prerequisite. Evidence: Pinned nav_model.py 129–247. Type: implementation. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://roomtour3d.github.io; https://github.com/jahhaoyang/GA-VLN
