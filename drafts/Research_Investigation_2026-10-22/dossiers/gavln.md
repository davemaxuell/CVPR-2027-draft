# GA-VLN

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**GA-VLN: Geometry-Aware BEV Representation for Efficient Vision-Language Navigation**. Jiahao Yang, Zihan Wang, Xiangyang Li, Xing Zhu, Yujun Shen, Yinghao Xu, Shuqiang Jiang. CVPR 2026.

Primary paper: [GA-VLN: Geometry-Aware BEV Representation for Efficient Vision-Language Navigation](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_GA-VLN_Geometry-Aware_BEV_Representation_for_Efficient_Vision-Language_Navigation_CVPR_2026_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: GA-VLN.pdf (Main PDF; 11 pages; SHA256 d8a2d32e7128191aca69daa818b1cd27f560f10e93bac63a5fa09963dfb7fb5f).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

GA-VLN reduces redundant history tokens and injects spatial geometry into a fine-tuned 7B multimodal navigation model. Its input is front-view RGB-D plus pose; monocular here means a single viewing direction, not RGB-only sensing.

## 3. Mechanism, training, and inference

§3 pp 4–5: SigLIP patch features and frozen VGGT-1B multi-view features are projected to world coordinates with depth/pose, transformed to agent coordinates, and mean-pooled into occupied 0.25m BEV cells within ±10m. Correct dimensional backprojection is pworld=R(DK⁻¹[u, v, 1]ᵀ)+t; printed Eq 1’s 3×4 transform times 3-vector omits homogeneous augmentation. Each cell token bi, j=(Σv∈S v)/|S|+ei, j combines explicit visual and projected geometry features; only occupied cells remain. A two-layer 4096-hidden MLP aligns VGGT features. LLaVA-Video-7B receives BEV, current view, instruction and predicts 4 actions per round; second round reuses BEV, refreshing it every 8 actions. Fine-tune all modules except VGGT for 2 epochs with action-token CE; visual learning rate 5e−6, others 2e−5. Data mix includes R2R-CE, RxR-CE, EnvDrop, ScaleVLN and 319,022 converted SRDF trajectories, without DAgger or VQA data. NavRAG evaluation additionally fine-tunes one epoch on NavRAG-CE, so it is not zero-shot transfer.

## 4. Novelty and evidence

Compact BEV geometry tokens integrate a 3D foundation encoder with an MLLM action policy. Table 2 controls representation under the full data mix; Table 3 repeats a restricted-data ablation without SRDF. This is a trained spatial-memory policy, not an online world model rollout planner.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 2 p 7 baseline 51.49 SR/46.18 SPL/342.9 ms; BEV without VGGT 59.21/53.87/212.9 ms; full 60.96/55.19/258.7 ms. Geometry adds accuracy but costs latency relative to BEV-only. Table 3 without SRDF reduces average tokens 4003→514 and yields 46.49→53.56 SR; token statistics use 121 training trajectories, not necessarily deployment distribution. VERDICT: useful evidence separating data and representation; stale 8-action map updates, pose/depth errors and 3D height collapse remain testable risks.

Table 4, PDF p 8, tests additive sensor noise: SR falls from 60.96 to 59.11 with depth noise, 59.82 with pose noise and 58.29 with 5-degree rotation noise. The rotation decrease is 2.67 percentage points, so the nearby claim of a drop below two percentage points is not true for every listed row. This finite Gaussian-noise test partially addresses sensor robustness but does not establish robustness to accumulated drift, bias or missing depth. The real-robot section presents examples in one room without a quantitative success denominator.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/jahhaoyang/GA-VLN](https://github.com/jahhaoyang/GA-VLN); inspected snapshot **cc6086b7081a346695abecf6821f827e2db44a43**. Snapshot metadata and source are under `repositories/01/gavln`.

Inspected frozen VGGT extraction, grid quantization, scatter-add/mean pooling, feature projection, dataset action setup and evaluation entrypoint.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [gavln/model/model_gavln.py](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/gavln/model/model_gavln.py) — encode_images127–261: VGGT aggregation runs under no_grad; SigLIP/VGGT features share grid accumulators and are averaged only in occupied cells before MLLM projection.
- [gavln/dataset/vln_action_dataset.py](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/gavln/dataset/vln_action_dataset.py) — 83–154;237–318: Conversation preprocessing masks prompt tokens with IGNORE_INDEX and supplies assistant/action text labels for autoregressive supervision.
- [gavln/gavln_eval.py](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/gavln/gavln_eval.py) — 178–215;412–454;456 onward: Greedy model generation is parsed into actions; Habitat metrics are recorded after rollout. Training driver was not verified from the official entry-point documentation.
- [scripts/gavln_eval.sh](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/scripts/gavln_eval.sh) — 6–25: Evaluation requires external checkpoints, SigLIP and VGGT paths plus distributed-launch settings.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **gavln-B1 — Uncertain because evidence is insufficient**. Pose/depth errors misregister feature memory and mean pooling collapses height and potentially conflicting observations. Evidence: Eqs 1–5 and scatter-add implementation; Table 4/PDF p 8 partially tests additive Gaussian sensor noise. Type: representation/sensor. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.
- **gavln-B2 — Partially addressed**. BEV reused for 8 actions can become stale at decision points; more history is not monotonically beneficial. Evidence: §3.3;Table 3 history-length saturation. Type: latency/feedback. Follow-ups: HSGM observation-triggered planning. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/Teacher-Tom/HSGM_public
