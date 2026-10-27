# RoomTour3D

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation**. Mingfei Han, Liang Ma, Kamila Zhumakhanova, Ekaterina Radionova, Jingyi Zhang, Xiaojun Chang, Xiaodan Liang, Ivan Laptev. CVPR 2025.

Primary paper: [RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_RoomTour3D_Geometry-Aware_Video-Instruction_Tuning_for_Embodied_Navigation_CVPR_2025_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: RoomTour3D.pdf (Main PDF; 11 pages; SHA256 62b6de3a862e710ea6976c7cd1d5e652507880b9603ec5c34725ec658cc6661f); RoomTour3D (slides).pdf (Conference slides; 18 pages; SHA256 520d889f1b99ee94d6c580720deb02da4a32e31c45efa615e36fdf23504f8615).

Main-paper analysis is supplemented by the explicitly listed associated-file review below. Exact SHA256 duplicates were deduplicated and not counted as additional evidence. RoomTour 3D (slides).pdf: pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]; All slide text inspected; slide 13 also rendered and visually inspected for the 100-action manual quality audit.

## 2. Problem and operating assumptions

Extract useful indoor navigation supervision from real room-tour videos, avoiding the lack of measured motion in unordered rental images and the limited scene diversity of simulator annotations. Video motion still does not automatically provide valid alternative robot actions.

## 3. Mechanism, training, and inference

§3 pp 3–5: sample video frames every 2 seconds; BLIP-2 predicts 16 room types with temporal smoothing; RAM tags objects, GroundingDINO boxes them, DepthAnything estimates depth. Text templates combine object category, relative direction/distance and room labels; GPT-4-Turbo turns frame sequences into trajectory descriptions. COLMAP reconstructs clips sampled 3 fps, 100 seconds with 10 second overlaps, merging reconstructions sharing>3 frames. Turning/revisited-location candidates are selected using orientation/cosine changes, NMS and DBSCAN; actual continuation is positive and strongly different orientations supply negatives. Dataset totals 200k instructions, 17k action-enriched trajectories, 1847 scenes. NaviLLM first learns trajectory summarization with token CE, then candidate-action prediction using action-enriched histories; human/simulator multi-task data remains in the regular supervised setting. Zero-shot-transfer variant removes simulator action/geometric training but still learns from RoomTour 3D actions.

## 4. Novelty and evidence

Real-video object/motion/geometry annotation and action-candidate mining, rather than a new backbone. Compared with AirBERT, ordered video and reconstructed geometry provide stronger motion evidence; reconstructed candidate views still differ from online reachable-neighbor observations.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 7 reproduced NaviLLM R2R validation SPL 56.7→62.4, REVERIE 31.4→37.4 with action data. Compare against reproduced baseline, not only original published 59/35.7. CVDN test GP 7.90 original versus 7.22 action data is not a universal gain. Zero-shot transfer on R2R 14.33 SR/10.86 SPL is far below GPT-4V MapGPT 43.7/34.8, though above certain open 7B baselines. §5.4 p 8 human audit 100 descriptions:mean 3.08/4, 74% mostly/perfectly relevant;26% fall below that bar. VERDICT: useful scalable supervision with measurable residual noise, not verified perfect trajectories.

Conference slides 8 and 13 report 1847 videos totaling 243 hours and a manual audit of 100 action examples, of which 87% pass the spatial-proximity/diversity criteria. This validates that limited annotation property; it does not validate physical executability, all instruction semantics, or human agreement. No uncertainty interval or inter-rater agreement is established by the slide.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/roomtour3d/roomtour3d-NaviLLM](https://github.com/roomtour3d/roomtour3d-NaviLLM); inspected snapshot **eac6c0753621e92ebc7ac8e92c3e25f14285e31c**. Snapshot metadata and source are under `repositories/01/roomtour3d`.

Official NaviLLM fork pinned. Retrieved task agent, video dataset/environment, model, train driver andmulti-task config. Geometry/data-curation release completeness is audited separately from training code availability.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [tasks/agents/tour3d_agent.py](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/tasks/agents/tour3d_agent.py) — 579–604;828–865;907–944: Teacher-forced video navigation criterion and separate summary generation loss; sampled DAgger path is commented out in this video agent.
- [tasks/datasets/tour3d_dataset.py](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/tasks/datasets/tour3d_dataset.py) — 121–189;370–462: Filters short trajectories, expands instruction records, and constructs video-frame navigation candidates with angle/3D-position metadata.
- [configs/pretrain.yaml](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/configs/pretrain.yaml) — Feature and Dataset sections: Explicit external EVA feature files and synthetic/MP3D task data paths.
- [configs/multi.yaml](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/configs/multi.yaml) — Feature and Dataset sections: Fine-tuning data mix includes video features; one path has a.bak suffix, requiring local data-path validation.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **roomtour3d-B1 — Partially addressed**. Video reconstruction and generated semantic labels can be wrong; sparse human audit establishes useful relevance, not exact action/landmark truth. Evidence: §3 COLMAP/experts;§5.4 human ratings. Type: data quality. Follow-ups: SRDF-style consistency audits, counterfactual data validation. Confidence: high.
- **roomtour3d-B2 — Uncertain because evidence is insufficient**. Video-derived candidate frames may contain future-position visual information unlike real current-view candidates; training/deployment observation matching must be controlled. Evidence: §3.2/§4.3 candidate construction. Type: data/protocol. Follow-ups: No resolving follow-up verified in this assignment; coordinator follow-up audit pending. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/roomtour3d/roomtour3d-NaviLLM
