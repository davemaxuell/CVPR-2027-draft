# AirBERT

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Airbert: In-Domain Pretraining for Vision-and-Language Navigation**. Pierre-Louis Guhur, Makarand Tapaswi, Shizhe Chen, Ivan Laptev, Cordelia Schmid. ICCV 2021.

Primary paper: [Airbert: In-Domain Pretraining for Vision-and-Language Navigation](https://openaccess.thecvf.com/content/ICCV2021/html/Guhur_Airbert_In-Domain_Pretraining_for_Vision-and-Language_Navigation_ICCV_2021_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: AirBERT.pdf (Main PDF; 10 pages; SHA256 f58e1a533dc9642c2a8d9da2084155224da514850b758f79a08f680dbda9930d).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Reduce the domain gap between generic web image–text pretraining and indoor instruction navigation without requiring new embodied human demonstrations. The BnB corpus uses approximately 140k rental listings, 1.4M images and 0.7M captions; images in one listing need not form a physically traversable trajectory.

## 3. Mechanism, training, and inference

PDF pp 3–5: clean rental images/captions, sample 4–7-image pseudo-paths from a listing, merge room views, concatenate captions, insert uncaptained views, and create instructions by template noun substitution or generation. A ViLBERT-style two-stream model encodes language tokens and visual regions with image separators. Besides masked token/region prediction, the shuffling objective contrasts a matched trajectory–instruction pair against shuffled order negatives. In Eq 3, for scalar compatibility sθ(V, W), L=−log[exp sθ(V, W)/(exp sθ(V, W)+Σn exp sθ(Vn−, Wn−))]; the contrast concerns ordering compatibility, not physical transition validity. The model supports two distinct downstream uses: discriminative reranking of explored complete trajectories, and initialization of recurrent VLN-BERT for action generation. These inference protocols must remain separated.

## 4. Novelty and evidence

In-domain non-embodied pretraining and order-sensitive shuffling are the mechanisms. The synthetic-path construction supplies scale but sacrifices measured geometry. Table 2 tests construction choices, Table 3 tests pretraining, and Table 5 tests noun/direction corruptions: this is relevant prior art for proposals based on counterfactual instruction discrimination.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 2 p 6 reports 62.45 SR with image-caption training versus 66.52 with rephrasing/merge/insertion; Table 3 full pretraining reaches 68.67. Table 5 unseen direction-corruption discrimination is 49.8 versus 45.3 for VLN-BERT; noun-swap 61.1 versus 52.3. Critical protocol correction: Table 8 p 7 test SR 77 uses pre-exploration and a 687m trajectory with SPL 0.01. It is not comparable to single-run navigation; recurrent AirBERT is a separate result. VERDICT: transfer and order discrimination improve, while the weak direction test and physically ungrounded pseudo-paths leave semantic/spatial fidelity unresolved.

Table 9, PDF p 8, evaluates fine-tuning on one or six training houses, averaging five sampled house sets and excluding houses with fewer than 80 paths. To avoid leakage through an EnvDrop generator trained on all houses, candidate paths are random shortest routes. One-house unseen SR is 50.00 for AirBERT versus 22.43 for VLN-BERT. Here “one-shot” means one training environment and still involves training; it does not mean training-free navigation.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/airbert-vln/airbert](https://github.com/airbert-vln/airbert); inspected snapshot **34f533dba3f2a78cc111e40e5ffd46315a349f1b**. Snapshot metadata and source are under `repositories/01/airbert`.

Inspected the model forward inputs, BnB dataset construction/shuffling, and the pretraining driver. No corpus was downloaded and no candidate search was executed.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [airbert.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/airbert.py) — forward nearline62: The model consumes text tokens and image-region features for multimodal scoring.
- [utils/dataset/bnb_dataset.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/utils/dataset/bnb_dataset.py) — load_shuffler284–286; caption construction288 onward; negatives536 and587–593: Shufflers produce negative captions/images and return them with positives, confirming the supervision mechanism.
- [train_bnb.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/train_bnb.py) — training entrypoint and loss aggregation: Pretraining is an optimization stage; inference is not training-free use of an unchanged general VLM.
- [test.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/test.py) — 125–192: Reranks externally supplied beams; test conversion explicitly prepends exploration_path. This confirms long exploration belongs in comparison cost.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **airbert-B1 — Partially addressed**. Listing membership and shuffled order do not guarantee a valid navigable transition or faithful directional language. Evidence: Pseudo-path construction §3; Table 5 direction errors. Type: data/methodological. Follow-ups: RoomTour3D video/geometry data, ScaleVLN rendered traversable environments, SRDF consistency filtering. Confidence: high.
- **airbert-B2 — Partially addressed**. Exploration reranking obtains high SR at extreme movement cost; compare physically charged trajectories and SPL. Evidence: Table 8 SR 77/TL 687/SPL 0.01. Type: inference cost. Follow-ups: DUET single-run global navigation. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://roomtour3d.github.io; https://github.com/wz0919/ScaleVLN
