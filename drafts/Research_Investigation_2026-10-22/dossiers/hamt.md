# HAMT

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**History Aware Multimodal Transformer for Vision-and-Language Navigation**. Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, Ivan Laptev. NeurIPS 2021.

Primary paper: [History Aware Multimodal Transformer for Vision-and-Language Navigation](https://proceedings.neurips.cc/paper/2021/hash/2e5c2cb8d13e8fba78d95211440ba326-Abstract.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 9, 10].

Associated files: HAMT.pdf (Main PDF; 14 pages; SHA256 3ab2e9c04bdcc6255ac06c7f0e4b41c4a3f778c2d5edf25d768b22a9cd765899); HAMT (arXiv).pdf (Alternate / duplicate PDF; 23 pages; SHA256 2baed3f1d02fc264398281cfd7d6bdb998cf503cc59a2713e971486566071157); HAMT (supplement).pdf (Supplement; 10 pages; SHA256 04cf30a9a09f504dc01d594cd012616f7ec42dcd0b6086f1afe2e915cb93fe59).

Main-paper analysis is supplemented by the explicitly listed associated-file review below. Exact SHA256 duplicates were deduplicated and not counted as additional evidence. HAMT (supplement).pdf: pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; All supplementary text, objective definitions, ablations, compute and qualitative captions inspected; no independent reproduction.; HAMT (arXiv).pdf: pages [15, 16, 17, 18, 19, 20, 21, 22, 23]; Extended appendix checked against standalone supplement; tables renumbered, same principal experiments. Pages 20–23 qualitative captions inspected; not an independent experiment.

## 2. Problem and operating assumptions

Long navigation histories cannot be reconstructed reliably from a fixed recurrent state. HAMT receives the instruction, all previous panoramic observations/actions, and the current panorama, then predicts the next navigable direction or STOP. Its graph-navigation experiments assume viewpoint localization and candidate directions; those assumptions are distinct from low-level robot control.

## 3. Mechanism, training, and inference

PDF pp 3–5: each panoramic view is encoded by a ViT; a panorama Transformer aggregates its K views, combining pooled panoramic context with the orientation-specific view. A temporal Transformer encodes the resulting history tokens, action directions, and step embeddings. A cross-modal Transformer jointly reasons over text, history, and current candidates. The hidden dimension is 768; reported configuration uses 9 language, 2 panorama, and 4 cross-modal layers. With t steps and K views, hierarchical attention costs O(tK²+t²), versus O(t²K²) for flattened history. This reduces attention cost but does not make memory constant. Pretraining combines masked language/region prediction, instruction–trajectory matching, single-action CE, action-angle squared error, and spatial-relation regression. In the latter, the target is the relative heading/elevation between two views and masking prevents trivial copying. Fine-tuning minimizes imitation CE plus an A3C policy/value/entropy objective; the supervised action term is −Σt log π(a*t|W, Ht, Ot). The reported imitation weight is 0.2 and discount 0.9. Training first freezes the ViT for 200k steps (4 P100), then updates it for 20k steps (20 V100); fine-tuning is 100k steps with batch 8. Inference greedily selects a candidate from the history-conditioned logits without search.

Supplement A.1 (p 1; extended p 15) makes the alignment objectives explicit. For word vocabulary size V, an MLP maps a masked token h_i in R^768 to a V-way distribution and minimizes token negative log likelihood. For each masked visual token, a 1000-way ImageNet teacher distribution P_i supplies soft-target cross entropy −sum_j P_ij log Q_ij, equal to KL up to teacher entropy. ITM scores text-CLS elementwise multiplied by history-CLS, both 768-dimensional, and uses a five-way contrastive objective with one positive, two other-trajectory negatives and two temporal shuffles. Supplement B.3 samples MLM:MRM:ITM:SAP:SAR:SPREL as 5:2:2:1:1:1. For R2R-Back, training requires two stops at distinct stages; success requires visiting the original endpoint before returning, and SPL uses the full reference route length. Ordinary start-goal geodesic SPL would be degenerate for this task.

## 4. Novelty and evidence

Explicit hierarchical history encoding and end-to-end visual adaptation are the principal mechanisms. The paper introduces R2R-Back and R2R-Last to probe history requirements. It does not introduce global graph action selection: history tokens influence a local action policy. Table 2 isolates the hierarchy with the same ResNet feature family; Table 3 separates proxy training, ViT features, and visual fine-tuning.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 2 p 7 reports hierarchical-from-scratch SR 54.4±0.4 versus recurrent 52.2±0.7; Table 3 p 8 shows much larger changes from pretraining and vision adaptation (ResNet no pretraining 54.4 SR; proxy-trained ResNet 63.5; ViT 64.4; end-to-end 65.7). Table 4 gives IL-only 57.2, RL-only 63.5, combined 65.7 SR. VERDICT: explicit history helps under matched features, but attributing the entire headline gain to memory would be incorrect. The cost analysis and limited trajectory lengths do not establish scalability to lifelong navigation.

Supplement C.1/Table 2 (extended Table 11) reports 2349 R2R unseen instructions on one P100: RecBERT 69s, HAMT 104s, and the no-text-to-vision variant 76s. This establishes a measured tradeoff for that setup, not constant history cost. Supplement D.1 compares a reimplemented PREVALENT without history, but also removes ITM, so the 57.5 versus 52.7 SPL contrast does not isolate history alone despite the later phrase “same proxy tasks.” D.4 finds single-stage end-to-end training inferior at the tested schedule (SPL 53.5 versus 56.5 without end-to-end training). REVERIE grounding remains weaker than competing object features; the authors hypothesize that R2R-tuned ViT features transfer poorly to object grounding. Failure captions identify unusual spatial orientation and object recognition errors.

Main Table 7, PDF p 9, compares models retrained on the R2R-Back task: unseen HAMT SR 57.2/SPL 53.1 versus RecBERT 41.1/37.7. This targeted return-to-start task supplies stronger memory evidence than a general R2R leaderboard difference. Table 6 gives R4R nDTW 50.3 versus 45.1 for reimplemented RecBERT. Table 8/p 10 reports CVDN goal progress 5.58m on test; it is not an SPL percentage.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/cshizhe/VLN-HAMT](https://github.com/cshizhe/VLN-HAMT); inspected snapshot **c8b9ee12125f9fe36c51d2ab928fde38f7d846bd**. Snapshot metadata and source are under `repositories/01/hamt`.

Inspected hierarchical history embeddings, multimodal navigation forward path, pretraining heads, and the imitation/RL optimizer loop. No training, dataset preprocessing, checkpoint load, or simulator evaluation was executed.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [finetune_src/models/vilmodel_cmt.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/models/vilmodel_cmt.py) — HistoryEmbeddings lines 523–559; NavCMT lines 624–629: History encodes temporal and panoramic structure and is passed alongside current visual observations into cross-modal navigation.
- [finetune_src/r2r/agent_cmt.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/r2r/agent_cmt.py) — training paths lines 549 and584–602: Imitation and reinforcement updates are explicit; gradients are clipped at40 before optimizer steps.
- [pretrain_src/model/pretrain_cmt.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/pretrain_src/model/pretrain_cmt.py) — task-dispatch and prediction-head definitions: Separate proxy heads implement the paper’s multi-task pretraining; this is training-time supervision, not an inference-time planner.
- [finetune_src/r2r/env.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/r2r/env.py) — evaluation methods: Checked final and oracle graph-distance evaluation, supporting distinction between endpoint SR and route fidelity.
- [finetune_src/r2r/data_utils.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/r2r/data_utils.py) — feature and instruction loaders: Inspected precomputed feature/encoded-instruction loading, not an executed preprocessing pipeline.
- [finetune_src/scripts/run_r2r.sh](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/scripts/run_r2r.sh) — released training/evaluation flags: Configuration snapshot inspected; scripts select task data, model dimensions and checkpoint paths.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **hamt-B1 — Partially addressed**. History storage grows with episode length and temporal attention remains quadratic in t. Evidence: PDF §3 hierarchy and complexity discussion pp 3–5. Type: architectural. Follow-ups: DUET topological map aggregation, GridMM spatial memory. Confidence: high.
- **hamt-B2 — Partially addressed**. A local candidate policy must discover its own multi-step return decisions even with rich history; explicit graph routing is not present. Evidence: PDF action prediction pp 4–5; inspected NavCMT. Type: methodological. Follow-ups: DUET global graph actions. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/cshizhe/VLN-DUET; https://github.com/MrZihan/GridMM
