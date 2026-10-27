# SAME

SAME: Learning Generic Language-Guided Visual Navigation with State-Adaptive Mixture of Experts

Gengze Zhou; Yicong Hong; Zun Wang; Chongyang Zhao; Mohit Bansal; Qi Wu. ICCV 2025. Investigation cutoff: 2026-10-22.

## Materials and evidence scope

The supplied full paper was reviewed for methods, training, inference, results and limitations using page-labelled PDF text. Relevant supplementary scope is listed below. Code evidence is restricted to pinned inspected files; neither downloaded files nor author claims are represented as executed experiments. No navigation benchmark was run. Supplement download hashes and URLs are recorded in evidence/04/supplements/*manifest.json; source inspection ranges are recorded in evidence/04/raw_source_reads.jsonl and source_read_ranges.json, with separately attributed assignment 05 audits where stated.

Main PDF: [SAME.pdf](../../Paper/SAME.pdf). Pages reviewed: ['3–8'].

## Problem and operating conditions

Naively mixing fine-grained route following, coarse object referring, dialogue and ObjectNav data produces conflicting navigation behavior. SAME aims for one trained policy across tasks, not continual parameter adaptation at inference.

## Mechanism, training, and inference

DUET’s global and local cross-modal encoders are retained. Visual-query projections become sparse MoE layers with eight experts and top-2 routing. Routing input concatenates mean panoramic visual features with language CLS, projects to hidden dimension, and produces expert weights. The mixture output is a weighted sum of selected projections. DAgger supervised action loss is combined with load-balancing loss weighted 0.8. Initialization is ScaleVLN pretraining (4.9 million synthetic examples); fine-tuning samples ScaleVLN:R2R:RxR-EN:CVDN:REVERIE:SOON:Habitat-Web at 10:1:1:1:1:1:2, keeping each batch from one dataset. AdamW lr 1e-5, 20,000 iterations, batch 16, one A100 80 GB. Continuous ObjectNav demonstrations are converted to graph trajectories with endpoint and connectivity filtering; 58,803 remain. Inference uses state-conditioned experts without task identity. Zero-shot CE transfer adds a waypoint predictor, not zero-training end-to-end navigation.

## Novelty and closest predecessors

Targets state-varying rather than token-only, task-only, or language-only routing and locates experts in visual query projections. It is a DUET-based multi-dataset method, distinct from VLM policy backbones and from M3E’s sequential expert consolidation.

## Results and independent verdict

Table 2 compares routing under shared initialization/data: SAME REVERIE SR 43.84 versus no MoE 40.81 and token MoE 42.02; checkpoint variation is reported across five converged checkpoints, not five independent seeds. Removing ScaleVLN initialization reduces R2R SR 73.86→59.89, much larger than the routing gain. Table 4 multi-task SAME R2R SR/SPL 76/66 versus shared-data ScaleVLN baseline 76/67; ObjectNav SR improves 72.3→76.3 but SPL falls 43.4→42.7. Thus it does not improve every metric. Table 5 CE R2R 47 SR/38 SPL trails specialized BEVBert 59/50. Table 6 latency increases DUET 28.78→59.14 ms despite only about 5.2% more FLOPs; parameter/FLOP savings are not equal to latency savings. Table 7 shows batch construction and DAgger effects much larger than small routing differences.

SUPPLEMENT CHECK: Appendix A explicitly separates high-level decision making from low-level collision avoidance. The DUET details and five-dataset results in the supplement do not remove the perception/control assumptions or establish a robust physical controller. The source batch-axis router discrepancy documented below is separate from these paper results.

## Implementation and reproducibility

INSPECTED CODE: CrossmodalEncoder forms routing input from text CLS and masked panorama embeddings and routes the query projection using selected experts (src/models/vilmodel.py:474–558,911–988). default.yaml and main_multi_q.yaml select Task routing, eight experts, top two, mixed R2R/REVERIE ScaleVLN plus PREVALENT/RxR/CVDN/SOON/ObjectNav data, and AdamW imitation/DAgger training. IMPORTANT RELEASE DISCREPANCY: TaskMoeRouting receives [batch, hidden] input, produces [batch, experts] logits, but applies softmax over dimension zero before selecting experts along the last dimension (lines 979–984). DEDUCTION: batch composition can change an example’s routing; with batch size one every pre-top-k expert probability is one, so this path loses state-dependent expert selection. This is a source-level finding, not evidence that the reported paper experiments used this exact release or that their performance is invalid. src/trainers/base_trainer.py:208–280 performs task-weighted imitation/DAgger loss and optimizer updates. Main published evaluation cannot be reproduced confidently until routing-axis intent and checkpoint compatibility are resolved; no benchmark was run.

Inspected release reference: [https://github.com/GengzeZhou/SAME at b74c57bbb189a0ed8c908e90b53d02dfbe417528](https://github.com/GengzeZhou/SAME/tree/b74c57bbb189a0ed8c908e90b53d02dfbe417528). File-level inspection is restricted to the paths listed below.

- src/models/vilmodel.py:474–558,911–988
- src/configs/default.yaml:1–140
- src/configs/main_multi_q.yaml
- src/trainers/base_trainer.py:208–280

## Specific bottlenecks

**same-conflict: Task mixing and expert routing still trade off path efficiency and continuous-environment transfer.**

Conditions: Joint fine-grained and goal-only training, or graph-to-continuous transfer. Cause: Different tasks reward distinct behavior; sparse routing does not remove waypoint/control discrepancies or all gradient conflict. Evidence: PAPER EXPERIMENT: Tables 4, 5 and 7.

Type: methodological. Status: Partially addressed. Confidence: High within the evaluated setting. Follow-ups: ['M3E', 'VLN-MME', 'Rethinking the Embodied Gap']

## Sources and limitations

- [Official implementation and ICCV metadata](https://github.com/GengzeZhou/SAME)
- [Official supplementary or complete author version; reviewed scope listed in associated material](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zhou_SAME_Learning_Generic_ICCV_2025_supplemental.pdf)
- Source has a batch-axis softmax discrepancy. Its impact on published checkpoints/results was not measured. Reported checkpoint variation is not independent-seed uncertainty.
- Associated material: {"file": "evidence/04/supplements/same_supp.txt", "pages_read": "1–3", "scope": "Full-text supplementary methods, protocols and result discussion; qualitative figures reviewed through captions/text unless separately stated. No independent image relabeling or video review."}
