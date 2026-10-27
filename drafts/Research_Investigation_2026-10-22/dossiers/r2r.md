# R2R

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments**. Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, Anton van den Hengel. CVPR 2018.

Primary paper: [Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments](https://arxiv.org/abs/1711.07280). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: R2R.pdf (Main PDF; 13 pages; SHA256 b7589865ee573d4753ee9e25bdc494cbbb199e6ec17d0a349091aed0fd86c420).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Define visually grounded natural-language navigation in realistic indoor panoramas. Matterport 3D Simulator uses 90 buildings, approximately 10,800 viewpoints and 194,400 RGB-D images; a graph discretizes traversable locations, so the benchmark does not directly test collision control, continuous localization, or dynamic obstacles.

## 3. Mechanism, training, and inference

PDF pp 3–6: weighted undirected navigation edges are checked by ray tracing/manual inspection, generally between viewpoints within 5m (average spacing 2.25m). R2R samples 7,189 shortest paths of 4–6 edges and longer than 5m and collects 3 instructions per path:21,567 instructions averaging 29 words. Annotators are explicitly told that exact path following is not necessary if the goal is reached. The original baseline is an attention sequence-to-sequence LSTM using ResNet 152 image features and previous action, choosing left/right/up/down 30-degree rotations, forward to a visible neighbor, or STOP. This differs from later panoramic graph-neighbor action abstractions. Evaluation measures endpoint graph distance, success within 3m, and oracle success at the closest visited location, ignoring final orientation. OSR and SR therefore distinguish visiting from stopping, but their difference alone does not identify why stopping failed.

Section 5.2, PDF p 7, trains per-step action cross entropy with the next shortest-path action from the current pose as target. Teacher forcing visits demonstration states; student forcing samples policy actions and relabels the visited states. The language and policy LSTMs each have 512 hidden units, word embeddings 256 dimensions, and action embeddings 32. Adam uses batch 100, dropout .5 and weight decay. Test decoding is greedy; the submitted test model uses both training and validation data, which must be separated from models used to report validation performance.

## 4. Novelty and evidence

A realistic photo-based simulator, connected navigation graph, human route descriptions, and held-out-building evaluation established the VLN task. Endpoint-oriented annotation is an intentional benchmark property, so instruction-path faithfulness requires additional metrics/datasets rather than assuming all goal shortcuts violate R2R’s original contract.

## 5. Results and independent verdict

AUTHOR DATA EVIDENCE: §3–4 provides scene/path/instruction statistics and explicit success definitions. The original split uses 61 train, 11 unseen-validation, 18 test buildings. VERDICT: foundational for controlled semantic navigation, with static graph/localization assumptions and acknowledged scene bias toward tidy, spacious homes (p 4). Later numeric SR improvements should not be interpreted as equivalent progress in real robot navigation.

Table 1, PDF p 7, gives student-forcing SR 38.6% on seen validation, 21.8% on unseen validation and 20.4% on test. Teacher forcing gives 19.6% unseen-validation SR, while the random baseline reaches 13.2% test SR and human navigators 86.4%. Figure 7/p 8 shows unseen performance plateauing while seen performance continues to improve. These measurements establish a concrete generalization gap under the original six-action interface.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/peteanderson80/Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator); inspected snapshot **589d091b111333f9e9f9d6cfd021b2eb68435925**. Snapshot metadata and source are under `repositories/01/r2r`.

Inspected original agent, model, and evaluator source. Their six-action sequence model and graph-distance evaluator provide the reference protocol; later panoramic candidates change action difficulty and observability.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [tasks/R2R/agent.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/agent.py) — Seq2SeqAgent action definitions and rollout: Low-level turn/look/forward/end actions are distinct from later panoramic neighbor actions.
- [tasks/R2R/model.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/model.py) — EncoderLSTM and attention decoder: Sequence encoder and image-conditioned recurrent decoder implement the original baseline.
- [tasks/R2R/eval.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/eval.py) — Evaluation._score_item and score: Uses shortest-path graph distance for endpoint and oracle proximity.
- [tasks/R2R/utils.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/utils.py) — 19–120: Loads connectivity with Euclidean edge weights; tokenizes/lowercases, reverses words and constructs training vocabulary.
- [tasks/R2R/train.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/train.py) — 45–101;105–151: Distinct train-validation and test-submission routines; the latter intentionally combines train/validation data, so it must not be reused to report held-out validation results.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **r2r-B1 — Partially addressed**. Endpoint success can coexist with failure to follow an intended route; OSR–SR cannot isolate softmax competition, grounding, exploration, or goal-estimation causes. Evidence: Annotation instructions p 5 and metric definitions p 6. Type: evaluation/task scope. Follow-ups: RxR dense path grounding, nDTW/SDTW evaluation, REVERIE object grounding. Confidence: high.
- **r2r-B2 — Partially addressed**. Static panoramas, known graph adjacency and exact viewpoint state bypass perception/control/localization failures encountered by robots. Evidence: Simulator construction pp 3–4. Type: operating assumption. Follow-ups: VLN-CE, Cross-Modal Map Learning, BEVBert/VER continuous variants. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/google-research-datasets/RxR; https://github.com/YuankaiQi/REVERIE
