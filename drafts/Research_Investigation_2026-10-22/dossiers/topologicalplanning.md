# Topological Planning

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Topological Planning with Transformers for Vision-and-Language Navigation**. Kevin Chen, Junshen K. Chen, Jo Chuang, Marynel Vazquez, Silvio Savarese. CVPR 2021.

Primary paper: [Topological Planning with Transformers for Vision-and-Language Navigation](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Topological_Planning_With_Transformers_for_Vision-and-Language_Navigation_CVPR_2021_paper.pdf). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: Topological Planning.pdf (Main PDF; 11 pages; SHA256 b2a1ee579e54ce27b9afceb9c285c22131c89b78d4e5651871268f037ffcbe97).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Separate semantic global planning from low-level control using a topological map built by the agent. Each environment is explicitly explored along ten trajectories before instruction execution. Mapping uses RGB-D panoramas, heading and ground-truth odometry; navigation subsequently uses learned localization.

## 3. Mechanism, training, and inference

Section 3, PDF pp 3–6: sparsify and merge exploration graphs using reachability estimates. Nodes store globally oriented panoramas; edges encode eight directions and three distance bins. A message-passing GNN encodes nodes, edges and global context, followed by map, language and cross-modal Transformer layers. The planner selects a node sequence y with p(y|G, L)=Πt p(yt|y<t, G, L); learned trajectory-position embeddings distinguish selected and unselected nodes. Node/STOP classification uses cross entropy and AdamW. A separately DAgger-trained controller receives previous, current and subgoal panoramas and predicts one of 24 heading sectors plus a scalar distance. A Gaussian bias toward the previous direction suppresses oscillation from perceptual aliasing. Low-level actions turn 15 degrees or move 0.25m. Predicted distance determines when to advance to the next planned node. The full plan is computed before execution, distinct from online exploration.

## 4. Novelty and evidence

Cross-modal global planning on an automatically constructed topology is combined with modular visual localization and control. Crucial stopping prior art appears in Section 4.1, p 6: categorical STOP is compared with a separate binary-cross-entropy STOP head jointly trained with node classification (CMTP-BCE), repeated-node stopping, and a no-STOP control. Binary stop supervision alone is therefore not a new contribution for 2027.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1, p 6: on unseen scenes with agent-generated maps, categorical stopping achieves 26.0% SR and separate BCE 27.3%; on R2R graphs the corresponding values are 28.7% and 30.2%. The no-STOP oracle success rates are 64.0% and 60.5% after 20 planning steps. That control adds exploration and does not isolate softmax normalization as the cause of stopping errors. Table 2, p 7 shows that heading hysteresis improves control: unseen agent-generated-map success rises 82.5→89.6 with variance 3. VERDICT: useful planner, controller and stopping ablations, with pre-exploration and sensor assumptions that prevent direct comparison to unknown-scene, single-run DUET.

Table 3/p 8 reports CMTP-BCE with R2R maps at 28.9 SR/24.1 SPL on unseen environments versus 26.4/22.7 for CMTP, but the autonomous-map setting has different values. Exploration distance is excluded from SPL; collected exploration is roughly 300 times the task navigation distance. Separate binary STOP prediction is therefore prior art, while its total deployment cost and map-availability assumptions differ from online DUET.

## 6. Implementation and reproducibility

The official project https://topovln.stanford.edu/ exposes map assets and an email contact for code instead of a public repository. No email was sent. No executable source or commit was inspected. This is a bounded search limitation, not a claim that no implementation exists.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

**Execution status:** No model source inspection is claimed beyond the explicit scope above. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **topologicalplanning-B1 — Partially addressed**. Stopping and map-quality errors persist even with a pre-explored topology. Separate binary cross entropy helps modestly but does not close the no-STOP oracle gap. Evidence: Table 1 p 6. Type: methodological. Follow-ups: DUET online topological navigation, HSGM double confirmation. Confidence: high.
- **topologicalplanning-B2 — Partially addressed**. Pre-exploration, true mapping odometry and a fixed initial plan move substantial costs and uncertainties outside instruction-time planning. Evidence: §3/§4. Type: protocol/resource. Follow-ups: CM² online map prediction, IVLN persistent memory. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://topovln.stanford.edu/; https://github.com/cshizhe/VLN-DUET
