# Self-Monitoring Agent

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**Self-Monitoring Navigation Agent via Auxiliary Progress Estimation**. Chih-Yao Ma, Jiasen Lu, Zuxuan Wu, Ghassan AlRegib, Zsolt Kira, Richard Socher, Caiming Xiong. ICLR 2019.

Primary paper: [Self-Monitoring Navigation Agent via Auxiliary Progress Estimation](https://arxiv.org/pdf/1901.03035). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18].

Associated files: Self-Monitoring Agent.pdf (Main PDF; 18 pages; SHA256 1525968ba70691be86bd72acfda361439fea1faf27d788aefc1932657db41818).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

An agent needs to estimate whether its movement advances instruction completion and to detect deterioration before committing to a wrong branch. The paper estimates progress using supervised distance-to-goal targets rather than human labels of clause completion.

## 3. Mechanism, training, and inference

PDF pp 3–5: a co-grounding LSTM attends to current panoramic views and instruction words. Its progress monitor combines prior hidden state, current memory cell, grounded visual context, and the vector of text-attention weights to predict scalar p̂t∈[−1, 1] using tanh. The target is pt=(d_0−dt)/d_0, overridden to 1 when dt≤3m; it can be negative when moving farther from the goal. Training uses weighted action CE and squared progress error. Although the PDF’s printed Eq 6 has a problematic negative sign on the MSE term, inspected code positively adds λMSE+(1−λ)CE; the implementation therefore minimizes squared error. At inference the monitor weights search candidates and can trigger backtracking when predicted progress decreases. Thus a progress monitor plus rollback is already established prior art.

Supplement PDF p 14 specifies 2176-dimensional view features (2048 ResNet plus 128 tiled angles), 512-dimensional recurrent states, a maximum of 80 instruction tokens, and a 592-to-1 progress projection combining hidden features with attention weights. Adam uses learning rate 1e−4 and batch 64; beam size is 15. The supplement describes physically logging and connecting searched trajectories for challenge submission.

## 4. Novelty and evidence

Auxiliary progress estimation jointly regularizes visual–language grounding and supplies an inference heuristic. The progress signal is distance based; an attention alignment visualization does not prove that the monitor represents semantic instruction completion.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 contains search results whose unrecorded-path and leaderboard movement-accounting interpretations differ: reported 57 SR/51 SPL on the clean search path corresponds to 57 SR/2 SPL when actual exploration is charged; augmented search reaches 61 SR with SPL 2 under that accounting. VERDICT: progress provides useful guidance, but efficiency claims require the correct path protocol. A monotone goal-distance target can penalize a valid instruction-mandated detour.

Main Table 2, PDF p 7, improves unseen greedy SR from 42 to 44 with the progress-training auxiliary, and then to 46 with progress-based inference. Supplement Table 3/p 14 reports augmented single-run test SR/SPL 43/32 for greedy decoding versus 48/35 for progress inference. Supplement pp 15–18 explicitly discuss failure despite apparently high completion, illustrating that progress estimates are imperfect. Text and figure captions were reviewed; the full image sequence was not independently visually adjudicated.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/chihyaoma/selfmonitoring-agent](https://github.com/chihyaoma/selfmonitoring-agent); inspected snapshot **28174ef695fb2483130cb00708ac306e465613ef**. Snapshot metadata and source are under `repositories/01/selfmonitoring`.

Inspected co-grounding policy, target construction, loss sign, rollout, and rollback candidate logic. The code resolves the printed loss-sign ambiguity; no training run was performed.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [tasks/R2R-pano/agents/pano_agent.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/agents/pano_agent.py) — lines 156–181: Target is normalized start-to-goal distance improvement, with≤3m override and positive MSELoss.
- [tasks/R2R-pano/agents/pano_agent.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/agents/pano_agent.py) — lines 311–329 and602–634: Positive weighted CE/MSE training; inference compares progress scores and records exploration/rollback.
- [tasks/R2R-pano/models/policy_model.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/models/policy_model.py) — co-grounding forward definitions lines 57 onward: Visual candidate and language attentions feed the recurrent action policy.
- [tasks/R2R-pano/trainer.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/trainer.py) — 28–70: Loads training ground-truth paths, selects the model-specific rollout, and backpropagates the returned loss.
- [tasks/R2R-pano/eval.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/eval.py) — 43–95: Final endpoint and nearest visited viewpoint determine success/oracle success; explicit path distances are scored.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **selfmonitoring-B1 — Partially addressed**. Distance progress conflates geometric goal approach with following an instruction that may require a detour or intermediate landmark. Evidence: Target formula/code lines 156–169; deduction for non-shortest instructions. Type: supervision. Follow-ups: Regretful Agent, RxR longer path-faithfulness tasks. Confidence: high.
- **selfmonitoring-B2 — Partially addressed**. Progress is not calibrated as a completion probability and search gains incur physical backtracking. Evidence: Tanh regression target; Table 1 protocol. Type: methodological/inference cost. Follow-ups: FAST, DUET. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/chihyaoma/regretful-agent; https://github.com/Kelym/FAST
