# Regretful Agent

Research cutoff: **2026-10-22**. Assignment 01.

## 1. Bibliography, versions, and evidence scope

**The Regretful Agent: Heuristic-Aided Navigation Through Progress Estimation**. Chih-Yao Ma, Zuxuan Wu, Ghassan AlRegib, Caiming Xiong, Zsolt Kira. CVPR 2019.

Primary paper: [The Regretful Agent: Heuristic-Aided Navigation Through Progress Estimation](https://openaccess.thecvf.com/content_CVPR_2019/html/Ma_The_Regretful_Agent_Heuristic-Aided_Navigation_Through_Progress_Estimation_CVPR_2019_paper.html). Page locators below are local PDF page numbers, not proceedings pagination. Inspected pages: [1, 3, 4, 5, 6, 7, 8].

Associated files: Regretful Agent.pdf (Main PDF; 9 pages; SHA256 24260be3fd5b0481154bf1cbca6497f4811e23eb514ed49ee0e8f21073bd058f).

The main PDF was used for the method/result analysis. Distinct associated files are not implied reviewed unless explicitly listed below; exact-hash duplicates add no independent evidence.

## 2. Problem and operating assumptions

Recover from mistaken local actions while avoiding repeated oscillation between previously visited viewpoints. The method extends the self-monitoring agent with a learned forward-versus-rollback decision and node-specific progress memory.

## 3. Mechanism, training, and inference

PDF §4 pp 3–5: compute progress p̂t and its difference Δpt=p̂t−p̂t−1. A learned 2-way softmax gate mixes forward embedding mf (instruction/current state) with rollback embedding mb (previous navigable direction): m=αfmf+αbmb. Progress values are detached before the gate, so action-loss gradients do not reshape the monitor through this path. A dictionary stores each visited node’s last progress; candidate features concatenate the difference between current progress and the stored value, tiled to balance feature scale. Unvisited nodes receive marker 1 and STOP 0. A post-rollback mask prevents immediate oscillation. Train with action CE, positive progress MSE, and entropy regularization βΣaπa logπa (β=.01, negative entropy encourages exploration); inference greedily selects an action. Memory is keyed by exact viewpoint identity, a simulator-provided assumption.

## 4. Novelty and evidence

Learned rollback mixing plus persistent per-node progress markers extends earlier heuristic rollback. Neither storing visited progress nor using progress changes as a recovery trigger is new to a prospective 2027 proposal.

## 5. Results and independent verdict

AUTHOR EXPERIMENT: Table 1 p 6 augmented test SR/SPL 48/40 versus self-monitoring 43/32. Table 2 unaugmented regret-only 47/37, marker-only 44/32, combined 48/37 show both components can help, with non-additive effects. VERDICT: useful local recovery under graph localization; it does not establish calibrated error detection or globally optimal recovery. The distance-derived target inherits the mismatch with semantically necessary detours.

Table 3, PDF p 7, blocks rollback at inference: unseen SR falls from 48 to 46 while SPL rises from 37 to 41. This intervention supports the claimed recovery role and simultaneously exposes its movement cost. It is not evidence that rollback improves all metrics. Table 4 also shows synthetic-only training retaining good oracle visitation while correct stopping remains weaker than real-data training.

## 6. Implementation and reproducibility

Official repository lead: [https://github.com/chihyaoma/regretful-agent](https://github.com/chihyaoma/regretful-agent); inspected snapshot **5caf7b500667981bc7064e4d31b49e83db64c95a**. Snapshot metadata and source are under `repositories/01/regretful`.

Inspected forward/rollback embeddings, detached marker and gate, node memory, entropy and CE/MSE loss, and action rollout.

Phase-specific implementation evidence and remaining gaps are indexed in [the assignment 01 implementation appendix](../evidence/01/implementation_appendix.md).

Inspected code locators:

- [tasks/R2R-pano/models/policy_model.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/models/policy_model.py) — lines 126–157: Detached progress difference drives2-way rollback attention; current-minus-stored progress features affect action logits.
- [tasks/R2R-pano/agents/pano_agent.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/agents/pano_agent.py) — lines 381–437: Per-episode dictionary tracks node progress; positive MSE/CE plus negative-entropy regularization trains the agent.
- [tasks/R2R-pano/trainer.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/trainer.py) — 27–70: Loads training ground-truth paths, selects the model-specific rollout, and backpropagates the returned loss.
- [tasks/R2R-pano/eval.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/eval.py) — 43–95: Final endpoint and nearest visited viewpoint determine success/oracle success; explicit path distances are scored.

**Execution status:** The source files listed above were inspected but not executed. No benchmark reproduction or checkpoint validation was performed.

## 7. Bottlenecks and follow-up status

- **regretful-B1 — Partially addressed**. Local rollback and an oscillation mask do not provide multi-step global route planning; distance progress can misclassify valid detours. Evidence: §4 policy and distance target; code gate. Type: methodological. Follow-ups: FAST frontier search, DUET graph routing. Confidence: high.
- **regretful-B2 — Uncertain because evidence is insufficient**. Progress markers require reliable node identity and do not quantify map localization uncertainty. Evidence: Dictionary keyed by simulator viewpoint. Type: operating assumption. Follow-ups: Continuous VLN mapping methods. Confidence: high.

## 8. Access limitations and source record

No independent numerical benchmark was run. Results remain author-reported. Public release completeness was assessed only in the pinned snapshot, not by installing training dependencies.

Related primary sources: https://github.com/Kelym/FAST; https://github.com/cshizhe/VLN-DUET
