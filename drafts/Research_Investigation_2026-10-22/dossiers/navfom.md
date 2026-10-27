# NavFoM

Assignment 03. Investigation cutoff: 2026-10-22.

## Bibliography and reviewed material

**Embodied Navigation Foundation Model**. Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, He Wang ICLR 2026 (2026).

Primary record: https://proceedings.iclr.cc/paper_files/paper/2026/hash/ceb45c48e29e7f49b0b47edb98e43691-Abstract-Conference.html



Supplied main file: [NavFoM.pdf](../../Paper/NavFoM.pdf). Pages actually reviewed: 1,3–10,19,21–22,26,28–30. Associated exact duplicates are identified by SHA256 in the corpus metadata; no claim of separately rereading identical bytes.

## Problem, motivation, and operating assumptions

Existing VLM navigation methods are tied to one camera layout/task/embodiment. NavFoM unifies indoor wheeled/quadruped, drones, vehicles and QA under bounded visual token processing. It still needs task-specific action scaling/controllers and known camera azimuths.

## Method, training, inference, and equations

 DINOv2 + SigLIP 576 patch features concatenate channels; grid pool latest view to 64 patches, history to 4, then 2 layer MLP to Qwen 2-7 B. TVI token E_base+P_time(Time PE(t))+P_angle(Angle PE(φ)) identifies time and camera angle; video QA omit angle, image QA omit both. Sinusoidal PE of sin/cosφpreserves circular identity; sums need not provably be injective overtime and angle. BATS samples history with P(t)=(1−ε)exp(k(t−T)/T)+ε,ε=.1. Eq 5 integral expected frames=(1−ε)(1−exp(−k))T/k+εT; Brent solve k under[(5 E_frames)+65]N≤B_tokens. This is an expectation constraint, not a hard per-instance token cap; floorεmakes feasibility fail at sufficiently large T (authors acknowledge 1120 steps for four cameras 2048 budget). Action hidden state→3 layer MLP predicts M=8 waypoints in R^4(x, y, z, yaw); task scaleαrescales the[-1,1]coordinates; ground vehicles ignore z. L=10 MSE (valid waypoint dimensions)+QA autoregressive CE. MSE mixes position and yaw units after task normalization, so weighting depends units/scales. 12.7 M total examples=8.02 M navigation+3.15 M image QA+1.61 M video QA;56 H100 ×72 h 4032 GPU-hours , one epoch. 8 M navigation includes 3.37 M VLN,1.02 M object goal,897 K tracking,681 K driving and web videos. Simulator uses trajectory actions replacing discrete Forward/Left/Right; deployment costs are version-specific; see Appendix Table 11 below.

Appendix A.2 supplies a practical fallback omitted from the core expectation equation: if the minimum sampling probability exceeds the budget at extremely long horizons, remove the oldest frames. Consequently the unresolved issue is evidence loss under truncation and whether every stochastic draw respects the intended budget, rather than an assertion that the entire method has no fallback. Appendix A.1 uses per-dimension99th-percentile trajectory scales. Appendix B.3 caches four-token history features offline while encoding current/QA images online, which is part of the reported training-efficiency recipe.

## Novelty and closest predecessors

Shared cross-task trajectory head plus explicit time/camera-token organization and budget-dependent history sampling, at much larger data scale. Closest Uni-NaVid unifies navigation tasks with token merging; NaVILA separates high-level navigation/locomotion. Sampling-policy and cross-task benefits require matched data/action representation.

## Results and independent verdict

Table 1 single view R2R 56.2 SR/51.2 SPL , RxR 57.4/49.4; four view 61.7/55.3 and 64.4/56.2. Table 2 unseen OVON 43.6/31.3 single view; different object goal training set, so zero-shot asterisk does not mean untrained navigation. Table 3 tracking 85.0 SR vs Track VLA 85.1 single-target,61.4 vs 57.6 distracted. Table 4 NAV SIM 84.3 PDM S vs PARA Drive 84.0, Uni AD 83.4, LAW 84.6. These broad results support generality, not superiority every metric. Data quantity and changed action space are material confounders.

Figure 8 (p.9) compares BATS, uniform sampling and token merging within the four-camera RxR setting: at 2048 tokens BATS64.4 SR/56.2 SPL versus token merging63.2/54.9; at 1024 tokens BATS62.5/53.9 versus uniform59.7/49.6. These are more informative architectural controls than the headline multi-task numbers. Appendix Table 11 reports RTX4090 FP 16 at 19.8 GB/218 ms and 4bit at 10.7 GB/248 ms;19.1 GB/566 ms refers to Jetson Thor16bit. Appendix F attributes49% of analyzed failures to model issues such as history alignment, early stopping and insufficient turning, and 51% to dataset/simulator issues; the categorization is the authors' analysis. Six cameras slightly underperform four under the same token budget, exposing coverage/history competition.

## Implementation audit

Official project NavFoM -Web links a placeholder github.com/xxxx/xxxx rather than usable implementation at inspection. No NavFoM model, loss, dataset, evaluator code inspected or executed.


## Bottlenecks and follow-up chain

- **NAVFOM-1: Budgeted history sampling and oldest-frame truncation can remove task-critical evidence** Conditions: Long multi camera episodes withεfloor Cause: BATS optimizes an expected recency budget; its practical extreme-horizon fallback removes oldest frames without instruction-conditioned utility Evidence: Main Eqs 4–5; Appendix A.2 truncation; Figure 8 memory controls; Appendix D camera-count tradeoff Type: Methodological / insufficiently tested. Follow-ups:  StreamVLN bounded segments and PolarMem retrieval offer other memory tradeoffs; exact hard budget/task coverage unresolved. Status: **Uncertain because evidence is insufficient**, confidence Moderate.

## Source/access record and remaining uncertainty

The main paper, architecture/data/budget appendices, matched history-token ablations, runtime table and failure analysis were reviewed. The official project linked a code placeholder at inspection, so no implementation or checkpoint was inspected. Exact stochastic budget enforcement and metric code remain unverified; Appendix B.2 also prints an nDTW normalization using start-to-goal distance that should be checked against benchmark code before comparison. No experiment was run.

