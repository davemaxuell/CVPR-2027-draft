# Assignment 06 follow-up source cards

Cutoff: 22 October 2026. These compact source summaries distinguish author reports, inspected code and deductions. Full-text review means the listed substantive sections were read, not that unlisted appendices or images were inspected. Limited-scope records explicitly identify inaccessible or partial material. No fetched project code or navigation benchmark was executed. Bibliography, PDF hashes, precise scopes and pinned code are in [sources.json](sources.json).

## FU06-ua_nwm: Uncertainty-Aware World Model for Aerial Image-Goal Navigation

Deyi Zhu; Haoyu Fan; Yinan Zhu; Weichen Zhang; Shilin Ma; Xinlei Chen; Yansong Tang. arXiv preprint. [v1, 6 August 2026](https://arxiv.org/abs/2608.05597v1). Reviewed: 1-8,13-25.

AUTHOR METHOD: A frozen DINOv3 feature world model predicts action-conditioned futures. A separately trained Hierarchical Error Projection head explains residuals through multiscale low-rank subspaces; candidate goals are scored by unexplained residual energy. This is structured error scoring, not calibrated failure probability. PAPER EXPERIMENT: The matched deterministic backbone improves aerial planning; AirSim uses a 20 m goal tolerance and the physical pilot five tasks. Supplement G requires strong anisotropy to motivate ignoring parallel residuals; greedy projections are not a general exact global projection. INSPECTED CODE: normalized training averages patchwise ratios rather than the paper's ratio of summed energies. Optional parallel-residual weighting and same-goal wrong-action contrastive losses already exist, although disabled by default. DEDUCTION: a new decision-aligned objective must compare these controls. Construction M3 below demonstrates a possible lost task direction, not trained-model failure.

Locators: §3 Eqs 8-15; Tables 1, 3; supplementary G-H.

Limitations: No benchmark run; deployment wrappers downloaded but not deeply inspected; scene-disjoint AirGoal split not established; final qualitative figures not visually audited.

## FU06-cpnav: CPNAV: Calibrated Vision-Language Navigation for Object-Goal Search in Unseen Environments

Qizhao Chen; Yuanhong Zeng; Shoh Nishino; Anushri Dixit. Preprint on official author publications page. [Author PDF retrieved 22 October 2026](https://practice-lab-ucla.github.io/papers/cpnav/). Reviewed: 1-8.

AUTHOR METHOD: Offline oracle search optimizes the minimum semantic score along a successful path; an episode-score quantile retains admissible actions for depth-first backtracking. Its stated exchangeability condition concerns complete episodes. PAPER/REPORTING: The text also describes independent observation/action calibration scenarios; the calibration unit, threshold complements and sample counts need reconciliation. Eq 12 uses the .99 upper Beta quantile while claiming 99% confidence, then says confidence 1−δ. The visually checked formula does not support the claimed confidence interpretation (M1). Oracle-path retention alone also does not ensure budgeted success (M5). Target visibility judged manually differs from standard distance-based ObjectNav. The linked official repository was inaccessible. DEDUCTION: do not present its score threshold as a certified collision-risk controller.

Locators: §III.E-F Eqs 11-15; Table I; Tables II-III.

Limitations: Repository not found at retrieval, not proof no code exists; no raw calibration scores; no benchmark executed.

## FU06-adaptive_rollout: Adaptive Rollout Truncation Based on Epistemic Uncertainty for Efficient Offline World Model Training

Nikodem Sebastian Zymla; Laurin Thiele; Johannes Pitz. Accepted IROS 2026 workshop, author-reported; not main conference. [v1, 18 September 2026](https://arxiv.org/abs/2609.21482v1). Reviewed: 1-8.

AUTHOR METHOD: A GRU world model with bootstrap heads or dropout truncates training rollouts when epistemic disagreement crosses a warm-up threshold, bounded between four and 32 steps. PAPER EXPERIMENT: ANYmal/Ant offline prediction and rollout-compute results support efficiency in that training setting. Appendix F explicitly finds that replaying the induced horizon schedule without uncertainty gating matches within seed noise; it disclaims coverage calibration. Appendix C reports no clear downstream MOPO-PPO advantage. DEDUCTION: adaptive horizon proposals require a matched schedule replay control, and prediction efficiency is insufficient evidence for better navigation decisions. No official implementation of this new truncation method was verified; the cited predecessor repository is not substituted for it.

Locators: §III; Table II; Appendices C, F.

Limitations: No new-method source inspection; no deployment experiment reproduced.

## FU06-pijepa: Policy-Guided World Model Planning for Language-Conditioned Visual Navigation

Amirhosein Chahe; Lifeng Zhou. CVPR 2026 WDFM-EAI workshop, pages 834-841. [CVF workshop version](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/papers/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.pdf). Reviewed: 1-8.

AUTHOR METHOD: Language-conditioned Octo proposals initialize MPPI; an action-conditioned JEPA scores terminal image-feature proximity. Training uses frozen image encoders and latent prediction. PAPER EXPERIMENT: CAST validation ATE/RPE are offline trajectory comparisons; proposal scoring nearly matches iterative optimization for several metrics. Closed-loop validation remains future work. INSPECTED CODE: eval_octo_mppi_cast.py:670-703 explicitly takes the recorded final frame as goal. Its defaults differ from the paper's candidate/elite/temperature settings; training defaults to one-step loss and optional rollout losses detach previous predictions. No live controller is evaluated by this script. DEDUCTION: report image-goal access and reranking compute separately when transferring this mechanism to instruction-only VLN.

Locators: §3; §4.2; Table 2; conclusion.

Limitations: No fetched code execution; full Octo training/data pipeline not audited; no checkpoint download.

## FU06-worldmap: WorldMAP: Bootstrapping Vision-Language Navigation Trajectory Prediction with Generative World Models

Hongjin Chen; Shangyun Jiang; Tonghua Su; Chen Gao; Xinlei Chen; Yong Li; Zhibo Chen. arXiv preprint. [v1, 9 April 2026](https://arxiv.org/abs/2604.07957v1). Reviewed: 1-9,11-12; p13 captions only.

AUTHOR METHOD: Generated future views, depth/pose reconstruction, language-guided segmentation and BEV/FMM planning produce teacher trajectories. A Qwen3-VL-based multihypothesis student distills these offline. PAPER EXPERIMENT: Target-Bench uses trajectories projected onto the initial image and pixel-space ADE/FDE/DTW. Pseudo-label-only ADE is 95.98 versus 78.34 for ground truth and 42.06 combined (Table III); gains therefore involve mixed supervision. Training-free direct VLM comparisons also change architecture and data. Appendix B's inference description invokes best-of-K without fully specifying confidence training/selection. DEDUCTION: projected path accuracy does not establish collision-free physical execution or online forecast reliability. No official code located in bounded title/author searches.

Locators: §III; Table III; AppendixA-B.

Limitations: No implementation inspection; no actual closed-loop execution evaluated in reviewed evidence; qualitative figures not visually interpreted.

## FU06-janusvln: JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation

Shuang Zeng; Dekang Qi; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Shiyi Liang; Mu Xu; Xing Wei; Ning Guo. ICLR 2026. [v2, 25 February 2026](https://arxiv.org/abs/2509.22548v2). Reviewed: 1-9,17-21.

AUTHOR METHOD: RGB-only Qwen2.5-VL combines semantic features with frozen VGGT geometry and bounded initial/recent memory. Training includes extra ScaleVLN and DAgger trajectories. PAPER EXPERIMENT: Geometry/cache ablations support the tested architecture; Appendix D acknowledges recovery and premature stopping failures. INSPECTED CODE: VGGT caches first 8/recent 48 frames. Released evaluation instead resamples up to eight historical RGB images over the whole episode and calls the language model afresh; persistent semantic/LLM caches claimed in the paper were not found on that path. DEDUCTION: old observations are neither universally retained nor universally absent after 48 frames. Memory comparisons must specify raw-frame storage, sampling and geometric cache separately.

Locators: §3; Tables 3, 5; AppendixD.

Limitations: No benchmark run; complete optimizer/preprocessing not audited; source path may differ from paper experiments.

## FU06-humanoidvln: HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments

Quan-Dung Pham; Anh Dao; The-Anh Nguyen; Minh Nguyen-Dinh; Phuong Nam Dang; Tri Pham; Hung Tran; Bach Dao; Tuyen P. Le; Truong Nguyen; Quan Nguyen. arXiv preprint; project says Humanoids 2026 under review. [v1, 13 August 2026](https://arxiv.org/abs/2608.12860v1). Reviewed: 1-7.

AUTHOR METHOD: Physics-based humanoid execution combines morphology-specific RL locomotion with PD/MPC tracking and curated large navigable scenes. All 933 episodes form one zero-shot evaluation set. PAPER EXPERIMENT: Fall rates vary substantially by robot/model; action interface, tracker and locomotion policy vary together. The 20-pair real/sim pilot uses one checkpoint and two scenes, explicitly not scene-level generalization. DEDUCTION: the suite makes physical failures measurable, but does not identify an architecture-only cause or solve them. Scene filtering excludes tight environments. Code/data are promised on acceptance, limiting immediate independent replication.

Locators: §III-VI; Table III; §VI.D.

Limitations: No available code/dataset inspection; no robot execution.

## FU06-reinsvln: ReInsVLN: Revisiting Historical Evidence for Training-Free Vision-and-Language Navigation

Siwei Cheng; Hao Wu; Zitong Qi; Xinming Zhang; Haoyu Xi; Peiran Yin; Haozhe Hu; Xudong Wang; Xiaoyu Shen; Wei Zhang. arXiv preprint. [v1, 19 October 2026](https://arxiv.org/abs/2610.26545v1). Reviewed: 1-8.

AUTHOR METHOD: ReInsBench freezes failed navigator checkpoints; deployment combines fixed-history revisits and instruction-aware visual-token compression. PAPER EXPERIMENT: R0 versus neutral reminders improves 19/812 branches, 2.34 pp, with trajectory-cluster 95% CI [-0.99, 5.68]. Diagnostic checkpoint/suffix construction uses reference actions; deployed fixed reminders at steps 2/13 use the original instruction. Seven of 11 pilot net gains already occur at initial registration. FHR also breaks 78/71 previously successful Stream/Intern episodes while recovering 173/126. DEDUCTION: the diagnostic does not isolate restoration of lost history from improved initial grounding; deployment efficacy and explanatory attribution are separate. Its small real-world ordinal comparison changes doorway conditions. Reuse of historical evidence alone is established prior art.

Locators: §III Tables I-II; §IV-V Table IV.

Limitations: Official implementation/supplement not found in bounded search; full eight-page paper reviewed; no experiments reproduced.

## FU06-dartvln: DART-VLN: Test-Time Memory Decay and Anti-Loop Regularization for Discrete Vision-Language Navigation

Shaoheng Zhang; Zhichen Li; Guangfu Ma; Jie Mei. IEEE SMC 2026 accepted, author-reported. [v4, 13 September 2026](https://arxiv.org/abs/2607.01043v4). Reviewed: 1-6.

AUTHOR METHOD: Frozen GridMM receives read-side age/repetition/novelty weighting and finite next-hop reversal/revisit penalties. PAPER EXPERIMENT: Same-checkpoint R2R/REVERIE comparisons improve several metrics; endpoint error is not uniformly improved. INSPECTED CODE: weights enter as log-priors in per-cell softmax; anti-loop penalties subtract from probabilities, with STOP unchanged. No memory-slot pruning occurs in that aggregation path. DEDUCTION: equal floored priors cancel exactly (M4), so this does not erase all stale evidence. Runtime reduction is an empirical report, not a consequence proved by reweighting alone. Finite penalties preserve the possibility of necessary backtracking but can still discourage it.

Locators: Eqs 1-5; Tables I-III; §IV.A.

Limitations: No fetched code execution; full environment metadata update not audited.

## FU06-cmmrvln: CMMR-VLN: Vision-and-Language Navigation via Continual Multimodal Memory Retrieval

Haozhou Li; Xiangyu Dong; Huiyan Jiang; Yaoming Zhou; Xiaoguang Ma. arXiv preprint. [v1, 9 March 2026](https://arxiv.org/abs/2603.07997v1). Reviewed: 1-6.

AUTHOR METHOD: Viewpoint-indexed image/landmark memory retrieves prior experience as high-priority navigation rules; successful routes and first erroneous failure steps update memory. PAPER EXPERIMENT: R2R and 20 physical instructions support gains under this experience-augmented protocol. However the source of first-error labels, initial memory access, episode ordering and learned attention matrix training are not sufficiently specified for a fully label-free interpretation. Ablations use a different subset from main results. DEDUCTION: outcome-aware retrieval is prior art, but changing memory/feedback contracts prevents attributing its gains purely to a retrieval mechanism. No official implementation verified in the bounded search.

Locators: §III.A-C Eq 3; Tables I-III; §IV.C.

Limitations: Feedback provenance and attention training unresolved; no source code inspected.

## FU06-imaginenavpp: ImagineNav++: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination

Teng Wang; Xinxin Zhao; Wenzhe Cai; Changyin Sun. Journal-style arXiv preprint; venue acceptance not independently verified. [v3, 30 April 2026; first version 2025](https://arxiv.org/abs/2512.17435v3). Reviewed: 1-15.

AUTHOR METHOD: A trained Where2Imagine module predicts waypoints; novel-view synthesis supplies candidate images to a frozen VLM, followed by learned VER PointNav. DINOv2-based hierarchical keyframes retain denser recent history. PAPER EXPERIMENT: Table V's full-history control does not improve success; selected history improves efficiency/quality, with an incremental approximation tradeoff. Table I distinguishes generated-future HM3D 58.5/26.6 SR/SPL from oracle-real-future 62.5/32.8. Author failure analysis includes hallucinated objects and ambiguous stopping. DEDUCTION: frozen high-level reasoning is not a globally untrained pipeline. The official repository contains only a coming-soon README at the inspected commit.

Locators: §III; Tables I, III, V; §IV.J.

Limitations: No executable implementation released at inspected snapshot; references16-17 not reviewed.

## FU06-schrodinger_nav: Schrodinger's Navigator: Imagining an Ensemble of Futures for Zero-Shot Object Navigation

Yu He; Da Huang; Zhenyang Liu; Zixiao Gu; Qiang Sun; Guangnan Ye; Yanwei Fu; Yu-Gang Jiang. CoRL 2026, paper and author project. [v5, 16 October 2026](https://arxiv.org/abs/2512.21201v5). Reviewed: 1,3-9,14-20,22-23.

AUTHOR METHOD: FlashWorld imagines left/right/overhead virtual views, aligns depth and combines source-tagged observed/imagined primitives into semantic/exploration/risk maps. These trajectories are complementary cameras, not independent posterior world samples. Imagined semantic cues receive a fixed discount; observations and futures remain separately stored. PAPER EXPERIMENT: Increasing imagination weight can hurt; author limitations deny formal safety guarantees. HM3D uses ShortestPathFollower. Table 1's ImagineNav++ comparator equals its oracle-real-view row; protocol/version clarification is needed. Physical evidence is 30 trials per condition in three rooms. DEDUCTION: source tagging, discounts and multiview imagination alone are not new; calibrated decision risk and actual low-level execution remain distinct questions.

Locators: §3-6; Tables 1-3; AppendicesC-G.

Limitations: No official executable code located; videos not reviewed; qualitative image-only pages not inspected.

## FU06-memoir: Dream to Recall: Imagination-Guided Experience Retrieval for Memory-Persistent Vision-and-Language Navigation

Yunzhe Xu; Yiyuan Pan; Zhe Liu. TPAMI accepted 15 March 2026 per official repository. [v2, 30 March 2026; first version 2025](https://arxiv.org/abs/2510.08553v2). Reviewed: 1,3-13,17-21.

AUTHOR METHOD: A language-conditioned world model queries separate observation/behavior memories; DUET gains a history encoder and three-way action fusion. Persistent graphs permit inter-episode retrieval. PAPER EXPERIMENT: Table V controls instruction/state/random/full retrieval; oracle retrieval remains much stronger. §V.E identifies similar-goal confusion, premature imagination and equal treatment of failed/successful history. Continuous transfer improves success but loses path fidelity versus episodic DUET. INSPECTED CODE: history matching ranks cumulative compatibility products rather than the described prefix-minimum criterion; inference termination threshold differs from the supplement. Teacher paths in the inspected retrieval helper only calculate diagnostics. DEDUCTION: generic future-guided retrieval is established, and confidence-aware retrieval is explicitly suggested future work; a new proposal needs a sharper mechanism/test.

Locators: Algorithms1-3; Tables II, V; §V.E; supplementIV.A.

Limitations: No code execution; full training/loss pipeline not audited; cited theorem derivations not independently certified.

## FU06-neubay: Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism

Tianwei Ni; Esther Derman; Vineet Jain; Vincent Taboga; Siamak Ravanbakhsh; Pierre-Luc Bacon. ICML 2026 PMLR306:92875-92924. [Published proceedings](https://proceedings.mlr.press/v306/ni26a.html). Reviewed: 1-9 main paper only.

AUTHOR METHOD: A history-dependent Bayesian offline-RL policy learns across an ensemble of dynamics models; rollouts stop using in-dataset uncertainty quantiles. Long horizons reduce bootstrap reliance while layer normalization controls predicted increments. PAPER EXPERIMENT: D4RL/NeoRL studies support dataset-dependent benefits; parameters are tuned using true-environment performance and narrow-coverage tasks remain difficult. DEDUCTION: uncertainty-triggered rollout truncation and history-based adaptation are established beyond VLN. Eq 4 bounds prediction magnitudes, not closeness to the true trajectory by itself. No probability-calibration or instruction-following guarantee follows. This is a scoped main-paper comparison, not a review of its extensive supplementary proofs or implementation.

Locators: §4 Algorithms1-2; §5; Eq 4.

Limitations: Full supplement pp10-50 and implementation not reviewed; no reproduction.

## FU06-vla_calibration: Confidence Calibration in Vision-Language-Action Models

Thomas P. Zollo; Richard Zemel. TMLR, October 2026, as stated in final manuscript. [v3, 16 October 2026](https://arxiv.org/abs/2507.17383v3). Reviewed: 1-14,18-21.

AUTHOR METHOD: Mean selected-token probabilities estimate eventual episode success; instruction-paraphrase ensembles and action-wise Platt scaling recalibrate the score. PAPER EXPERIMENT: Fine-tuned token-based VLAs in LIBERO support calibration improvements, not physical safety. Temporal alignment uses realized episode length retrospectively; the proposed progress-aware intervention is qualitative and requires an online progress estimator. Rephrased actions are discarded: original instructions control execution. INSPECTED CODE: each dimension is fitted separately against the same terminal label; the experiment explicitly averages outputs despite a product constructor default. DEDUCTION: neither paraphrase-based confidence nor generic progress-conditioned monitoring is new. Separate risk discrimination, calibration and intervention utility under policy/scene shift.

Locators: §§3.3-3.6; §4.2 pp.8-11; §4.4; §§6-7; Appendix B-C.1.

Limitations: Supplement pp.22-34 not reviewed; no notebook execution or raw rollout inspection; OpenReview verification page inaccessible.

## FU06-reliablenav: ReliableNav: Uncertainty-Aware Zero-Shot Vision-Language Navigation in Continuous Environments

Zhanbo Shi; Lin Zhang; Shengjie Zhao; Yicong Zhou. IEEE TMM Early Access, 21 August 2026. [Retrieved 22 October 2026](https://doi.org/10.1109/TMM.2026.3726310). Reviewed: Full paper inaccessible; publisher abstract and targeted official code only.

AUTHOR ABSTRACT: Atomic action reasoning and a reliability-check map address semantic and sampling uncertainty. INSPECTED CODE: clustered same-class landmark instances receive numbered image masks and trigger another subgoal query; geometry rejects nonnavigable choices. This is explicit disambiguation/replanning, not a verified probability-calibration mechanism. The released configuration defaults to replayed action/LLM logs for 100 R2R-CE episodes. DEDUCTION: generic uncertainty-triggered re-prompting is established prior art. Quantitative paper claims were not independently audited because full text was inaccessible.

Locators: Publisher abstract; official README Usage; released map_handler.py and reliable_agent.py locators.

Limitations: Publisher full text blocked; project paper button says coming soon; full planner and LLM parser not audited; no API or benchmark calls.

## FU06-safety_gating: Uncertainty-Calibrated Safety Gating for Vision-Language-Action Manipulation Under Domain Shift: Reliability Gains and Intervention-Efficiency Trade-Offs

Atef M. Ghaleb; Ali S. Allahloh; Sobhi Mejjaouli; Mohammed A. H. Ali; Adel Al-Shayea. Sensors 26(10), 3140; 15 May 2026. [Retrieved 22 October 2026](https://doi.org/10.3390/s26103140). Reviewed: HTML sections 3.2,4.1-4.3,5.3,6.1-6.4,9 and Tables2-7; partial text review.

AUTHOR METHOD: A calibrated failure-risk score selects normal control, pause/reobserve or a MoveIt fallback. PAPER EXPERIMENT: Two simulated manipulation tasks under structured visual shifts show an intervention-efficiency tradeoff; the more aggressive uncalibrated baseline has stronger raw success/collision results. Threshold sweeps are retrospective, not rerun policies; most interventions use fallback, and residual contacts occur during pause/fallback. The authors explicitly deny a formal safety guarantee. DEDUCTION: a calibrated three-way gate alone is not a new mechanism; equal intervention budgets and closed-loop testing are essential. No released executable implementation or underlying logs were inspected.

Locators: §§4.1-4.3, 5.3, 6.1-6.4; Tables 2-7.

Limitations: Partial HTML review; access intermittent; data offered on request; no code or raw-data audit.
