# Assignment 05 source reviews

Cutoff: 22 October 2026. Each entry is a scoped review; unread pages and unexecuted code are explicit. Machine-readable provenance: [sources.json](sources.json).

## 05-closer — CLOSER: Evidence-Aware Active Visual Verification for Embodied Object-Goal Navigation

Pengying Wu; Mingye Jiang; Yunze Hu; Zhejia Wen; Kangjie Zhou; Zongru Yang; Zhe Peng; Chang Liu. arXiv 2610.28476v1, submitted 20 October 2026.

[Primary full text](https://arxiv.org/pdf/2610.28476v1) · local `followups\sources\closer.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].

AUTHOR METHOD: candidate tracking and diverse view caching feed a trained yes/no/uncertain verifier; uncertainty activates a learned approach policy. PAPER EXPERIMENT: Table IV separates static and active verification; false rejection falls from 15.1% to 9.3%. The method already implements evidence acquisition before committing. Section IV-G documents mixed-instance tracks, persistent occlusion and incorrect forced binary decisions at close range. Physical demonstrations test verification; deployment of the complete approach policy remains future work.

Precise locators: Sections III–IV; Tables IV,VIII; Section IV-G.

Limits: Exact-title/author GitHub searches did not locate an official implementation. No release absence claim beyond this search. Reported HM3D ObjectNav results are not instruction-route VLN evidence.

## 05-vernav — VerNav: Verifier-First Low-Latency Vision-and-Language Navigation

Zhixin Wang; Chengzheyi Yao; Leyuan Liu; Xiaosong Zhang; Yongzhao Zhang. arXiv 2609.00920v1, submitted 1 September 2026.

[Primary full text](https://arxiv.org/pdf/2609.00920v1) · local `followups\sources\vernav.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: batched binary candidate verification includes STOP; normalized action entropy triggers additional model reasoning. Pairwise verifier training includes expert and one-step-deviation contexts, followed by reinforcement fine-tuning. PAPER EXPERIMENT: Table 3 reports the verifier-only system; Table 4 separately charges collaboration latency. The 0.08-second figure excludes caption generation. DEDUCTION: entropy over competing actions does not identify confidently incorrect decisions, and reinterpreting existing observations is distinct from acquiring a new view.

Precise locators: Section 3, Eq.9; Tables 3–4; Figure 6.

Limits: No official repository verified in bounded title/GitHub search. Adaptive results and main-table verifier results use different settings.

## 05-mind_gap — Mind the Gap: Improving Success Rate of Vision-and-Language Navigation by Revisiting Oracle Success Routes

Chongyang Zhao; Yuankai Qi; Qi Wu. ACM Multimedia 2023; reviewed arXiv 2308.03244v1.

[Primary full text](https://arxiv.org/pdf/2308.03244v1) · local `followups\sources\mind_gap.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].

AUTHOR METHOD: MG-VLN scores previously visited endpoints using spatial, heading and temporal visual attention with instruction conditioning; focal plus Dice losses train the ranker. PAPER EXPERIMENT: DUET R2R SR rises 72→75, but actual-return SPL stays 60; retrospectively cropped SPL is a different protocol. REVERIE grounding is unchanged when the precomputed grounding choice is retained. DEDUCTION: ranking cannot recover a suitable endpoint absent from the visited pool. Thus reranking with physical return is established prior art, while endpoint quality and exploration remain separable.

Precise locators: Section 3; Tables 1,2,5,6; RxR appendix.

Limits: No verifiable official implementation located. Author homepage verifies venue; ranking dataset includes generated paths and optional HM3D augmentation.

## 05-retrospective_rectification — Nipping the Drift in the Bud: Retrospective Rectification for Robust Vision-Language Navigation

Gang He; Zhenyang Liu; Kepeng Xu; Li Xu; Tong Qiao; Wenxin Yu; Chang Wu; Weiying Xie. arXiv 2602.06356v1, submitted 6 February 2026.

[Primary full text](https://arxiv.org/pdf/2602.06356) · local `followups\sources\retrospective_rectification.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7].

AUTHOR METHOD: BudVLN uses successful probes for GRPO and hard probes for weighted supervised rectification. It rolls back to the furthest visited reference waypoint, preserves history and generates an oracle suffix; early correction steps receive greater weight. PAPER EXPERIMENT: its StreamVLN ablation gives 57.0/50.5 SR/SPL at baseline, 57.2/50.1 with rectification alone, and 57.6/51.1 for the combination. DEDUCTION: a geometrically selected rollback anchor does not certify which instruction obligations were satisfied.

Precise locators: Section 3; Algorithm 1; Table 2.

Limits: No official implementation verified. Main equation/algorithm sign conventions are ambiguous; no implementation error inferred. Cost comparison is author-reported.

## 05-cast — CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models

Catherine Glossop; William Chen; Arjun Bhorkar; Dhruv Shah; Sergey Levine. Reviewed arXiv 2508.13446v2, 8 June 2026; initial 2025.

[Primary full text](https://arxiv.org/pdf/2508.13446v2) · local `followups\sources\cast.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].

AUTHOR METHOD: alternate language labels for a fixed image are paired with predicted atomic-policy action chunks. The conditional-information argument motivates data design; INSPECTED CODE uses ordinary masked token CE for the main VLA, while the atomic policy uses diffusion MSE. PAPER EXPERIMENT: human checks report 69.4% counterfactual-label accuracy, including collision/grounding criteria. Dynamic-setting degradation and absence of counterfactual future observations are acknowledged limitations. This is direct prior art for same-observation, language-discriminative action augmentation.

Precise locators: Section IV, Lemma 4.1; Section VI, Figure 3; Tables I–II; Appendix B.

Limits: Source inspected, never run. Main VLA code is a separately pinned submodule; training configs reference cloud TFDS assets. Short-horizon robot tasks differ from R2R/RxR routes.

Code: https://github.com/catglossop/CAST at `ec7a214e76167e0f844800ea91d6664863a3d9b1`; inspected files recorded in sources.json. Source inspection only; no model execution.

## 05-rsi_nav — Data-Centric Recursive Self-Improvement for Navigation via Policy-Synchronized Language-Action Dual Bootstrapping

Songze Li; Haoming Gu; Ziyang Gong; Zhitong Wang; Gengze Zhou; Zun Wang; Yuhan Zhu; Tianxiang Jiang; Juanzi Li; Qi Wu; Weinan Zhang; Yu Qiao; Wenwei Zhang; Yi Wang. arXiv 2610.15782v1, submitted 11 October 2026; under review.

[Primary full text](https://arxiv.org/pdf/2610.15782v1) · local `followups\sources\rsi_nav.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19].

AUTHOR METHOD: RSI-Nav synchronizes instruction generation, execution filtering and supervised policy updates in small shards. INSPECTED CODE accepts success/SPL thresholds and then supervises original source-path actions, not the verification rollout. Assistant-response tokens supply training labels. PAPER EXPERIMENT: R2R/RxR improvements include nDTW; update frequency is not monotonically beneficial in Table 3. DEDUCTION: synchronized self-selection still lacks independent route-language validity certification. CLI thresholds (.9/1.0 SPL) differ from the pilot launcher (.4/.4), making the exact launch configuration material.

Precise locators: Section 3, Eq.4; Tables 1,3; appendices.

Limits: No checkpoints, simulator or project scripts executed. Supervised loss is delegated to the loaded Transformers model; its external implementation was not independently audited.

Code: https://github.com/OpenGVLab/RSI-Nav at `5e8fa2c74cf8e7dc305b33088a963b200e20805d`; inspected files recorded in sources.json. Source inspection only; no model execution.

## 05-realm — From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation

Xiangyu Shi; Ruoxi Yang; Wei Tao; Jiwen Zhang; Yanyuan Qiao; Qi Wu. arXiv 2607.03792v1, submitted 4 July 2026.

[Primary full text](https://arxiv.org/pdf/2607.03792) · local `followups\sources\realm.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].

AUTHOR METHOD: REALM adds a post-STOP approach policy, target-span extraction and detector grounding. A visibility-aware margin discourages the first STOP token when ground-truth training visibility is absent. PAPER EXPERIMENT: REVERIE-AIM reports proximity and bounding-box grounding separately; visibility-loss ablation modestly improves approach success. Standard upstream navigation scores are held fixed, so these tables do not measure the complete added movement/latency cost. DEDUCTION: region arrival, visible evidence and correct instance selection are distinct targets.

Precise locators: Sections 3–4; Table 2; ablation of visibility-aware stopping.

Limits: No official code verified. New task/annotations and an extra trained model prevent a direct frozen-DUET comparison. Marginal score ratios do not establish joint conditional failure rates.

## 05-seekvln — Seek Before You Move: Evidence Seeking for Progress Grounding in Vision-Language Navigation

Zhimin Wang; Meiyuan Zhu; Duo Wu; Linjia Kang; Yajun Wang; Yuan Ni; Xiaohang Wang; Tianlu Pan; Jingyan Jiang; Yaowei Wang; Zhi Wang. arXiv 2609.37353v1, submitted 29 September 2026.

[Primary full text](https://arxiv.org/pdf/2609.37353v1) · local `followups\sources\seekvln.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21].

AUTHOR METHOD: SEEK obtains three directional views and textual progress grounding; C2PO clones simulator state, observation history and context to compare forced SEEK/NAV continuations. Their reward difference trains the trigger. PAPER EXPERIMENT: matched-subset ablation supports counterfactual credit beyond unpaired rewards. DEDUCTION: this already occupies generic matched-branch value-of-information training. The objective uses progress/success/SPL without an explicit acquisition-cost penalty; the periodic-trigger comparison does not match query rates.

Precise locators: Section 3, C2PO; Appendix Tables 2,4,6; Figure 3.

Limits: No official code verified. Exact horizon/weight choices and simulated rotation charging are insufficiently specified; physical demonstrations are qualitative.

## 05-cag — When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs

Yu Fang; Yuchun Feng; Dong Jing; Jiaqi Liu; Yue Yang; Zhenyu Wei; Daniel Szafir; Mingyu Ding. arXiv 2602.17659v2, 15 July 2026; initial 19 February 2026.

[Primary full text](https://arxiv.org/pdf/2602.17659v2) · local `followups\sources\cag.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].

AUTHOR METHOD: LIBERO-CF changes feasible instructions at fixed scenes; CAG combines conditional and vision-only action outputs. The prior is obtained by dropping language or training a separate vision-action model. PAPER EXPERIMENT: guidance improves several grounding scores but excessive guidance hurts success. DEDUCTION: Appendix A switches from arithmetic policy combination to log-density combination; these are distinct operations. Its Bayesian interpretation requires score/log-density assumptions and is not a guarantee for arbitrary action interpolation. Manipulation evidence does not establish navigation gains.

Precise locators: Sections III–VI; Tables III–IV; Figure 4; Appendix A, Eqs8–13.

Limits: Official serving code inspected, never executed. It explicitly distinguishes continuous-action interpolation from discrete-logit guidance; the formal caveat does not imply these branches are implementation bugs. No empirical results reproduced.

Code: https://github.com/yuffish/LIBERO-CF at `8460457bfca6e0ef2e856bc104e2c60b023ef2a7`; inspected files recorded in sources.json. Source inspection only; no model execution.

## 05-diagnosing_vln — Diagnosing Vision-and-Language Navigation: What Really Matters

Wanrong Zhu; Yuankai Qi; Pradyumna Narayana; Kazoo Sone; Sugato Basu; Eric Xin Wang; Qi Wu; Miguel Eckstein; William Yang Wang. NAACL 2022.

[Primary full text](https://aclanthology.org/2022.naacl-main.438.pdf) · local `followups\sources\diagnosing_vln.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: controlled token masking/replacement and visual interventions compare nine agents across indoor/outdoor tasks. PAPER EXPERIMENT: indoor HAMT is sensitive to object and direction words beyond random-mask controls. DEDUCTION: reduced performance after corruption supports dependence but cannot distinguish correct grounding from distribution-shift fragility. A valid alternative instruction with a separately known correct action is a stronger test of semantic control.

Precise locators: Sections 3–5; Tables 6,8,12.

Limits: Official repository located but not code-audited: https://github.com/VegB/Diagnose_VLN . Reviewed historical agents do not establish the behavior of 2026 models.

## 05-instruction_evaluation — On the Evaluation of Vision-and-Language Navigation Instructions

Ming Zhao; Peter Anderson; Vihan Jain; Su Wang; Alexander Ku; Jason Baldridge; Eugene Ie. EACL 2021; reviewed arXiv 2101.10504v1.

[Primary full text](https://arxiv.org/pdf/2101.10504) · local `followups\sources\instruction_evaluation.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: a learned instruction/trajectory compatibility score uses hard language/path negatives and contrastive/focal training, compared against human wayfinding outcomes. PAPER EXPERIMENT: ranking synthetic pairs improves downstream learning relative to equal-size random selection, but instance correlations and out-of-domain system rankings remain imperfect. This precedes navigator self-filtering and is a necessary independent-score baseline. DEDUCTION: learned compatibility is evidence, not a semantic oracle.

Precise locators: Sections 3–5; Tables 2–3; Figure 2.

Limits: No official code inspected; appendix-only implementation detail outside the recorded pages was not reviewed. Human judgments are limited to the evaluated instruction systems.

## 05-aps — Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler

Tsu-Jui Fu; Xin Eric Wang; Matthew F. Peterson; Scott T. Grafton; Miguel P. Eckstein; William Yang Wang. ECCV 2020.

[Primary full text](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510069.pdf) · local `followups\sources\aps.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14].

AUTHOR METHOD: APS maximizes the current navigator loss through policy-gradient path sampling, while the navigator minimizes it; a frozen speaker labels sampled paths. Training-scene paths are shortest-path-converted; unseen-scene pre-exploration preserves sampled paths and freezes APS. PAPER EXPERIMENT: equal-count augmentation beats random sampling in the tested models. Footnote 3 acknowledges jointly updating APS during pre-exploration generates unhelpful cycles. DEDUCTION: policy-adaptive hard-data selection is longstanding; high loss alone does not separate informative difficulty from noisy labels.

Precise locators: Sections 3.4–3.5; Algorithm 1; Tables 1–2; Figures 5–8.

Limits: No project implementation audited. Pre-exploration incurs adaptation and scene access beyond ordinary single-episode unseen evaluation.

## 05-counterfactual_unseen — Counterfactual Vision-and-Language Navigation: Unravelling the Unseen

Amin Parvaneh; Ehsan Abbasnejad; Damien Teney; Javen Qinfeng Shi; Anton van den Hengel. NeurIPS 2020.

[Primary full text](https://proceedings.neurips.cc/paper/2020/file/39016cfe079db1bfb359ca72fcba3fd8-Paper.pdf) · local `followups\sources\counterfactual_unseen.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8].

AUTHOR METHOD: adversarially optimized feature interpolation perturbs visual observations while a speaker constrains instruction compatibility; original and perturbed trajectories both train imitation/RL objectives. PAPER EXPERIMENT: matched EnvDrop-family comparisons improve unseen results; interpolation weight has a nonmonotonic optimum. DEDUCTION: feature-space invariance is not a verified scene intervention, nor a guarantee that every landmark remains valid. Same-instruction visual robustness differs from changed-instruction action discrimination.

Precise locators: Section 3, Eqs9–11; Tables 1–3.

Limits: No official method implementation audited. The referenced EnvDrop repository is a baseline, not evidence of the proposed implementation.

## 05-drive_nav — DRIVE-Nav: Directional Reasoning, Inspection, and Verification for Efficient Open-Vocabulary Navigation

Maoguo Gao; Zejun Zhu; Zhiming Sun; Zhengwei Ma; Longze Yuan; Zhongjing Ma; Zhigang Gao; Jinhui Zhang; Suli Zou. arXiv 2603.28691v2, 27 June 2026; initial 30 March.

[Primary full text](https://arxiv.org/pdf/2603.28691) · local `followups\sources\drive_nav.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8].

AUTHOR METHOD: persistent direction tracking schedules partial scans; Qwen3-VL enriches detector prompts and verifies up to three approach frames. If none rejects, an unresolved candidate is accepted. PAPER EXPERIMENT: module ablations hold backbones fixed; rotation-inclusive action analysis uses only common successful long episodes. Physical deployment is a five-category demonstration with slower sparse reasoning calls, not a statistical robot benchmark. DEDUCTION: unresolved evidence handling and complete cost accounting remain testable design choices.

Precise locators: Section IV-C; Tables II–VI; Section V-E.

Limits: Project https://coolmaoguo.github.io/drive-nav-page/ located; no implementation audited. Object-goal results do not validate ordered language-route completion.

## 05-behavior — Behavioral Analysis of Vision-and-Language Navigation Agents

Zijiao Yang; Arjun Majumdar; Stefan Lee. CVPR 2023.

[Primary full text](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Behavioral_Analysis_of_Vision-and-Language_Navigation_Agents_CVPR_2023_paper.pdf) · local `followups\sources\behavior.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: matched teacher-forced histories receive valid stop, turn, object or room interventions; scores measure probability on known correct actions. Hierarchical bootstrap and mixed-effects models address shared trajectories/scenes. PAPER EXPERIMENT: HAMT strongly responds to stopping/directions but weakly to object/room instructions. DEDUCTION: controlled language intervention, correct-action scoring and clustered uncertainty are established prior art; off-policy history and atomic templates limit conclusions about naturally occurring recovery.

Precise locators: Sections 3–5; Figures 3,5–8; Table 1.

Limits: Official code https://github.com/Yoark/vln-behave located; not executed. Main paper reviewed; separate supplement not reviewed.

## 05-stay_path — Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation

Vihan Jain; Gabriel Magalhaes; Alexander Ku; Ashish Vaswani; Eugene Ie; Jason Baldridge. ACL 2019.

[Primary full text](https://aclanthology.org/P19-1181.pdf) · local `followups\sources\stay_path.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: R4R concatenates routes to expose non-shortest instruction following; CLS rewards route coverage and matching length. PAPER EXPERIMENT: fidelity-rewarded RCM uses full instructions more effectively than goal-only training in the tested setting. DEDUCTION: endpoint success and even perfect SPL do not certify the instructed route. CLS itself ignores visitation order in coverage, and its unique-optimum discussion assumes an acyclic reference; ordered predicates still require more than endpoint or unordered coverage.

Precise locators: Sections 2–5; Eqs1–4,11–12; Tables 2–4.

Limits: Official R4R construction link is google-research/google-research/tree/master/r4r; code not inspected. This is a foundational outside-corpus diagnostic, not a post-2026 follow-up.

## 05-ccc — Counterfactual Cycle-Consistent Learning for Instruction Following and Generation in Vision-Language Navigation

Hanqing Wang; Wei Liang; Jianbing Shen; Luc Van Gool; Wenguan Wang. CVPR 2022.

[Primary full text](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Counterfactual_Cycle-Consistent_Learning_for_Instruction_Following_and_Generation_in_Vision-Language_CVPR_2022_paper.pdf) · local `followups\sources\ccc.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8].

AUTHOR METHOD: speaker and follower train through reciprocal cycle losses, including unlabeled paths; a creator mixes visual region features while a discriminator encourages instruction/action compatibility. PAPER EXPERIMENT: Table 4 isolates cycle terms, unlabeled data and counterfactual environments; a single cycle term can hurt navigation. DEDUCTION: mutually consistent models can agree without independent semantic validity, and feature mixing is not rendered physical counterfactual evidence. Joint speaker/follower refinement and counterfactual consistency are therefore established baselines.

Precise locators: Section 3, Eqs6–12; Tables 1–4.

Limits: Official repository https://github.com/HanqingWangAI/CCC-VLN located, not code-audited. Separate supplementary derivation not reviewed.

## 05-rss — Stable Language Guidance for Vision–Language–Action Models

Zhihao Zhan; Yuhao Chen; Jiaying Zhou; Qinhan Lyu; Hao Liu; Keze Wang; Liang Lin; Guangrun Wang. ACL 2026 main conference, pp.4137–4159.

[Primary full text](https://aclanthology.org/2026.acl-long.190.pdf) · local `followups\sources\rss.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: RSS combines sampled instruction reformulations with conditional-minus-unconditional guidance. INSPECTED CODE concatenates spatial-layout and subtask captions and applies guidance to flow velocities. The data intervention therefore adds information as well as changing syntax. PAPER EXPERIMENT: novel-composition results use few-step adaptation; the combined method does not always beat reformulation alone. DEDUCTION: the linearized guidance argument is not a proof of orthogonal visual/language separation. Accuracy under blank instructions cannot independently establish correct language following.

Precise locators: Sections 3–4; Proposition 1; Tables 1,4.

Limits: Pages 10–23, including remaining appendices, not reviewed. No model executed. Caption validity and training-data provenance were not independently reproduced.

Code: https://github.com/Doo-mon/RSS at `f51a0a3f7373b35a61b0690696496066bb5616b8`; inspected files recorded in sources.json. Source inspection only; no model execution.

## 05-uln — ULN: Towards Underspecified Vision-and-Language Navigation

Weixi Feng; Tsu-Jui Fu; Yujie Lu; William Yang Wang. EMNLP 2022.

[Primary full text](https://aclanthology.org/2022.emnlp-main.429.pdf) · local `followups\sources\uln.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: granularity routing and uncertainty-triggered lookahead freeze history during exploration. PAPER EXPERIMENT: HAMT SR rises 43.5→44.6 while SPL falls 39.5→25.9. INSPECTED CODE: the uncertainty MLP uses attended features instead of Eq6 score tensors; labels compare actions with stored result trajectories. Lookahead explicitly sums raw logits. DEDUCTION: exploration/history isolation are established; neither action disagreement nor cross-state raw-logit scores certify semantic ambiguity or intervention benefit.

Precise locators: Sections 3–4; Eq.7; Table 4.

Limits: Pages 10–19 not reviewed by05. HAMT release and exact isolated high-level encoder training remain incomplete. Exploration traces are filtered/spliced from parallel simulator branches; physical cost equivalence was not verified. No project code executed.

Code: https://github.com/weixi-feng/ULN at `0f251df3ffc6dd9c538d075db8674e6ad54a596b`; inspected files recorded in sources.json. Source inspection only; no model execution.

## 05-rulevln — Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification

Jiawen Wen; Penglei Sun; Wenjie Zhang; Suixuan Qiu; Weisheng Xu; Xiaofei Yang; Xiaowen Chu. arXiv 2604.16993v2, 1 July 2026; initial 18 April.

[Primary full text](https://arxiv.org/pdf/2604.16993v2) · local `followups\sources\rulevln.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].

AUTHOR METHOD: Rule-VLN inserts synthetic street regulations, recognizes signs using a regulation bank and gates actions through semantic rules; local geometry redirects movement away from visited traps. PAPER EXPERIMENT: constraint violations decline but remain nonzero; removing geometric rectification reduces violations while sharply hurting completion. DEDUCTION: semantic gates with recovery are existing mechanisms, not guarantees under fallible perception. Outdoor static regulations differ from history-dependent indoor instruction obligations.

Precise locators: Sections 3–6; Tables 2,5.

Limits: Pages 16–20 not reviewed. Official repository https://github.com/Jeffry-wen/Rule-VLN located, not code-audited. Wider-field-of-view base adaptation makes the full pipeline distinct from an entirely training-free system.

## 05-mtrack — One Step at a Time: Long-Horizon Vision-and-Language Navigation with Milestones

Chan Hee Song; Jihyung Kil; Tai-Yu Pan; Brian M. Sadler; Wei-Lun Chao; Yu Su. CVPR 2022; reviewed arXiv 2202.07028.

[Primary full text](https://arxiv.org/pdf/2202.07028) · local `followups\sources\mtrack.pdf` · read PDF pages [3, 4, 5, 6, 7, 8].

AUTHOR METHOD: M-TRACK extracts typed object milestones, checks learned reachability and interaction overlap, and rejects incompatible actions before execution. PAPER EXPERIMENT: controlled train/test tracker combinations support benefits on ALFRED. AUTHOR LIMITATION: procedural commands require action history; hard milestone order restricts alternative valid plans. DEDUCTION: explicit grounded obligation checking and proactive rejection are established mechanisms. The learned detector and fixed object vocabulary remain fallible; this is not a route-VLN or calibrated-verification guarantee.

Precise locators: Sections 4–6; Tables 2–4.

Limits: Pinned release is incomplete: simplified unused checker, missing documented launcher target, AST-confirmed env.py syntax error. No project code imported or executed. Appendix pages 11–16 not reviewed by05; consult original-paper owner04 for any additional associated-material review.

Code: https://github.com/chanhee-luke/M-Track at `a7eaf93fc8631c9d1e7bbef437cc858b5b60f36a`; inspected files recorded in sources.json. Source inspection only; no model execution.

## 05-landmark_rxr — Landmark-RxR: Solving Vision-and-Language Navigation with Fine-Grained Alignment Supervision

Keji He; Yan Huang; Qi Wu; Jianhua Yang; Dong An; Shuanglin Sima; Liang Wang. NeurIPS 2021.

[Primary full text](https://papers.nips.cc/paper/2021/file/0602940f23884f782058efac46f64b0f-Paper.pdf) · local `followups\sources\landmark_rxr.pdf` · read PDF pages [1, 2, 3, 4, 5, 6, 7, 8, 9].

AUTHOR METHOD: human chunk/landmark annotations support fine-grained training and landmark-weighted rewards. Evaluation can reset an agent to a missed landmark and supply the remaining instruction after a failed segment. PAPER EXPERIMENT: focal rewards trade global nDTW against success and reset-based failure counts. DEDUCTION: the reset protocol diagnoses later skills with oracle intervention; it does not demonstrate autonomous recovery. The hard landmark reward counts visits without enforcing their order.

Precise locators: Sections 3–6; Eqs1–4; Tables 2–4.

Limits: Supplement not reviewed. Official repository https://github.com/hekj/Landmark-RxR located; 04 reports data-only release at 6c9c6f5a0313f7c799c5a55538c9540829469031, not independently code-inspected by 05.

## 05-marky — Less is More: Generating Grounded Navigation Instructions from Landmarks

Su Wang; Ceslee Montgomery; Jordi Orbay; Vighnesh Birodkar; Aleksandra Faust; Izzeddin Gur; Natasha Jaques; Austin Waters; Jason Baldridge; Peter Anderson. CVPR 2022; reviewed arXiv 2111.12872v4, 4 April 2022.

[Primary full text](https://arxiv.org/pdf/2111.12872) · local `followups\sources\marky.pdf` · read PDF pages [1, 3, 4, 5, 6, 7, 8].

AUTHOR METHOD: timestamp-biased CTC aligns landmark phrases with views; a route-conditioned detector and multilingual generator produce instructions. PAPER EXPERIMENT: human wayfinding separates predicted/silver landmarks and reveals quality differences obscured by automatic scores. AUTHOR LIMITATION: retrospectively mentioned landmarks violate monotone alignment, and landmark-only generation omits contextual ambiguity. DEDUCTION: navigator success and n-gram similarity are insufficient as sole instruction-quality tests. Grounded generation and auxiliary text rewriting are established.

Precise locators: Sections 3–7; Figure 4; Tables 1–3.

Limits: Appendices not reviewed. No official implementation audited. The supplied generated data are not independently reannotated.

## 05-vln_nf — VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions

Hung-Ting Su; Ting-Jun Wang; Jia-Fong Yeh; Min Sun; Winston H. Hsu. ACL 2026 main conference, pp.25134–25150.

[Primary full text](https://aclanthology.org/2026.acl-long.1152.pdf) · local `followups\sources\vln_nf.pdf` · read PDF pages [3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17].

AUTHOR METHOD: ROAM combines room navigation, coverage-oriented search and FOUND/NOT-FOUND decisions on paired feasible/infeasible goals. REV-SPL includes annotated-object coverage and reference-length factors. PAPER EXPERIMENT: decision failures include room-reaching and perception errors; longer exploration is not uniformly better. AUTHOR LIMITATION: only 45% of REVERIE is retained; absence reasoning remains operational rather than belief-theoretic. DEDUCTION: coverage-aware abstention is established, while annotated-object coverage is not calibrated evidence that a particular target is absent.

Precise locators: Sections 3–5; Eq2; Tables2,5; AppendixA.1,A.4.

Limits: Official project https://vln-nf.github.io/ still says code/data links will be added. No implementation inspected. The paper denies oracle handoff, but exact room-membership inference could not be verified in source. Pages 1–2,10–12 not reviewed.
