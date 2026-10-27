# Assignment03 source and inspection ledger

Cutoff:22 October 2026. Code inspection is separate from execution. All project-code execution flags are false.

## AdaNav

Title: AdaNav: Adaptive Reasoning with Uncertainty for Vision-Language Navigation

Authors: Xin Ding, Jianyu Wei, Yifan Yang, Shiqi Jiang, Qianxi Zhang, Hao Wu, Fucheng Jia, Liang Mi, Yuxuan Yan, Weijun Wang, Yunxin Liu, Zhibo Chen, Ting Cao

Publication/status: ICML 2026; preprint first posted 2025; year 2026.

Paper sources: https://arxiv.org/abs/2509.24387

Reviewed main pages: 1-8, 12-17

Repository: https://github.com/xinding-sys/AdaNav; commit `None`; status Advertised official repository inaccessible.

Limits: The main paper and relevant appendices were reviewed. The advertised official code URL was inaccessible at the cutoff; no implementation or benchmark was executed.

## Analogical Descriptions

Title: Vision-and-Language Navigation with Analogical Textual Descriptions in LLMs

Authors: Yue Zhang, Tianyi Ma, Zun Wang, Yanyuan Qiao, Parisa Kordjamshidi

Publication/status: EMNLP 2025; year 2025.

Paper sources: https://aclanthology.org/2025.emnlp-main.759/

Reviewed main pages: 1-6

Repository: https://github.com/zhangyuejoslin/VLN-Analogical-Reasoning; commit `5f8ddd9da9f235519956e116436f12b6a7f16734`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- gpt_agent.py:16-22,156-237,340-352
- GPT/gpt_prompt.py:1-32
- scripts/gpt4o.sh:1-17
- eval_utils.py:6-25

Limits: The supplied six-page paper, released comparative prompts, agent and evaluator were reviewed. No API or benchmark run was executed. Exact caption caches, model-release drift, candidate geometry and additional calls remain comparison controls.

## AO-Planner

Title: Affordances-Oriented Planning Using Foundation Models for Continuous Vision-Language Navigation

Authors: Jiaqi Chen, Bingqian Lin, Xinmin Liu, Lin Ma, Xiaodan Liang, Kwan-Yee K. Wong

Publication/status: AAAI 2025; year 2025.

Paper sources: https://ojs.aaai.org/index.php/AAAI/article/download/34526/36681

Reviewed main pages: 1, 3-7

Repository: https://github.com/chen-judge/AO-Planner; commit `719f42a1c9bbb3ebee4815287713ec6c73c0e468`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- vlnce_baselines/zero_shot_agent.py:77-106,255-305,383-386,579-633,690-825,926-939
- vlnce_baselines/models/Policy_ViewSelection_ETP.py:109-214,317-340
- run_r2r/zero_shot.bash:1-24
- llm/grounded_sam_Gemini.py:313-316,430-437
- llm/utils.py:22-69,87-142
- llm/prompting/Gemini.py:14,43-47

Limits: Conference PDF reviewed; arXiv implementation appendix remains an explicit access-scope limitation. No API calls or navigation benchmark executed.

## Aux-Think

Title: Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation

Authors: Shuo Wang, Yongcai Wang, Wanting Li, Xudong Cai, Yucheng Wang, Maiyue Chen, Kaihui Wang, Zhizhong Su, Deying Li, Zhaoxin Fan

Publication/status: NeurIPS 2025; year 2025.

Paper sources: https://proceedings.neurips.cc/paper_files/paper/2025/hash/2c90bf5bffe6497730ecade3a3a37458-Abstract-Conference.html

Reviewed main pages: 1-10, 13-16

Repository: https://github.com/HorizonRobotics/RoboOrchardLab; commit `48ac43987df9ac1ba40fef72dd7a2b4679c592b4`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- robo_orchard_lab/models/aux_think/model.py:32-40,72-110
- robo_orchard_lab/models/aux_think/processor.py:33-156
- projects/aux_think/README.md:1-110

Limits: No benchmark execution. Checklist/reference pages are not additional empirical evidence; remaining appendix generation prompts were not fully audited.

## AwareVLN

Title: AwareVLN: Reasoning with Self-awareness for Vision-Language Navigation

Authors: Wenxuan Guo, Xiuwei Xu, Yichen Liu, Xiangyu Li, Hang Yin, Huangxing Chen, Wenzhao Zheng, Jianjiang Feng, Jie Zhou, Jiwen Lu

Publication/status: CVPR 2026; year 2026.

Paper sources: https://openaccess.thecvf.com/content/CVPR2026/html/Guo_AwareVLN_Reasoning_with_Self-awareness_for_Vision-Language_Navigation_CVPR_2026_paper.html

Reviewed main pages: 1–7,12–15; alternate version4–7

Repository: https://github.com/GWxuan/AwareVLN; commit `8f14eb8450cdd14ff4e3ba1ea04824255121e6be`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- evaluation/vlnce_baselines/awarevln_trainer.py:31-50,81-99,160-267,269-353
- scripts/train/sft_8frames.sh:1-106
- llava/train/train.py:631-653
- llava/model/loss.py:1-48

Limits: The main paper, embedded supplementary annotation/cross-dataset discussion (pp.12–15), alternate proceedings method/results and targeted training/inference code were reviewed. The contradictory RxR training description is preserved as uncertainty. Dataset generation, exact loss masking and all annotation examples were not independently reproduced. No benchmark was executed.

## CA-Nav

Title: Constraint-Aware Zero-Shot Vision-Language Navigation in Continuous Environments

Authors: Kehan Chen, Dong An, Yan Huang, Rongtao Xu, Yifei Su, Yonggen Ling, Ian Reid, Liang Wang

Publication/status: arXiv v4; final journal status not independently verified; year 2026.

Paper sources: https://arxiv.org/abs/2412.10137v4

Reviewed main pages: 1, 3-8

Repository: https://github.com/Chenkehan21/CA-Nav-code; commit `8430359ff930880ccc74131d4c31d74a598a9b6e`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- vlnce_baselines/common/constraints.py:1-113
- vlnce_baselines/ZS_Evaluator_mp.py:115-157,322-343,551-617
- vlnce_baselines/map/value_map.py:34-66,140-175
- vlnce_baselines/map/history_map.py:1-41

Limits: The full v4 method and principal tables plus targeted constraint/value-map/evaluation code were reviewed. Remaining qualitative appendix examples were not exhaustively reviewed. Final journal acceptance was not verified. No benchmark was executed.

## CONSOLE

Title: Correctable Landmark Discovery via Large Models for Vision-Language Navigation

Authors: Bingqian Lin, Yunshuang Nie, Ziming Wei, Yi Zhu, Hang Xu, Shikui Ma, Jianzhuang Liu, Xiaodan Liang

Publication/status: IEEE TPAMI 2024 (author publication record); DOI10.1109/TPAMI.2024.3407759; year 2024.

Paper sources: https://arxiv.org/abs/2405.18721

Reviewed main pages: 1, 3-8

Repository: https://github.com/expectorlin/CONSOLE; commit `7417d4f4d454c8157b913a03b04838f9d4527b54`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- finetune_src/models/vln_clip.py:1-90
- finetune_src/r2r/agent_cmt.py:93-108,347-424,436-439,480-511,700-709

Limits: Principal method and results of the supplied expanded paper and the HAMT/CMT code integration were reviewed. The author publication record confirms TPAMI2024. All alternative integrations and preprocessing assets were not independently reconstructed. No benchmark was executed.

## Context-Nav

Title: Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for InstanceNavigation

Authors: Won Shik Jang, Ue-Hwan Kim

Publication/status: CVPR 2026; year 2026.

Paper sources: https://openaccess.thecvf.com/content/CVPR2026/html/Jang_Context-Nav_Context-Driven_Exploration_and_Viewpoint-Aware_3D_Spatial_Reasoning_for_Instance_CVPR_2026_paper.html

Reviewed main pages: 1-7

Repository: https://github.com/AutoCompSysLab/ContextNav; commit `d7ec95a03c1d8a31aaab8cab9ef460c0172cda81`; status Commit-pinned source retrieved through sparse git checkout.

Inspected files/ranges:

- vlfm/policy/_spatial_reasoning.py:11-323
- vlfm/policy/_ynu_defer.py:15-196,236-309
- vlfm/policy/base_objectnav_policy.py:118-121,170-178,306-310,392-421
- vlfm/measurements/distractor_success.py:1-74

Limits: Main paper reviewed; supplementary failure cases and thresholds not all inspected. No benchmark executed. Predicate feasibility does not prove the narrator's intended viewpoint or true instance identity.

## CorrectNav

Title: CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model

Authors: Zhuoyuan Yu, Yuxing Long, Zihan Yang, Chengyan Zeng, Hongwei Fan, Jiyao Zhang, Hao Dong

Publication/status: AAAI 2026; year 2026.

Paper sources: https://ojs.aaai.org/index.php/AAAI/article/view/38942

Reviewed main pages: 1-7

Repository: None; commit `None`; status No official implementation verified in scoped search.

Limits: The paper mechanism and data/iteration ablations were reviewed. No official executable implementation was verified in the scoped source search. Data regeneration and benchmark reproduction were not attempted.

## DiscussNav

Title: Discuss Before Moving: Visual Language Navigation via Multi-expert Discussions

Authors: Yuxing Long, Xiaoqi Li, Wenzhe Cai, Hao Dong

Publication/status: ICRA 2024; year 2024.

Paper sources: https://sites.google.com/view/discussnav

Reviewed main pages: 1-7

Repository: https://github.com/LYX0501/DiscussNav; commit `1b0c926c25e78122f0459ef50fd87a3889b64580`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- DiscussNav.py:55-92,102-405,407-445,448-455

Limits: The paper and the released single-script method/evaluation path were inspected. No API or robot experiments were executed. The paper's terminology of beam search should not be equated with independent API sampling without checking the backend.

## ECoT

Title: Robotic Control via Embodied Chain-of-Thought Reasoning

Authors: Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, Sergey Levine

Publication/status: CoRL 2024; proceedings published as PMLR270 in 2025; year 2025.

Paper sources: https://proceedings.mlr.press/v270/zawalski25a.html

Reviewed main pages: 1, 3-10

Repository: https://github.com/MichalZawalski/embodied-CoT; commit `1813ad76001f1e08095088f94a86c43fc0e457a3`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- prismatic/vla/datasets/datasets.py:32-48,69-108,137-151
- vla-scripts/finetune.py:167-169,233-247,295-297
- prismatic/models/vlas/openvla.py:33-46,94-111,128-156,188-229
- experiments/bridge/reasoning_server.py:65-81
- experiments/bridge/eval_model_in_bridge_env.py:84-87,126-175

Limits: No robot experiments executed. Appendix training specifics and all 24 pages not fully audited. Do not generalize manipulation evidence into navigation performance.

## EvoNav

Title: History to Future: Evolving Agent with Experience and Thought for Zero-shot Vision-and-Language Navigation

Authors: Guangzhao Dai, Shuo Wang, Zihan Wang, Guo-Sen Xie, Yang Yang, Jinshan Pan, Qianru Sun, Xiangbo Shu

Publication/status: CVPR 2026; year 2026.

Paper sources: https://openaccess.thecvf.com/content/CVPR2026/html/Dai_History_to_Future_Evolving_Agent_with_Experience_and_Thought_for_CVPR_2026_paper.html

Reviewed main pages: 1, 3-7

Repository: None; commit `None`; status No released official implementation verified.

Limits: Main paper and the separate four-page supplement were read. Supplement A.1 uses 100 sampled episodes for R2R-CE and 100 random NavRAG episodes. Section C.3 explicitly says no ground-truth trajectory is supplied to or stored by the agent: feedback comprises its own rollout plus terminal SR/SPL. The critique therefore concerns ground-truth-derived outcome feedback and episode ordering, not use of an oracle path at inference. No verified executable official release was inspected and no benchmark was executed.

## GC-VLN

Title: GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation

Authors: Hang Yin, Haoyu Wei, Xiuwei Xu, Wenxuan Guo, Jie Zhou, Jiwen Lu

Publication/status: CoRL 2025; year 2025.

Paper sources: https://proceedings.mlr.press/v305/yin25a.html

Reviewed main pages: 1, 3-8

Repository: https://github.com/bagh2178/GC-VLN; commit `1beddca8d90d79855c110c5a433f7b8856ca263a`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- src/solver/constraint.py
- src/agent/agent_policy.py:115-143,310-322
- src/solver/navigation_tree.py:76-121
- src/solver/region_solver.py:72-85,141-170,199-215,777-790,905-916
- src/agent/navigation_planner.py:124-234

Limits: Main equations and results read; printed angular Eq.2 appears algebraically different from conventional angle tolerance and needs independent visual equation verification before labeling a mathematical error. No benchmark executed.

## HAM-VLN

Title: HAM-VLN: Harnessing Hierarchical Agentic Memory for Zero-Shot Vision-and-Language Navigation

Authors: An Liu, Bingxi Liu, Hongyu Ding, Yixuan Jiang, Yaran Chen, Fulin Tang, Cong Leng, Hong Zhang, Jian Cheng

Publication/status: arXiv preprint v1,31 July 2026; year 2026.

Paper sources: https://arxiv.org/abs/2607.29600v1

Reviewed main pages: 1, 3-7

Repository: None; commit `None`; status No official implementation verified.

Limits: The supplied method and reported subset experiments were reviewed. No verified official implementation was inspected. Memory fidelity, full API traces and a matched large-scale evaluation remain untested.

## IDEAL-VLN

Title: Instruction Decomposition and Action Alignment for Vision-and-Language Navigation

Authors: Zihao Xin, Wentong Li, Yixuan Jiang, Bin Wang, Piji Li, Jianke Zhu, Jie Qin, Shengjun Huang

Publication/status: ICML 2026, PMLR 306 (paper masthead); year 2026.

Paper sources: https://raw.githubusercontent.com/mlresearch/v306/main/assets/xin26a/xin26a.pdf

Reviewed main pages: 1, 3–8

Repository: None; commit `None`; status Advertised project unavailable; official code unverified.

Limits: The supplied main paper was reviewed. The advertised project URL was inaccessible, so no official implementation, exact dataset pipeline or training run was inspected. The causal language is assessed as a modeling claim, not an identification theorem.

## MapGPT

Title: MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation

Authors: Jiaqi Chen, Bingqian Lin, Ran Xu, Zhenhua Chai, Xiaodan Liang, Kwan-Yee K. Wong

Publication/status: ACL 2024 long papers; year 2024.

Paper sources: https://aclanthology.org/2024.acl-long.529/

Reviewed main pages: 1,3–8,12–13

Repository: https://github.com/chen-judge/MapGPT; commit `7c642f4507a8dde6703e2c61c8fd8ff3bc4bd322`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- GPT/one_stage_prompt_manager.py:41-165,216-352
- vln/gpt_agent.py:149-176

Limits: Main method/results, Appendix A implementation and task prompts (pp.12–13), and the released graph/prompt/action path were inspected. Remaining selected qualitative figures were not exhaustively assessed. No API benchmark was executed.

## MapNav

Title: MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based Vision-and-Language Navigation

Authors: Lingfeng Zhang, Xiaoshuai Hao, Qinwen Xu, Qiang Zhang, Xinyao Zhang, Pengwei Wang, Jing Zhang, Zhongyuan Wang, Shanghang Zhang, Renjing Xu

Publication/status: ACL 2025 long papers; year 2025.

Paper sources: https://aclanthology.org/2025.acl-long.638/

Reviewed main pages: 1,3–8,12–15

Repository: https://github.com/linglingxiansen/MapNav; commit `4a3acc70270688630b4ca8f6df39f26585898e0d`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- mapper.py:13-69,120-177
- r2rnav_agent_nohis.py:180-274,387-473
- caculate_metrics.py:1-43

Limits: Main method/results and Appendix A–E data/training/sensor/ablation details were read, alongside the mapper/action/metric helper code. Complete DAgger regeneration and VLM loss masking were not independently reconstructed. Memory savings refer to the observation representation; no independent runtime or benchmark was measured.

## NavFoM

Title: Embodied Navigation Foundation Model

Authors: Jiazhao Zhang, Anqi Li, Yunpeng Qi, Minghan Li, Jiahang Liu, Shaoan Wang, Haoran Liu, Gengze Zhou, Yuze Wu, Xingxing Li, Yuxin Fan, Wenjun Li, Zhibo Chen, Fei Gao, Qi Wu, Zhizheng Zhang, He Wang

Publication/status: ICLR 2026; year 2026.

Paper sources: https://proceedings.iclr.cc/paper_files/paper/2026/hash/ceb45c48e29e7f49b0b47edb98e43691-Abstract-Conference.html

Reviewed main pages: 1,3–10,19,21–22,26,28–30

Repository: None; commit `None`; status Official project found; linked code placeholder.

Limits: The main paper, architecture/data/budget appendices, matched history-token ablations, runtime table and failure analysis were reviewed. The official project linked a code placeholder at inspection, so no implementation or checkpoint was inspected. Exact stochastic budget enforcement and metric code remain unverified; Appendix B.2 also prints an nDTW normalization using start-to-goal distance that should be checked against benchmark code before comparison. No experiment was run.

## NavGPT

Title: NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models

Authors: Gengze Zhou, Yicong Hong, Qi Wu

Publication/status: AAAI 2024; year 2024.

Paper sources: https://ojs.aaai.org/index.php/AAAI/article/download/28597/29161

Reviewed main pages: 1, 3–7

Repository: https://github.com/GengzeZhou/NavGPT; commit `b3fc8a21b8a1f66b09dde833bcfc25767ef1962d`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- nav_src/agent.py:300-440,558-622
- nav_src/env.py:264-321

Limits: The supplied main paper and targeted prompt/inference/evaluation code were inspected. The complete original caption caches and API backend cannot be reconstructed from this review; additional supplementary examples were not exhaustively reviewed. No benchmark was executed.

## NavGPT-2

Title: NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models

Authors: Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, Qi Wu

Publication/status: ECCV 2024; year 2024.

Paper sources: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01143.pdf

Reviewed main pages: 1, 4–10

Repository: https://github.com/GengzeZhou/NavGPT-2; commit `1f535cbeedfe2aad0a94925a0f9a613fe857f211`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- map_nav_src/models/NavGPT_model.py:283-351,400-597,854-859,890-944
- map_nav_src/models/lavis/models/blip2_models/blip2_t5_instruct_nav.py:78-238
- map_nav_src/r2r/agent.py:32-46,380-461
- map_nav_src/models/lavis/projects/blip2/train/r2r_navgpt_ft_xxl.yaml:1-66
- map_nav_src/scripts/run_r2r_xxl.sh:1-70

Limits: Main method, principal results and targeted Q-Former/LLM/policy/loss/configuration code were inspected. Full feature preprocessing, all appendix experiments and exact pretrained artifacts were not independently reproduced. No project code was executed.

## NavGRPO

Title: Trajectory-Diversity-Driven Robust Vision-and-Language Navigation

Authors: Jiangyang Li, Cong Wan, Songlin Dong, Chenhao Ding, Qiang Wang, Zhiheng Ma, Yihong Gong

Publication/status: CVPR 2026 Findings (explicit supplied-PDF watermark; not main-conference assumption); year 2026.

Paper sources: https://arxiv.org/abs/2603.15370

Reviewed main pages: 1, 3–8

Repository: https://github.com/Songlin-Dong/NavGRPO-CVPR2026; commit `4668d42506858ea971c62e26fdc6e4b71b75064e`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- map_nav_src/r2r/agent_grpo.py:508-665,667-748,750-973
- map_nav_src/r2r/main_nav.py:19-70,150-212
- map_nav_src/scripts/run_r2r.sh:1-75
- README.md:49-63,108-122

Limits: The PDF advertises a different GitHub account from the inspected Songlin-Dong release. This provenance difference and the reward-code discrepancy are recorded; neither proves what private implementation produced the tables. No benchmark was executed.

## NaVid

Title: NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation

Authors: Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, He Wang

Publication/status: Robotics: Science and Systems 2024; year 2024.

Paper sources: https://roboticsproceedings.org/rss20/p079.pdf

Reviewed main pages: 1, 3–7

Repository: https://github.com/jzhzhang/NaVid-VLN-CE; commit `ce2f804a43fb23f1d52431ec24be1acf74c3f4f4`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- navid/model/navid_arch.py:126-196,200-254
- navid/model/language_model/llava_navid.py:55-120
- navid/train/train.py:1165-1186
- agent_navid.py:54-69,128-141,231-238

Limits: Main method/results and targeted video-token/model-loss/train/inference code were inspected. Complete annotation generation and exact evaluation assets were not reconstructed. No independent runtime or robot measurement was made.

## NaVILA

Title: NaVILA: Legged Robot Vision-Language-Action Model for Navigation

Authors: An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, Xiaolong Wang

Publication/status: Robotics: Science and Systems 2025; year 2025.

Paper sources: https://roboticsproceedings.org/rss21/p018.html

Reviewed main pages: Alternate arXiv v2 pages1,3–8 (local image-only proceedings PDF not text-reviewed)

Repository: https://github.com/AnjieCheng/NaVILA; commit `76b98f233dd0fff05dfcd69435eec6740febff9d`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- evaluation/vlnce_baselines/navila_trainer.py:38-52,163-234,313-330
- scripts/train/sft_8frames.sh:10-41

Limits: The local image-only PDF was replaced for text review by author arXiv2412.04453v2; later proceedings changes are not assumed reviewed. High-level inference/SFT configuration and linked low-level observations/rewards/PPO were inspected. Physical calibration, all training assets and robot results were not reproduced. No Isaac or robot experiment was executed.

## PolarMem

Title: PolarMem: A Training-Free Polarized Latent Graph Memory for Verifiable Vision-Language Models

Authors: Zhisheng Chen, Tingyu Wu, Zijie Zhou, Zhengwei Xie, Jinhan Li, Ziyan Weng, Liang Lin, Jingwei Song, Zikai Xiao, Yingwei Zhang

Publication/status: arXiv 2602.00415v2, 30 May 2026; conference acceptance not verified; year 2026.

Paper sources: https://arxiv.org/abs/2602.00415

Reviewed main pages: 1,3–7

Repository: https://github.com/czs-ict/PolarMem; commit `4d0de0befe3571ddeaccafd55ac19aa764ef9b0d`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- src/backbone/qwen_wrapper.py:25-39,56-127
- src/logic/otsu_gate.py:1-48
- src/pipeline/retriever.py:110-139
- src/database/neo4j_client.py:120-146,197-240,284-362

Limits: Core method, principal retrieval results and pinned graph-memory implementation were inspected. Full index construction, all dataset splits and end-to-end retrieval costs were not reproduced. It supplies no VLN performance evidence.

## ProFocus

Title: ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation

Authors: Wei Xue, Mingcheng Li, Xuecheng Wu, Jingqun Tang, Dingkang Yang, Lihua Zhang

Publication/status: CVPR 2026, identified by the main-paper watermark; year 2026.

Paper sources: https://arxiv.org/abs/2603.05530

Reviewed main pages: 1,3–7

Repository: None; commit `None`; status Official implementation unverified.

Limits: Main method, tables and the corresponding alternate-version result pages were read. No official executable code was verified. Exact sampled episode identities, API budgets and loop limits were not established by the reviewed material. No benchmark was executed.

## Progress-Think

Title: Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation

Authors: Shuo Wang, Yucheng Wang, Guoxin Lian, Yongcai Wang, Maiyue Chen, Kaihui Wang, Bo Zhang, Zhizhong Su, Yutian Zhou, Wanting Li, Deying Li, Zhaoxin Fan

Publication/status: CVPR 2026, identified by the main-paper watermark; year 2026.

Paper sources: https://arxiv.org/abs/2511.17097

Reviewed main pages: 1,3–8

Repository: https://github.com/HorizonRobotics/RoboOrchardLab; commit `48ac43987df9ac1ba40fef72dd7a2b4679c592b4`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- robo_orchard_lab/models/progress_think/progress_model.py:32-56
- robo_orchard_lab/models/progress_think/action_model.py:32-56
- projects/progress_think/infer.py:193-205,306-326
- projects/progress_think/analyze_results.py:27-66
- projects/progress_think/README.md:38-119

Limits: The paper objective and the released inference interfaces were inspected. Exact prefix-CE reduction, empty-prefix treatment and joint-GRPO training implementation were not established by the thin released wrappers. No benchmark was executed.

## PSG-Nav

Title: PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making

Authors: Rufeng Chen, Yue Chang, Xiaqiang Tang, Hechang Chen, Sihong Xie

Publication/status: ICML 2026, identified by the PMLR 306 masthead; year 2026.

Paper sources: https://arxiv.org/abs/2606.01313

Reviewed main pages: 1,3–8,13–15,21

Repository: https://github.com/Chenrf1121/psg-nav; commit `181d878ff207845a31e542a19a6592925c1e9bab`; status Source snapshot retrieved; file inspection tracked separately.

Inspected files/ranges:

- graph/scenegraph_combinations.py:60-125
- graph/enumerated_groups.py:60-80,125,252-263
- PSG_Nav.py:1484-1554
- calculate_avg_results.py:7-19,41-52,72-82

Limits: Main method, Appendix A technical/memory/false-positive results and the released world-enumeration/outcome-memory/aggregation code were inspected. Exact episode order and full detector outputs were not independently reconstructed. The selected false-positive subset does not measure calibration on all deployment states. No ObjectNav benchmark was run.

## R3

Title: Run, Ruminate, and Regulate: A Dual-process Thinking System for Vision-and-Language Navigation

Authors: Yu Zhong, Zihao Zhang, Rui Zhang, Lingdong Huang, Haihan Gao, Shuo Wang, Da Li, Ruijian Han, Jiaming Guo, Shaohui Peng, Di Huang, Yunji Chen

Publication/status: AAAI 2026 copyright in the supplied arXiv v1; year 2026.

Paper sources: https://arxiv.org/abs/2511.14131

Reviewed main pages: 1,3–8

Repository: https://github.com/IAII-CAS/navigation_R3; commit `50fca6ed7e29f7543e8cc2f40d0838df061f6286`; status Official repository exists; README-only, implementation unreleased.

Inspected files/ranges:

- README.md:1-2

Limits: No execution. Restart cost/reset protocol, exact repo and scorer training details need additional verification.

## RL4VLA

Title: What Can RL Bring to VLA Generalization? An Empirical Study

Authors: Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, Yu Wang

Publication/status: NeurIPS 2025; reviewed arXiv v4 from January 2026; year 2025.

Paper sources: https://arxiv.org/abs/2505.19789

Reviewed main pages: 1,3–8,17–24

Repository: https://github.com/gen-robot/RL4VLA; commit `de46e0174e7ac310255a2836718fcefab28b8963`; status Commit-pinned source retrieved through sparse git checkout.

Inspected files/ranges:

- SimplerEnv/simpler_env/policies/openvla/openvla_train.py:49-75,104-109,152-163,197-209,273-319
- openvla/prismatic/extern/hf/modeling_prismatic.py:627-642,690-716,824-835
- SimplerEnv/simpler_env/eval_ms3_collect.py:23-44,62-78,90-128

Limits: Main method, Appendix A and additional results in Appendix B were read. The targeted actor/critic/PPO implementation was inspected at the pinned commit, but the full released task suite and evaluation orchestration were not reproduced. All reported benchmark/robot numbers are authors' experiments. No project code was executed.

## SkillNav

Title: Breaking Down and Building Up: Mixture of Skill-Based Vision-and-Language Navigation Agents

Authors: Tianyi Ma, Yue Zhang, Zehao Wang, Parisa Kordjamshidi

Publication/status: ACL 2026, long papers; year 2026.

Paper sources: https://aclanthology.org/2026.acl-long.595/

Reviewed main pages: 1,3–8,10,13–22

Repository: https://github.com/HLR/SkillNav; commit `b56de12a6af07ce09c9252938b1fb69e55d33103`; status Commit-pinned source retrieved through sparse git checkout.

Inspected files/ranges:

- skillnav/backbones/scalevln/maps_nav_src/moe/agent_moe_top1.py:98-104,281-295,595-684,688-879
- skillnav/backbones/scalevln/maps_nav_src/r2r/agent.py:538-592,650-653
- skillnav/backbones/scalevln/maps_nav_src/r2r/env.py:342-352,386-414,447-710
- skillnav/backbones/scalevln/maps_nav_src/scripts/train_r2r_b16_mix_temporal.sh:1-77
- skillnav/backbones/scalevln/maps_nav_src/scripts/test_r2r_b16_moe-top1.sh:1-98

Limits: Appendices A–I (PDF pp.13–22) were inspected for skill definitions, data integrity, training and routing failures. Appendix G/Table 13 reports Temporal DUET alone at 82.59 SR/75.40 SPL, versus full SkillNav 82.59/76.50 in Table 8: most R2R improvement over ScaleVLN 79.23/69.96 already occurs after temporal-data fine-tuning. Appendix I.3 examines 17 failure cases, with the landmark expert optimal in 13 but correctly selected once. The authors' vocabulary-overlap analysis is not a proof against all leakage; they explicitly acknowledge HM3D scene overlap. No benchmark was executed. The inspected batch-routing issue and configuration differences limit reproduction confidence. Appendix Table 10, visually checked in the rendered PDF p.18 (saved as evidence/03/skillnav_table10.png), reports Region DUET SR 36.67 and SPL 43.02. Under standard identical-episode SPL, SPL cannot exceed SR; the discrepancy needs author clarification before that row is used. The audit does not identify its cause. A separate four-line toy dispatch example reproduces the last-expert-over-batch mechanism, but no SkillNav checkpoint or simulator was run.

## StreamVLN

Title: StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling

Authors: Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, Xihui Liu, Jiangmiao Pang

Publication/status: ICRA 2026 according to the official project/repository; reviewed arXiv v2 from July 2026; year 2026.

Paper sources: https://arxiv.org/abs/2507.05240

Reviewed main pages: 1,3–7

Repository: https://github.com/InternRobotics/StreamVLN; commit `e48f6ff7e9201d93aae003e8f64b04c00cec13bc`; status Commit-pinned source retrieved through sparse git checkout.

Inspected files/ranges:

- streamvln/streamvln_agent.py:169-258,265-290
- streamvln/model/stream_video_vln.py:111-152
- streamvln/streamvln_train.py:1743-1766,1816,1852-1855

Limits: The paper's metric-version inconsistencies are recorded. Targeted streaming inference and training wiring were inspected, but complete loss masking, data generation and optional voxel pruning were not exhaustively audited. No independent runtime or benchmark was measured.

## SUSA

Title: Agent Journey Beyond RGB: Hierarchical Semantic-Spatial Representation Enrichment for Vision-and-Language Navigation

Authors: Xuesong Zhang, Yunbo Xu, Jia Li, Ruonan Liu, Zhenzhen Hu

Publication/status: AAAI 2026; year 2026.

Paper sources: https://github.com/HCI-LMC/VLN-SUSA

Reviewed main pages: 1,3–7

Repository: https://github.com/HCI-LMC/VLN-SUSA; commit `03b48cb289650374f85ef017c549ddab5174141a`; status Commit-pinned source retrieved through sparse git checkout.

Inspected files/ranges:

- map_nav_src/models/vilmodel.py:800-810,978-1049
- map_nav_src/r2r/agent.py:27-58,75-120,551-585,659-664
- pretrain_src/model/pretrain_cmt.py:71-124,145-149,198-207,264-269

Limits: The main file and its byte-identical alternate were accounted for. The four-branch model, fine-tuning losses and inherited pretraining dispatch were inspected; complete caption/depth generation and the claimed historical-action pretraining configuration remain unverified. No project code was executed.

## Uni-NaVid

Title: Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks

Authors: Jiazhao Zhang, Kunyu Wang, Shaoan Wang, Minghan Li, Haoran Liu, Songlin Wei, Zhongyuan Wang, Zhizheng Zhang, He Wang

Publication/status: Robotics: Science and Systems 2025; year 2025.

Paper sources: https://roboticsproceedings.org/rss21/p013.html

Reviewed main pages: Alternate arXiv v2:1,3–12 (main method, task results and ablations)

Repository: https://github.com/jzhzhang/Uni-NaVid; commit `79ef5ea3fea14c205342d1ab070563d84c7a966a`; status Commit-pinned source retrieved through sparse git checkout.

Inspected files/ranges:

- uninavid/model/uninavid_arch.py:113-155,168-210
- uninavid/model/language_model/llava_llama_vid.py:103-121

Limits: An accessible author arXiv version, including main task results and TableX ablations, was reviewed instead of the local image-only proceedings PDF. Separate external supplemental videos and any proceedings-only changes were not reviewed. The released merge implementation was inspected; no benchmark or physical robot run was executed.

