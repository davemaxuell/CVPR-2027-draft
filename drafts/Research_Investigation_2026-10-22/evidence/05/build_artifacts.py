"""Assignment 05 evidence assembly. No imported project code is executed."""
from pathlib import Path
import json, csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/05'

def source(sid,title,authors,year,status,pages,locators,review,limits,repo=None,commit=None,files=None):
    base = ROOT / 'followups/sources' / f'{sid}.json'
    if not base.exists(): base = OUT/'sources'/f'{sid}.json'
    raw=json.loads(base.read_text(encoding='utf-8-sig'))
    pdf=ROOT/'followups/sources'/f'{sid}.pdf'
    if not pdf.exists(): pdf=OUT/'sources'/f'{sid}.pdf'
    return dict(source_id='05-'+sid,title=title,authors=authors.split(';'),year=year,venue_or_status=status,
        primary_url=raw.get('source_url',raw.get('url')),retrieved='2026-10-22',cutoff='2026-10-22',
        local_pdf=str(pdf.relative_to(ROOT)),sha256=raw['sha256'],pdf_pages=raw['pages'],
        full_text_pages_actually_inspected=pages,evidence_locators=locators,scoped_review=review,
        repository_url=repo,repository_commit=commit,code_files_inspected=files or [],
        code_executed=False,benchmark_executed=False,limitations=limits,review_status='Substantive scoped primary-source review')

S=[]
S.append(source('closer','CLOSER: Evidence-Aware Active Visual Verification for Embodied Object-Goal Navigation',
 'Pengying Wu;Mingye Jiang;Yunze Hu;Zhejia Wen;Kangjie Zhou;Zongru Yang;Zhe Peng;Chang Liu',2026,'arXiv 2610.28476v1, submitted 20 October 2026',[1,2,3,4,5,6,7,8,9,10,11,12,13],
 ['Sections III–IV; Tables IV,VIII; Section IV-G'],
 'AUTHOR METHOD: candidate tracking and diverse view caching feed a trained yes/no/uncertain verifier; uncertainty activates a learned approach policy. PAPER EXPERIMENT: Table IV separates static and active verification; false rejection falls from 15.1% to 9.3%. The method already implements evidence acquisition before committing. Section IV-G documents mixed-instance tracks, persistent occlusion and incorrect forced binary decisions at close range. Physical demonstrations test verification; deployment of the complete approach policy remains future work.',
 ['Exact-title/author GitHub searches did not locate an official implementation. No release absence claim beyond this search. Reported HM3D ObjectNav results are not instruction-route VLN evidence.']))
S.append(source('vernav','VerNav: Verifier-First Low-Latency Vision-and-Language Navigation',
 'Zhixin Wang;Chengzheyi Yao;Leyuan Liu;Xiaosong Zhang;Yongzhao Zhang',2026,'arXiv 2609.00920v1, submitted 1 September 2026',list(range(1,10)),
 ['Section 3, Eq.9; Tables 3–4; Figure 6'],
 'AUTHOR METHOD: batched binary candidate verification includes STOP; normalized action entropy triggers additional model reasoning. Pairwise verifier training includes expert and one-step-deviation contexts, followed by reinforcement fine-tuning. PAPER EXPERIMENT: Table 3 reports the verifier-only system; Table 4 separately charges collaboration latency. The 0.08-second figure excludes caption generation. DEDUCTION: entropy over competing actions does not identify confidently incorrect decisions, and reinterpreting existing observations is distinct from acquiring a new view.',
 ['No official repository verified in bounded title/GitHub search. Adaptive results and main-table verifier results use different settings.']))
S.append(source('mind_gap','Mind the Gap: Improving Success Rate of Vision-and-Language Navigation by Revisiting Oracle Success Routes',
 'Chongyang Zhao;Yuankai Qi;Qi Wu',2023,'ACM Multimedia 2023; reviewed arXiv 2308.03244v1',list(range(1,12)),
 ['Section 3; Tables 1,2,5,6; RxR appendix'],
 'AUTHOR METHOD: MG-VLN scores previously visited endpoints using spatial, heading and temporal visual attention with instruction conditioning; focal plus Dice losses train the ranker. PAPER EXPERIMENT: DUET R2R SR rises 72→75, but actual-return SPL stays 60; retrospectively cropped SPL is a different protocol. REVERIE grounding is unchanged when the precomputed grounding choice is retained. DEDUCTION: ranking cannot recover a suitable endpoint absent from the visited pool. Thus reranking with physical return is established prior art, while endpoint quality and exploration remain separable.',
 ['No verifiable official implementation located. Author homepage verifies venue; ranking dataset includes generated paths and optional HM3D augmentation.']))
S.append(source('retrospective_rectification','Nipping the Drift in the Bud: Retrospective Rectification for Robust Vision-Language Navigation',
 'Gang He;Zhenyang Liu;Kepeng Xu;Li Xu;Tong Qiao;Wenxin Yu;Chang Wu;Weiying Xie',2026,'arXiv 2602.06356v1, submitted 6 February 2026',list(range(1,8)),
 ['Section 3; Algorithm 1; Table 2'],
 'AUTHOR METHOD: BudVLN uses successful probes for GRPO and hard probes for weighted supervised rectification. It rolls back to the furthest visited reference waypoint, preserves history and generates an oracle suffix; early correction steps receive greater weight. PAPER EXPERIMENT: its StreamVLN ablation gives 57.0/50.5 SR/SPL at baseline, 57.2/50.1 with rectification alone, and 57.6/51.1 for the combination. DEDUCTION: a geometrically selected rollback anchor does not certify which instruction obligations were satisfied.',
 ['No official implementation verified. Main equation/algorithm sign conventions are ambiguous; no implementation error inferred. Cost comparison is author-reported.']))
S.append(source('cast','CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models',
 'Catherine Glossop;William Chen;Arjun Bhorkar;Dhruv Shah;Sergey Levine',2026,'Reviewed arXiv 2508.13446v2, 8 June 2026; initial 2025',list(range(1,13)),
 ['Section IV, Lemma 4.1; Section VI, Figure 3; Tables I–II; Appendix B'],
 'AUTHOR METHOD: alternate language labels for a fixed image are paired with predicted atomic-policy action chunks. The conditional-information argument motivates data design; INSPECTED CODE uses ordinary masked token CE for the main VLA, while the atomic policy uses diffusion MSE. PAPER EXPERIMENT: human checks report 69.4% counterfactual-label accuracy, including collision/grounding criteria. Dynamic-setting degradation and absence of counterfactual future observations are acknowledged limitations. This is direct prior art for same-observation, language-discriminative action augmentation.',
 ['Source inspected, never run. Main VLA code is a separately pinned submodule; training configs reference cloud TFDS assets. Short-horizon robot tasks differ from R2R/RxR routes.'],
 'https://github.com/catglossop/CAST','ec7a214e76167e0f844800ea91d6664863a3d9b1',[
 'cast/data/generate_cast_dataset.py','cast/data/utils/action_generation.py','cast/data/prompts/cf_step_prompt.txt','cast/data/prompts/filter_step_prompt.txt','cast/atomic_model/train/training/train_utils.py:240–282','cast/deployment/navigate_vla.py:1–220',
 'Submodule catglossop/cast-vla@dfc67f43ae69fdd0b0f27005cf469b8b8e9ac9fe: configs/cast_config.py; src/palivla/train_step.py; src/palivla/dataset.py:1–52; src/palivla/standardization_transforms.py:883–924; README.md']))
S.append(source('rsi_nav','Data-Centric Recursive Self-Improvement for Navigation via Policy-Synchronized Language-Action Dual Bootstrapping',
 'Songze Li;Haoming Gu;Ziyang Gong;Zhitong Wang;Gengze Zhou;Zun Wang;Yuhan Zhu;Tianxiang Jiang;Juanzi Li;Qi Wu;Weinan Zhang;Yu Qiao;Wenwei Zhang;Yi Wang',2026,'arXiv 2610.15782v1, submitted 11 October 2026; under review',list(range(1,10))+[14,15,16,17,18,19],
 ['Section 3, Eq.4; Tables 1,3; appendices'],
 'AUTHOR METHOD: RSI-Nav synchronizes instruction generation, execution filtering and supervised policy updates in small shards. INSPECTED CODE accepts success/SPL thresholds and then supervises original source-path actions, not the verification rollout. Assistant-response tokens supply training labels. PAPER EXPERIMENT: R2R/RxR improvements include nDTW; update frequency is not monotonically beneficial in Table 3. DEDUCTION: synchronized self-selection still lacks independent route-language validity certification. CLI thresholds (.9/1.0 SPL) differ from the pilot launcher (.4/.4), making the exact launch configuration material.',
 ['No checkpoints, simulator or project scripts executed. Supervised loss is delegated to the loaded Transformers model; its external implementation was not independently audited.'],
 'https://github.com/OpenGVLab/RSI-Nav','5e8fa2c74cf8e7dc305b33088a963b200e20805d',[
 'evo/evo_rollout.py:1490–1540,1570–1600','evo/evo_utils.py:1013–1105','evo/r2r_single.py:682–765','evo/run_8gpu_pilot.sh:1–92','qwenvl/data/data_processor.py:289–376','qwenvl/train/train_qwen.py:80–190']))
S.append(source('realm','From Region Arrival to Instance-Level Grounding in Vision-and-Language Navigation',
 'Xiangyu Shi;Ruoxi Yang;Wei Tao;Jiwen Zhang;Yanyuan Qiao;Qi Wu',2026,'arXiv 2607.03792v1, submitted 4 July 2026',list(range(1,11)),
 ['Sections 3–4; Table 2; ablation of visibility-aware stopping'],
 'AUTHOR METHOD: REALM adds a post-STOP approach policy, target-span extraction and detector grounding. A visibility-aware margin discourages the first STOP token when ground-truth training visibility is absent. PAPER EXPERIMENT: REVERIE-AIM reports proximity and bounding-box grounding separately; visibility-loss ablation modestly improves approach success. Standard upstream navigation scores are held fixed, so these tables do not measure the complete added movement/latency cost. DEDUCTION: region arrival, visible evidence and correct instance selection are distinct targets.',
 ['No official code verified. New task/annotations and an extra trained model prevent a direct frozen-DUET comparison. Marginal score ratios do not establish joint conditional failure rates.']))
S.append(source('seekvln','Seek Before You Move: Evidence Seeking for Progress Grounding in Vision-Language Navigation',
 'Zhimin Wang;Meiyuan Zhu;Duo Wu;Linjia Kang;Yajun Wang;Yuan Ni;Xiaohang Wang;Tianlu Pan;Jingyan Jiang;Yaowei Wang;Zhi Wang',2026,'arXiv 2609.37353v1, submitted 29 September 2026',list(range(1,10))+list(range(12,22)),
 ['Section 3, C2PO; Appendix Tables 2,4,6; Figure 3'],
 'AUTHOR METHOD: SEEK obtains three directional views and textual progress grounding; C2PO clones simulator state, observation history and context to compare forced SEEK/NAV continuations. Their reward difference trains the trigger. PAPER EXPERIMENT: matched-subset ablation supports counterfactual credit beyond unpaired rewards. DEDUCTION: this already occupies generic matched-branch value-of-information training. The objective uses progress/success/SPL without an explicit acquisition-cost penalty; the periodic-trigger comparison does not match query rates.',
 ['No official code verified. Exact horizon/weight choices and simulated rotation charging are insufficiently specified; physical demonstrations are qualitative.']))
S.append(source('cag','When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs',
 'Yu Fang;Yuchun Feng;Dong Jing;Jiaqi Liu;Yue Yang;Zhenyu Wei;Daniel Szafir;Mingyu Ding',2026,'arXiv 2602.17659v2, 15 July 2026; initial 19 February 2026',list(range(1,15)),
 ['Sections III–VI; Tables III–IV; Figure 4; Appendix A, Eqs8–13'],
 'AUTHOR METHOD: LIBERO-CF changes feasible instructions at fixed scenes; CAG combines conditional and vision-only action outputs. The prior is obtained by dropping language or training a separate vision-action model. PAPER EXPERIMENT: guidance improves several grounding scores but excessive guidance hurts success. DEDUCTION: Appendix A switches from arithmetic policy combination to log-density combination; these are distinct operations. Its Bayesian interpretation requires score/log-density assumptions and is not a guarantee for arbitrary action interpolation. Manipulation evidence does not establish navigation gains.',
 ['Official serving code inspected, never executed. It explicitly distinguishes continuous-action interpolation from discrete-logit guidance; the formal caveat does not imply these branches are implementation bugs. No empirical results reproduced.'],
 'https://github.com/yuffish/LIBERO-CF','8460457bfca6e0ef2e856bc104e2c60b023ef2a7',[
 'openvla-oft-cf/serving/serve_openvla_oft_TF.py:1–65','openvla-oft-cf/serving/serve_openvla_oft_VA.py:1–84','eval/main_cf.py:181–250']))
S.append(source('diagnosing_vln','Diagnosing Vision-and-Language Navigation: What Really Matters',
 'Wanrong Zhu;Yuankai Qi;Pradyumna Narayana;Kazoo Sone;Sugato Basu;Eric Xin Wang;Qi Wu;Miguel Eckstein;William Yang Wang',2022,'NAACL 2022',list(range(1,10)),
 ['Sections 3–5; Tables 6,8,12'],
 'AUTHOR METHOD: controlled token masking/replacement and visual interventions compare nine agents across indoor/outdoor tasks. PAPER EXPERIMENT: indoor HAMT is sensitive to object and direction words beyond random-mask controls. DEDUCTION: reduced performance after corruption supports dependence but cannot distinguish correct grounding from distribution-shift fragility. A valid alternative instruction with a separately known correct action is a stronger test of semantic control.',
 ['Official repository located but not code-audited: https://github.com/VegB/Diagnose_VLN . Reviewed historical agents do not establish the behavior of 2026 models.']))
S.append(source('instruction_evaluation','On the Evaluation of Vision-and-Language Navigation Instructions',
 'Ming Zhao;Peter Anderson;Vihan Jain;Su Wang;Alexander Ku;Jason Baldridge;Eugene Ie',2021,'EACL 2021; reviewed arXiv 2101.10504v1',list(range(1,10)),
 ['Sections 3–5; Tables 2–3; Figure 2'],
 'AUTHOR METHOD: a learned instruction/trajectory compatibility score uses hard language/path negatives and contrastive/focal training, compared against human wayfinding outcomes. PAPER EXPERIMENT: ranking synthetic pairs improves downstream learning relative to equal-size random selection, but instance correlations and out-of-domain system rankings remain imperfect. This precedes navigator self-filtering and is a necessary independent-score baseline. DEDUCTION: learned compatibility is evidence, not a semantic oracle.',
 ['No official code inspected; appendix-only implementation detail outside the recorded pages was not reviewed. Human judgments are limited to the evaluated instruction systems.']))
S.append(source('aps','Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler',
 'Tsu-Jui Fu;Xin Eric Wang;Matthew F. Peterson;Scott T. Grafton;Miguel P. Eckstein;William Yang Wang',2020,'ECCV 2020',list(range(1,15)),
 ['Sections 3.4–3.5; Algorithm 1; Tables 1–2; Figures 5–8'],
 'AUTHOR METHOD: APS maximizes the current navigator loss through policy-gradient path sampling, while the navigator minimizes it; a frozen speaker labels sampled paths. Training-scene paths are shortest-path-converted; unseen-scene pre-exploration preserves sampled paths and freezes APS. PAPER EXPERIMENT: equal-count augmentation beats random sampling in the tested models. Footnote 3 acknowledges jointly updating APS during pre-exploration generates unhelpful cycles. DEDUCTION: policy-adaptive hard-data selection is longstanding; high loss alone does not separate informative difficulty from noisy labels.',
 ['No project implementation audited. Pre-exploration incurs adaptation and scene access beyond ordinary single-episode unseen evaluation.']))
S.append(source('counterfactual_unseen','Counterfactual Vision-and-Language Navigation: Unravelling the Unseen',
 'Amin Parvaneh;Ehsan Abbasnejad;Damien Teney;Javen Qinfeng Shi;Anton van den Hengel',2020,'NeurIPS 2020',list(range(1,9)),
 ['Section 3, Eqs9–11; Tables 1–3'],
 'AUTHOR METHOD: adversarially optimized feature interpolation perturbs visual observations while a speaker constrains instruction compatibility; original and perturbed trajectories both train imitation/RL objectives. PAPER EXPERIMENT: matched EnvDrop-family comparisons improve unseen results; interpolation weight has a nonmonotonic optimum. DEDUCTION: feature-space invariance is not a verified scene intervention, nor a guarantee that every landmark remains valid. Same-instruction visual robustness differs from changed-instruction action discrimination.',
 ['No official method implementation audited. The referenced EnvDrop repository is a baseline, not evidence of the proposed implementation.']))
S.append(source('drive_nav','DRIVE-Nav: Directional Reasoning, Inspection, and Verification for Efficient Open-Vocabulary Navigation',
 'Maoguo Gao;Zejun Zhu;Zhiming Sun;Zhengwei Ma;Longze Yuan;Zhongjing Ma;Zhigang Gao;Jinhui Zhang;Suli Zou',2026,'arXiv 2603.28691v2, 27 June 2026; initial 30 March',list(range(1,9)),
 ['Section IV-C; Tables II–VI; Section V-E'],
 'AUTHOR METHOD: persistent direction tracking schedules partial scans; Qwen3-VL enriches detector prompts and verifies up to three approach frames. If none rejects, an unresolved candidate is accepted. PAPER EXPERIMENT: module ablations hold backbones fixed; rotation-inclusive action analysis uses only common successful long episodes. Physical deployment is a five-category demonstration with slower sparse reasoning calls, not a statistical robot benchmark. DEDUCTION: unresolved evidence handling and complete cost accounting remain testable design choices.',
 ['Project https://coolmaoguo.github.io/drive-nav-page/ located; no implementation audited. Object-goal results do not validate ordered language-route completion.']))
S.append(source('behavior','Behavioral Analysis of Vision-and-Language Navigation Agents',
 'Zijiao Yang;Arjun Majumdar;Stefan Lee',2023,'CVPR 2023',list(range(1,10)),
 ['Sections 3–5; Figures 3,5–8; Table 1'],
 'AUTHOR METHOD: matched teacher-forced histories receive valid stop, turn, object or room interventions; scores measure probability on known correct actions. Hierarchical bootstrap and mixed-effects models address shared trajectories/scenes. PAPER EXPERIMENT: HAMT strongly responds to stopping/directions but weakly to object/room instructions. DEDUCTION: controlled language intervention, correct-action scoring and clustered uncertainty are established prior art; off-policy history and atomic templates limit conclusions about naturally occurring recovery.',
 ['Official code https://github.com/Yoark/vln-behave located; not executed. Main paper reviewed; separate supplement not reviewed.']))
S.append(source('stay_path','Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation',
 'Vihan Jain;Gabriel Magalhaes;Alexander Ku;Ashish Vaswani;Eugene Ie;Jason Baldridge',2019,'ACL 2019',list(range(1,10)),
 ['Sections 2–5; Eqs1–4,11–12; Tables 2–4'],
 'AUTHOR METHOD: R4R concatenates routes to expose non-shortest instruction following; CLS rewards route coverage and matching length. PAPER EXPERIMENT: fidelity-rewarded RCM uses full instructions more effectively than goal-only training in the tested setting. DEDUCTION: endpoint success and even perfect SPL do not certify the instructed route. CLS itself ignores visitation order in coverage, and its unique-optimum discussion assumes an acyclic reference; ordered predicates still require more than endpoint or unordered coverage.',
 ['Official R4R construction link is google-research/google-research/tree/master/r4r; code not inspected. This is a foundational outside-corpus diagnostic, not a post-2026 follow-up.']))
S.append(source('ccc','Counterfactual Cycle-Consistent Learning for Instruction Following and Generation in Vision-Language Navigation',
 'Hanqing Wang;Wei Liang;Jianbing Shen;Luc Van Gool;Wenguan Wang',2022,'CVPR 2022',list(range(1,9)),
 ['Section 3, Eqs6–12; Tables 1–4'],
 'AUTHOR METHOD: speaker and follower train through reciprocal cycle losses, including unlabeled paths; a creator mixes visual region features while a discriminator encourages instruction/action compatibility. PAPER EXPERIMENT: Table 4 isolates cycle terms, unlabeled data and counterfactual environments; a single cycle term can hurt navigation. DEDUCTION: mutually consistent models can agree without independent semantic validity, and feature mixing is not rendered physical counterfactual evidence. Joint speaker/follower refinement and counterfactual consistency are therefore established baselines.',
 ['Official repository https://github.com/HanqingWangAI/CCC-VLN located, not code-audited. Separate supplementary derivation not reviewed.']))

S.append(source('rss','Stable Language Guidance for Vision–Language–Action Models',
 'Zhihao Zhan;Yuhao Chen;Jiaying Zhou;Qinhan Lyu;Hao Liu;Keze Wang;Liang Lin;Guangrun Wang',2026,'ACL 2026 main conference, pp.4137–4159',list(range(1,10)),
 ['Sections 3–4; Proposition 1; Tables 1,4'],
 'AUTHOR METHOD: RSS combines sampled instruction reformulations with conditional-minus-unconditional guidance. INSPECTED CODE concatenates spatial-layout and subtask captions and applies guidance to flow velocities. The data intervention therefore adds information as well as changing syntax. PAPER EXPERIMENT: novel-composition results use few-step adaptation; the combined method does not always beat reformulation alone. DEDUCTION: the linearized guidance argument is not a proof of orthogonal visual/language separation. Accuracy under blank instructions cannot independently establish correct language following.',
 ['Pages 10–23, including remaining appendices, not reviewed. No model executed. Caption validity and training-data provenance were not independently reproduced.'],
 'https://github.com/Doo-mon/RSS','f51a0a3f7373b35a61b0690696496066bb5616b8',[
 'src/openpi/models/pi0_cfg.py:159–202,255–286,292–376','src/openpi/transforms.py:309–336','src/openpi/training/misc/libero_cfg_config.py:20–103','src/openpi/policies/libero_policy.py:105–145','README.md:1–90']))
S.append(source('uln','ULN: Towards Underspecified Vision-and-Language Navigation',
 'Weixi Feng;Tsu-Jui Fu;Yujie Lu;William Yang Wang',2022,'EMNLP 2022',list(range(1,10)),
 ['Sections 3–4; Eq.7; Table 4'],
 'AUTHOR METHOD: granularity routing and uncertainty-triggered lookahead freeze history during exploration. PAPER EXPERIMENT: HAMT SR rises 43.5→44.6 while SPL falls 39.5→25.9. INSPECTED CODE: the uncertainty MLP uses attended features instead of Eq6 score tensors; labels compare actions with stored result trajectories. Lookahead explicitly sums raw logits. DEDUCTION: exploration/history isolation are established; neither action disagreement nor cross-state raw-logit scores certify semantic ambiguity or intervention benefit.',
 ['Pages 10–19 not reviewed by05. HAMT release and exact isolated high-level encoder training remain incomplete. Exploration traces are filtered/spliced from parallel simulator branches; physical cost equivalence was not verified. No project code executed.'],
 'https://github.com/weixi-feng/ULN','0f251df3ffc6dd9c538d075db8674e6ad54a596b',[
 'ULN-Rec-VLN-BERT/r2r_src/agent.py:109–145,191–228,249–352,438–511,516–780,851–895,986–996,1038–1140,1200–1290,1373–1412,1468–1487',
 'ULN-Rec-VLN-BERT/r2r_src/model.py:796–882','ULN-Rec-VLN-BERT/r2r_src/model_PREVALENT.py:1–165','ULN-Rec-VLN-BERT/r2r_src/vlnbert/vlnbert_PREVALENT.py:350–486',
 'ULN-Rec-VLN-BERT/r2r_src/train.py:37–100,190–245,260–290,390–450','ULN-Rec-VLN-BERT/r2r_src/env.py:EnvBatch and R2RBatch preprocessing, normalized logical lines1–235',
 'ULN-Rec-VLN-BERT/r2r_src/eval.py:1–200','ULN-Rec-VLN-BERT/r2r_src/param.py:78–112','ULN-Rec-VLN-BERT/r2r_src/vlnbert/vlnbert_init.py',
 'ULN-Rec-VLN-BERT/run/train_explorer.bash; run/train_agent_mix.bash; run/test_agent.bash; run/test_agent_final.bash; README.md']))
S.append(source('rulevln','Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification',
 'Jiawen Wen;Penglei Sun;Wenjie Zhang;Suixuan Qiu;Weisheng Xu;Xiaofei Yang;Xiaowen Chu',2026,'arXiv 2604.16993v2, 1 July 2026; initial 18 April',list(range(1,16)),
 ['Sections 3–6; Tables 2,5'],
 'AUTHOR METHOD: Rule-VLN inserts synthetic street regulations, recognizes signs using a regulation bank and gates actions through semantic rules; local geometry redirects movement away from visited traps. PAPER EXPERIMENT: constraint violations decline but remain nonzero; removing geometric rectification reduces violations while sharply hurting completion. DEDUCTION: semantic gates with recovery are existing mechanisms, not guarantees under fallible perception. Outdoor static regulations differ from history-dependent indoor instruction obligations.',
 ['Pages 16–20 not reviewed. Official repository https://github.com/Jeffry-wen/Rule-VLN located, not code-audited. Wider-field-of-view base adaptation makes the full pipeline distinct from an entirely training-free system.']))

S.append(source('mtrack','One Step at a Time: Long-Horizon Vision-and-Language Navigation with Milestones',
 'Chan Hee Song;Jihyung Kil;Tai-Yu Pan;Brian M. Sadler;Wei-Lun Chao;Yu Su',2022,'CVPR 2022; reviewed arXiv 2202.07028',list(range(3,9)),
 ['Sections 4–6; Tables 2–4'],
 'AUTHOR METHOD: M-TRACK extracts typed object milestones, checks learned reachability and interaction overlap, and rejects incompatible actions before execution. PAPER EXPERIMENT: controlled train/test tracker combinations support benefits on ALFRED. AUTHOR LIMITATION: procedural commands require action history; hard milestone order restricts alternative valid plans. DEDUCTION: explicit grounded obligation checking and proactive rejection are established mechanisms. The learned detector and fixed object vocabulary remain fallible; this is not a route-VLN or calibrated-verification guarantee.',
 ['Pinned release is incomplete: simplified unused checker, missing documented launcher target, AST-confirmed env.py syntax error. No project code imported or executed. Appendix pages 11–16 not reviewed by05; consult original-paper owner04 for any additional associated-material review.'],
 'https://github.com/chanhee-luke/M-Track','a7eaf93fc8631c9d1e7bbef437cc858b5b60f36a',[
 'mtrack_src/milestone_checker.py:1–30','mtrack_src/milestone.py:18–95','mtrack_src/milestone/src/model/transformer_crf.py:14–43','mtrack_src/vision/alfred_head.py:31–123','mtrack_src/vnn.py:156–203','mtrack_src/env.py:1–165','mtrack_src/agent.py:96–115,236–345','mtrack_src/train.py:1–90,250–268','mtrack_src/eval.py:17–89','run/train_agent.bash','run/test_agent.bash']))
S.append(source('landmark_rxr','Landmark-RxR: Solving Vision-and-Language Navigation with Fine-Grained Alignment Supervision',
 'Keji He;Yan Huang;Qi Wu;Jianhua Yang;Dong An;Shuanglin Sima;Liang Wang',2021,'NeurIPS 2021',list(range(1,10)),
 ['Sections 3–6; Eqs1–4; Tables 2–4'],
 'AUTHOR METHOD: human chunk/landmark annotations support fine-grained training and landmark-weighted rewards. Evaluation can reset an agent to a missed landmark and supply the remaining instruction after a failed segment. PAPER EXPERIMENT: focal rewards trade global nDTW against success and reset-based failure counts. DEDUCTION: the reset protocol diagnoses later skills with oracle intervention; it does not demonstrate autonomous recovery. The hard landmark reward counts visits without enforcing their order.',
 ['Supplement not reviewed. Official repository https://github.com/hekj/Landmark-RxR located; 04 reports data-only release at 6c9c6f5a0313f7c799c5a55538c9540829469031, not independently code-inspected by 05.']))
S.append(source('marky','Less is More: Generating Grounded Navigation Instructions from Landmarks',
 'Su Wang;Ceslee Montgomery;Jordi Orbay;Vighnesh Birodkar;Aleksandra Faust;Izzeddin Gur;Natasha Jaques;Austin Waters;Jason Baldridge;Peter Anderson',2022,'CVPR 2022; reviewed arXiv 2111.12872v4, 4 April 2022',[1,3,4,5,6,7,8],
 ['Sections 3–7; Figure 4; Tables 1–3'],
 'AUTHOR METHOD: timestamp-biased CTC aligns landmark phrases with views; a route-conditioned detector and multilingual generator produce instructions. PAPER EXPERIMENT: human wayfinding separates predicted/silver landmarks and reveals quality differences obscured by automatic scores. AUTHOR LIMITATION: retrospectively mentioned landmarks violate monotone alignment, and landmark-only generation omits contextual ambiguity. DEDUCTION: navigator success and n-gram similarity are insufficient as sole instruction-quality tests. Grounded generation and auxiliary text rewriting are established.',
 ['Appendices not reviewed. No official implementation audited. The supplied generated data are not independently reannotated.']))

S.append(source('vln_nf','VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions',
 'Hung-Ting Su;Ting-Jun Wang;Jia-Fong Yeh;Min Sun;Winston H. Hsu',2026,'ACL 2026 main conference, pp.25134–25150',list(range(3,10))+list(range(13,18)),
 ['Sections 3–5; Eq2; Tables2,5; AppendixA.1,A.4'],
 'AUTHOR METHOD: ROAM combines room navigation, coverage-oriented search and FOUND/NOT-FOUND decisions on paired feasible/infeasible goals. REV-SPL includes annotated-object coverage and reference-length factors. PAPER EXPERIMENT: decision failures include room-reaching and perception errors; longer exploration is not uniformly better. AUTHOR LIMITATION: only 45% of REVERIE is retained; absence reasoning remains operational rather than belief-theoretic. DEDUCTION: coverage-aware abstention is established, while annotated-object coverage is not calibrated evidence that a particular target is absent.',
 ['Official project https://vln-nf.github.io/ still says code/data links will be added. No implementation inspected. The paper denies oracle handoff, but exact room-membership inference could not be verified in source. Pages 1–2,10–12 not reviewed.']))

OUT.mkdir(parents=True,exist_ok=True)
(OUT/'sources.json').write_text(json.dumps(S,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Wrote {len(S)} source records')
