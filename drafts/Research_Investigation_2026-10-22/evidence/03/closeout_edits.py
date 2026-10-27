from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'evidence'/'03'
paths=list(E.glob('notes_*.json'))
notes={n['id']:n for p in paths for n in json.loads(p.read_text(encoding='utf-8'))}
audits=json.loads((E/'code_audits.json').read_text(encoding='utf-8'))

# Repair technical terms that generic word segmentation damaged. Preserve URL/code identifiers.
replacements={
 'Nav ig ation':'Navigation','sum mari z ation':'summarization','45°F oV':'45° field of view',
 'L XM ERT':'LXMERT','GPT3 .5':'GPT-3.5','GPT4 o':'GPT-4o','GPT4 V':'GPT-4V','GPT4V':'GPT-4V','GPT4o':'GPT-4o',
 'EVA - CLIP':'EVA-CLIP','EVA -CLIP':'EVA-CLIP','CLIP -B 16/ BERT':'CLIP-B/16 and BERT',
 'Q-former':'Q-Former','latent s':'latents','waypointpredictor':'waypoint predictor',
 'son either':'so neither','HAM T':'HAMT','LLaMA -VID':'LLaMA-VID','<OB S>':'<OBS>','< NAV >':'<NAV>',
 'En v Drop':'EnvDrop','AS M':'ASM','vox el':'voxel','Dvo x els':'D voxels','o dome try':'odometry','o dom':'odometry',
 'travers able':'traversable','embodiment s':'embodiments','re balanced':'rebalanced','Entail ment':'Entailment',
 'un modeled':'unmodeled','ensemble d':'ensembled','conflate s':'conflates','degradation s':'degradations',
 'backbonegeneral':'backbone general','BM RAG':'B MRAG','MM MU':'MMMU','iterative ly':'iteratively',
 'BD-M CT S':'BD-MCTS','M CT S':'MCTS','Deep See kV 3':'DeepSeek-V3','GL M 4.5 V':'GLM-4.5V',
 'Qwen 3 VLM ax':'Qwen3-VL-Max','re implemented':'reimplemented','un established':'unestablished',
 'monotonic it y':'monotonicity','pre train stage slr':'pretraining stage learning rate',
 'ratio sEq':'ratios in Eq.','Object Nav':'ObjectNav','G LIP':'GLIP','GroundedSAM':'Grounded-SAM',
 'detection s':'detections','conform alor':'conformal or','pro b graph':'probabilistic graph','eva l':'evaluation',
 'retain sGT':'retains GT','Authorprotocol':'Author protocol','SuppliedPDF':'Supplied PDF',
 'Grid MM':'GridMM','GN N':'GNN','orG NN':'or GNN','forrest':'for the rest','spl its':'splits',
 'R3\'shand off':'R3\'s handoff','hand off':'handoff','ver i fier':'verifier','R2 R TLand':'R2R TL and',
 'counter factual':'counterfactual','Man iS kill':'ManiSkill','Widow X':'WidowX','end-effect or':'end-effector',
 'POM DP':'POMDP','M DP':'MDP','robot in it':'robot initialization','temporalreordering':'temporal reordering',
 'Nav Nuances':'NavNuances','APIcalls':'API calls','finegrained':'fine-grained','clip s':'clips',
 'MP3 D':'MP3D','R2 R':'R2R','KV QA':'K VQA','GPUh':'GPU-hours','re ce n cy':'recency',
 'Require sGT':'Requires GT','complement ari ty':'complementarity','con cat':'concatenation',
 'standalonegenerative':'standalone generative','ML M':'MLM','Image Net':'ImageNet','chun king':'chunking',
 'HM3 D':'HM3D','a sync':'asynchronous','short term':'short-term','long term':'long-term',
 'Accessibletextreplacement for image-onlylocalPDF':'Accessible text replacement for the local image-only PDF',
 'Textreplacement for image-onlylocalPDF':'Text replacement for the local image-only PDF',
 'NeurIPS2025; reviewedarxivv4Jan2026':'NeurIPS 2025; reviewed arXiv v4 from January 2026',
 'ICML2026, PMLR306masthead':'ICML 2026, identified by the PMLR 306 masthead',
 'AAAI2026copyright in suppliedarxivv1':'AAAI 2026 copyright in the supplied arXiv v1',
 'ICRA2026fromofficialproject/repo;reviewedarxivv2July2026':'ICRA 2026 according to the official project/repository; reviewed arXiv v2 from July 2026',
 'CVPR2026mainpaperwatermark':'CVPR 2026, identified by the main-paper watermark','AAAI2026':'AAAI 2026',
 'ACL2026longpapers':'ACL 2026, long papers','arXiv2602.00415v2,30May2026':'arXiv 2602.00415v2, 30 May 2026',
 'Robotics: Science and Systems2025':'Robotics: Science and Systems 2025',
 '1 st/99 th':'1st/99th','8 D oF':'8 DoF','Data set':'Dataset','data set':'dataset',
}
def clean(x):
 if isinstance(x,str):
  if x.startswith(('http://','https://')): return x
  for a,b in replacements.items(): x=x.replace(a,b)
  return x
 if isinstance(x,list):return [clean(y) for y in x]
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 return x
notes={k:clean(v) for k,v in notes.items()};audits=clean(audits)

audits['ecot']={
 'code': "INSPECTED CODE. prismatic/vla/datasets/datasets.py:32–48 controls reasoning dropout. Its RLDS transform at69–108 combines the image, instruction, annotated reasoning and tokenized action; labels are copied from input_ids, with optional EOS masking. This inspected transform does not independently mask the instruction prompt. Lines137–151 specify a one-frame window, no depth/proprioception and action normalization by quantiles. vla-scripts/finetune.py:167–169,233–247,295–297 runs AdamW, the model loss, backward and optimizer steps. It is a released fine-tuning path, not verification of the entire original pretraining run.\n\nopenvla.py:94–111 builds the task prompt;128–156 and188–205 implement reuse of asynchronously generated reasoning;215–229 converts seven discrete action tokens into normalized continuous controls. The reasoning server at experiments/bridge/reasoning_server.py:65–81 loads the model and serves raw generation over ZMQ. The Bridge evaluator at84–87 defaults to a2 Hz control loop,600 steps and500 episodes;139–165 times action calls. These files substantiate the reasoning/action interface and deployment mechanism. Complete original reasoning-label generation, all pretraining configurations and physical robot results were not independently reproduced. No project code was executed.",
 'code_files':['prismatic/vla/datasets/datasets.py:32-48,69-108,137-151','vla-scripts/finetune.py:167-169,233-247,295-297','prismatic/models/vlas/openvla.py:33-46,94-111,128-156,188-229','experiments/bridge/reasoning_server.py:65-81','experiments/bridge/eval_model_in_bridge_env.py:84-87,126-175']}
audits['gc_vln']={
 'code': "INSPECTED CODE. src/solver/constraint.py implements the four families of geometric masks. Direction tolerances differ between RxR and R2R (π/4 versus π/6), so the geometric prior is dataset-configured. agent_policy.py:115–143 receives simulator pose/context and the parsed instruction DAG;310–322 calls the navigation-tree constraint process. navigation_tree.py:76–121 topologically orders the instruction DAG and groups object constraints by stage. region_solver.py:72–85 filters detected instances by range,141–170 additionally uses detection count, and199–215 converts semantic relations into geometric constraints. Its failure branches at777–790 and905–916 invoke backtracking, then bounded exploration/final fallback. These are finite heuristics over estimated instances; they do not certify instruction satisfaction.\n\nnavigation_planner.py:124–234 constructs an FMM distance field, follows its gradient and handles incomplete paths, including a geodesic-to-straight-distance threshold of3. No navigation gradient training is introduced. The inspected path covers constraint representation, instance selection and local motion, while exact upstream detector outputs, instruction parsing reproducibility and complete benchmark aggregation were not independently reproduced. The printed directional equation was not treated as a proven error because coordinate/typographic conventions require visual disambiguation. No project code was executed.",
 'code_files':['src/solver/constraint.py','src/agent/agent_policy.py:115-143,310-322','src/solver/navigation_tree.py:76-121','src/solver/region_solver.py:72-85,141-170,199-215,777-790,905-916','src/agent/navigation_planner.py:124-234']}
audits['navgpt']={
 'code': "INSPECTED CODE. nav_src/agent.py:300–325 serializes directional captions, object headings/distances and navigable viewpoint IDs. Lines353–390 construct action/history prompts and call the LLM;397–406 reject nonexistent viewpoint IDs before simulator movement;421–431 summarize the resulting history. create_vln_agent:558–622 constructs action/backtrace tools and a LangChain executor with an iteration cap. This is prompted inference with no navigation-specific optimizer or training loss.\n\nThe released nav_src/env.py:264–290 evaluates final and oracle error, uses the success margin, and computes SPL as success times reference length divided by the larger reference/predicted length. Lines294–321 aggregate the metrics. The prompt file and core loop were inspected; BLIP/object-description caches and exact historical API behavior were not regenerated. Consequently source inspection verifies interfaces and evaluator structure, not the numerical paper results. No API call, MatterSim episode or model execution was performed.",
 'code_files':['nav_src/agent.py:300-440,558-622','nav_src/env.py:264-321'],
 'limitations':"The supplied main paper and targeted prompt/inference/evaluation code were inspected. The complete original caption caches and API backend cannot be reconstructed from this review; additional supplementary examples were not exhaustively reviewed. No benchmark was executed."}
audits['psgnav']={
 'code': "INSPECTED CODE. graph/scenegraph_combinations.py:60–125 sorts per-object categorical choices and enumerates high-probability combinations with a heap; log probabilities are added, implementing the paper's independence product. The search stops after at most10,000 visited configurations in that path. graph/enumerated_groups.py:252–263 invokes the LLM filter, so retained worlds still depend on semantic-prior judgments. PSG_Nav.py:1484–1489 reads Habitat outcome metrics. Its feedback path at1497–1554 inserts successful detections into positive memory and, on a failed episode, puts recent detections into the false-positive database.\n\nThe latter is a potential credit-assignment problem: episode failure can result from control or stopping rather than false object recognition. The code alone does not show how often this mislabels memory, so an oracle-perception versus observed-feedback experiment is needed. calculate_avg_results.py:7–19 parses saved dictionaries and41–52,72–82 averages SR/SPL, replacing NaN numeric values by zero. No task-specific gradient training is introduced, but persistent outcome-labeled updates are adaptation. Complete perception preprocessing, all planner branches and released-model outputs were not reproduced. No project code was executed.",
 'code_files':['graph/scenegraph_combinations.py:60-125','graph/enumerated_groups.py:60-80,125,252-263','PSG_Nav.py:1484-1554','calculate_avg_results.py:7-19,41-52,72-82'],
 'limitations':"The main method, benchmark protocol and targeted world-enumeration/outcome-memory/evaluation code were reviewed. Exact episode order, memory warm-up/reset policy and detector calibration were not independently reconstructed. No ObjectNav benchmark was run."}
audits['rl4vla']={
 'pages':'1,3–8,17–24',
 'code':"INSPECTED CODE. SimplerEnv/simpler_env/policies/openvla/openvla_train.py:49–75 inserts LoRA and exposes the trainable value head;104–109 creates separate AdamW optimizers. Its action evaluation returns scalar values/log probabilities, and273–319 implements clipped PPO, clipped Huber value loss and entropy regularization, followed by shared gradient clipping and both optimizer steps. The total minimized loss is policy_loss + value_loss − entropy_coefficient × entropy; accumulation divides this loss before backward.\n\nopenvla/prismatic/extern/hf/modeling_prismatic.py:627–642 selects the value-head mode. In the a0 path at690–716, the pre-action hidden state gives a B×1 value, action logits have shape B×7×256, and seven token log probabilities are summed to form the joint action log probability. This matches Appendix A.1's product-of-token-probabilities convention. The fetched eval_ms3_collect.py:23–44,62–78,90–128 is an Octo data-collection utility, not an RL4VLA performance evaluator; it is explicitly not treated as validation of the paper's PPO evaluation. Full main training orchestration, all reward/task implementations and the exact final evaluator remain outside this targeted source audit. No project code was executed.",
 'code_files':['SimplerEnv/simpler_env/policies/openvla/openvla_train.py:49-75,104-109,152-163,197-209,273-319','openvla/prismatic/extern/hf/modeling_prismatic.py:627-642,690-716,824-835','SimplerEnv/simpler_env/eval_ms3_collect.py:23-44,62-78,90-128'],
 'method':"Section3 uses OpenVLA: SigLIP and DINOv2 image features feed Llama2-7B, which predicts a seven-dimensional end-effector/gripper action. Each action coordinate is discretized into256 bins between training quantiles. The base setting observes one640×480 image and an instruction. All compared fine-tuning methods use rank32 LoRA; the value head is fully trained.\n\nAppendix A.1, Eqs1–2 defines PPO with scalar ratio r_t=πθ(a_t|s_t)/πold(a_t|s_t), where π(a_t|s_t) is the product of the seven action-token probabilities. The maximized surrogate is E[min(r_t A_t, clip(r_t,1−ε,1+ε)A_t)]. GAE estimates A_t by summing discounted temporal-difference residuals, Σ_l(γλ)^l[r_(t+l)+γV(s_(t+l+1))−V(s_(t+l))]. These are policy-gradient estimators, not a convergence guarantee. The three-layer shared critic reads the first action-token hidden state. GRPO instead uses32 groups of8 trajectories and group-normalized terminal outcomes; TPO/DPO compares whole trajectory preferences using the same sparse reward.\n\nThe base pick-and-place reward gives partial credit for grasping/holding and a larger terminal placement reward. A140-demonstration warm start reduces interaction requirements. Motion-planner demonstrations are filtered when translation is below0.01 and Euler-angle change below0.06 (Appendix A.2), removing about one third of idle actions. This preprocessing is important when comparing SFT and RL. The shared critic uses44.4GB versus81.3GB for a separate critic; the main run takes roughly42 hours on one A100. Appendix B.5 adds four-action chunks with action-dimension clipping and retuned γ/λ, so it is a separate recipe. Appendix B.6 adds articulated-object opening on a Franka robot, extending the original pick-and-place scope.",
 'results':"Figure3 compares PPO, GRPO and DPO under the authors' setup; it supports PPO as a strong baseline, not a universal ordering of optimizers. Appendix Table1 reports task success: in-distribution SFT0.781 versus RL0.938; unseen objects0.453 versus0.714; changed robot pose0.339 versus0.797; mid-episode object repositioning0.286 versus0.745. Because base success differs, relative degradation as well as absolute SR matters. Figure7 reports similar relative visual robustness and stronger semantic/execution transfer.\n\nThe extended tests qualify the conclusion. With action chunks, Table3 shows a regression on unseen multiple receptacles (0.365 SFT versus0.297 RL). On articulated objects, Table4 reports visual-average SR0.646 versus0.591 and rephrased-instruction SR0.703 versus0.620; RL improves execution-average SR0.300→0.398. Thus RL does not uniformly improve every semantic or visual shift. Thirty preliminary real trials in Table2 yield pick-and-place SR0 versus0.27, which is promising but small. These manipulation findings motivate matched tests in VLN; they are not navigation results.",
 'limitations':"Main method, Appendix A and additional results in Appendix B were read. The targeted actor/critic/PPO implementation was inspected at the pinned commit, but the full released task suite and evaluation orchestration were not reproduced. All reported benchmark/robot numbers are authors' experiments. No project code was executed."}

audits['aoplanner']['code']+="\n\nAdditional preprocessing inspection: llm/grounded_sam_Gemini.py:313–316,430–437 runs GroundingDINO and SAM; llm/utils.py:22–69,87–142 converts normalized depth and image-space waypoints into world coordinates using camera intrinsics and poses. llm/prompting/Gemini.py:14 selects gemini-1.5-pro-latest, an unpinned service alias. These dependencies constrain reproducibility even when navigation itself is training-free. Distillation-training provenance was not independently reconstructed."
audits['aoplanner']['code_files']+=['llm/grounded_sam_Gemini.py:313-316,430-437','llm/utils.py:22-69,87-142','llm/prompting/Gemini.py:14,43-47']

audits['auxthink']['code']+="\n\nThe linked training repository was also inspected at markinruc/VILA commit f5853256e9e2e796ee9245c72a17a6f853743d45: llava/data/dataset_impl/llava_cot.py:32–99,130–133 substitutes annotated CoT conversations, while llava/model/language_model/llava_llama.py:122–149 forwards embeddings/labels and exposes a generic optional soft-token loss. These files do not by themselves establish the full Aux-Think-specific objective or original data pipeline. The linked markinruc/VLN_CE commit3a6259b2336b1164ee6f25a43a13f52041cbfdff, habitat_extensions/config/vlnce_task_aux_think_r2r.yaml:1–49, specifies RGB640×480,15° turns,0.25m forward steps, sliding enabled and3m SR/SPL. Its task sensors also register oracle-progress/shortest-path information for the environment; this registration alone does not show that the policy consumes those fields. This is a partial training/evaluation provenance audit, not a claim of exact reproduction."
audits['auxthink']['code']+="\n\nPinned related sources: [CoT dataset transform](https://github.com/markinruc/VILA/blob/f5853256e9e2e796ee9245c72a17a6f853743d45/llava/data/dataset_impl/llava_cot.py), [language-model forward](https://github.com/markinruc/VILA/blob/f5853256e9e2e796ee9245c72a17a6f853743d45/llava/model/language_model/llava_llama.py), [Habitat task configuration](https://github.com/markinruc/VLN_CE/blob/3a6259b2336b1164ee6f25a43a13f52041cbfdff/habitat_extensions/config/vlnce_task_aux_think_r2r.yaml)."
audits['navila']['code']=audits['navila']['code'].replace('The low-level controller and its additional sensing/training must be audited separately through the linked legged-loco repository;','The low-level controller has a separate linked legged-loco release;')
audits['navila']['code']+="\n\nThe high-level scripts/train/sft_8frames.sh:10–41 specifies eight frames, SigLIP, all three VLM components trainable, one epoch and lr1e-4, with a mixture including R2R, RxR, EnvDrop, human videos and general QA. The linked yang-zj1026/legged-loco commit87b0d3d18404e784abc0a62227bc41c940f29ecc was also inspected. go2_low_vision_cfg.py:301–324 explicitly consumes angular velocity, orientation, commands, joint positions/velocities, previous actions and a LiDAR-derived height map. Its reward block205–260 includes termination, joint deviation, base-height, action-smoothness, power, foot-sliding and collision penalties. actor_critic_depth_cnn.py:13–60 separately embeds proprioception and depth, concatenates two128-dimensional latents and predicts joint commands;84–166 adds a Gaussian policy and scalar critic. rsl_rl/algorithms/ppo.py:151–175 implements the clipped surrogate/value/entropy update. This checks the sensing and learning mechanism, not the physical calibration or exact paper checkpoint."
audits['navila']['code']+="\n\nPinned related sources: [robot observations and rewards](https://github.com/yang-zj1026/legged-loco/blob/87b0d3d18404e784abc0a62227bc41c940f29ecc/isaaclab_exts/omni.isaac.leggedloco/omni/isaac/leggedloco/config/go2/go2_low_vision_cfg.py), [actor/critic](https://github.com/yang-zj1026/legged-loco/blob/87b0d3d18404e784abc0a62227bc41c940f29ecc/rsl_rl/rsl_rl/modules/actor_critic_depth_cnn.py), [PPO](https://github.com/yang-zj1026/legged-loco/blob/87b0d3d18404e784abc0a62227bc41c940f29ecc/rsl_rl/rsl_rl/algorithms/ppo.py)."
audits['navila']['code_files']+=['scripts/train/sft_8frames.sh:10-41']
audits['navila']['limitations']="The local image-only PDF was replaced for text review by author arXiv2412.04453v2; later proceedings changes are not assumed reviewed. High-level inference/SFT configuration and linked low-level observations/rewards/PPO were inspected. Physical calibration, all training assets and robot results were not reproduced. No Isaac or robot experiment was executed."

audits['awarevln']['pages']='1–7,12–15; alternate version4–7'
audits['awarevln']['results']=notes['awarevln']['results']+"\n\nVersion-specific caution: the main PDF's supplementary Section B (p.12) says training exclusively on RxR followed by RxR evaluation, while Table5 (p.13) is captioned as evaluation without RxR training. The latter reports39.8 SR/36.0 SPL. These statements conflict, so that transfer number is not treated as a settled protocol. The11-page proceedings alternate's method and Tables1,3,4 were checked and retain the main65.4/55.1 R2R and67.6/56.1 RxR results, including the ablation where removing stopping-error data improves R2R SPL to57.5."
audits['evonav']={
 'code':notes['evonav']['code'],
 'limitations':"Main paper and the separate four-page supplement were read. Supplement A.1 uses100 sampled episodes for R2R-CE and100 random NavRAG episodes. Section C.3 explicitly says no ground-truth trajectory is supplied to or stored by the agent: feedback comprises its own rollout plus terminal SR/SPL. The critique therefore concerns ground-truth-derived outcome feedback and episode ordering, not use of an oracle path at inference. No verified executable official release was inspected and no benchmark was executed.",
 'results':notes['evonav']['results']+"\n\nThe supplement distinguishes online accumulation from a larger offline bank. Outcome-derived feedback and episode order must be matched against frozen-memory baselines. Its correction text is LLM-generated from the agent's own failed experience; it is not an oracle-corrected demonstration path.",
}
audits['skillnav']['limitations']+=" Appendix Table10, visually checked in the rendered PDF p.18 (saved as evidence/03/skillnav_table10.png), reports Region DUET SR36.67 and SPL43.02. Under standard identical-episode SPL, SPL cannot exceed SR; the discrepancy needs author clarification before that row is used. The audit does not identify its cause. A separate four-line toy dispatch example reproduces the last-expert-over-batch mechanism, but no SkillNav checkpoint or simulator was run."

audits['uninavid']['pages']='Alternate arXiv v2:1,3–12 (main method, task results and ablations)'
audits['uninavid']['versions']='The local supplied PDF has no extractable text. Reviewed the authors\' arXiv2412.06224v2 (6 February2025), saved as evidence/03/uninavid_alternate.pdf. Numerical claims refer to that version; byte equivalence with the supplied proceedings PDF is not assumed.'
audits['uninavid']['associated_files_reviewed']=[{'file':'evidence/03/uninavid_alternate.pdf','version':'arXiv2412.06224v2','pages':'1,3–12','reason':'Accessible text replacement for the supplied image-only PDF'}]
audits['uninavid']['results']=notes['uninavid']['results'].replace('Multi-task success does not yet isolate synergy without later ablation tables.','Figure8 compares single-task and multi-task learning and data scale, supporting synergy under the authors\' recipe; it does not by itself equalize every task\'s effective data/compute exposure.')+"\n\nTableX (alternate p.11) reports VLN SR9.61 with only the current frame,39.7 with current/short memory and48.7 with long memory added. Each ablation retrains the model. Removing VQA reduces EQA accuracy47.3→1.19, showing that preserving general language knowledge is a major part of the mixture. EQA TableVI distinguishes autonomous navigation47.3 from ground-truth-trajectory54.4. The memory ablation therefore supports retaining history, but not guaranteed preservation of rare landmarks. Additional external supplemental videos/results were not reviewed."
audits['uninavid']['limitations']="An accessible author arXiv version, including main task results and TableX ablations, was reviewed instead of the local image-only proceedings PDF. Separate external supplemental videos and any proceedings-only changes were not reviewed. The released merge implementation was inspected; no benchmark or physical robot run was executed."

# Clear stale promises while retaining actual access boundaries.
limits={
 'adanav':'The main paper and relevant appendices were reviewed. The advertised official code URL was inaccessible at the cutoff; no implementation or benchmark was executed.',
 'correctnav':'The paper mechanism and data/iteration ablations were reviewed. No official executable implementation was verified in the scoped source search. Data regeneration and benchmark reproduction were not attempted.',
 'ham_vln':'The supplied method and reported subset experiments were reviewed. No verified official implementation was inspected. Memory fidelity, full API traces and a matched large-scale evaluation remain untested.',
 'idealvln':'The supplied main paper was reviewed. The advertised project URL was inaccessible, so no official implementation, exact dataset pipeline or training run was inspected. The causal language is assessed as a modeling claim, not an identification theorem.',
 'navfom':'The paper and official project were reviewed; the advertised GitHub link was a placeholder at inspection time. No implementation or checkpoint was inspected. Full data licenses, scene overlap and hardware throughput were not independently audited.',
 'profocus':'Main method, tables and the corresponding alternate-version result pages were read. No official executable code was verified. Exact sampled episode identities, API budgets and loop limits were not established by the reviewed material. No benchmark was executed.',
 'navgpt2':'Main method, principal results and targeted Q-Former/LLM/policy/loss/configuration code were inspected. Full feature preprocessing, all appendix experiments and exact pretrained artifacts were not independently reproduced. No project code was executed.',
 'navgrpo':'The PDF advertises a different GitHub account from the inspected Songlin-Dong release. This provenance difference and the reward-code discrepancy are recorded; neither proves what private implementation produced the tables. No benchmark was executed.',
 'navid':'Main method/results and targeted video-token/model-loss/train/inference code were inspected. Complete annotation generation and exact evaluation assets were not reconstructed. No independent runtime or robot measurement was made.',
 'polarmem':'Core method, principal retrieval results and pinned graph-memory implementation were inspected. Full index construction, all dataset splits and end-to-end retrieval costs were not reproduced. It supplies no VLN performance evidence.',
 'progressthink':'The paper objective and the released inference interfaces were inspected. Exact prefix-CE reduction, empty-prefix treatment and joint-GRPO training implementation were not established by the thin released wrappers. No benchmark was executed.',
 'streamvln':'The paper\'s metric-version inconsistencies are recorded. Targeted streaming inference and training wiring were inspected, but complete loss masking, data generation and optional voxel pruning were not exhaustively audited. No independent runtime or benchmark was measured.',
 'susa':'The main file and its byte-identical alternate were accounted for. The four-branch model, fine-tuning losses and inherited pretraining dispatch were inspected; complete caption/depth generation and the claimed historical-action pretraining configuration remain unverified. No project code was executed.'
}
for k,v in limits.items():audits.setdefault(k,{})['limitations']=v

# Material versions: each file gets an honest associated-material record.
for ident in ['awarevln','evonav','profocus','progressthink','susa']:
 meta=json.loads((ROOT/'corpus'/ident/'metadata.json').read_text(encoding='utf-8'))
 entries=[]; main_sha=meta['local_files'][0]['sha256']
 for f in meta['local_files'][1:]:
  duplicate=f['sha256']==main_sha
  pages='Exact SHA256 duplicate; no separate reread' if duplicate else {'awarevln':'4–7','evonav':'1–4','profocus':'5–8','progressthink':'4–7'}.get(ident,'')
  entries.append({'file':f['filename'],'sha256':f['sha256'],'pages':pages,'review_status':'Duplicate accounted for' if duplicate else 'Relevant method/results or supplement reviewed'})
 audits.setdefault(ident,{})['associated_files_reviewed']=entries
 audits[ident]['versions']='Associated local versions are recorded below with SHA256 identifiers. Exact duplicate bytes were not reread. Distinct alternatives were checked for the relevant method/results; their complete bibliographies were not separately reread.'

# Normalize the requested five-category verdict vocabulary. Narrow claims with no matched follow-up to uncertain.
uncertain={'idealvln','mapgpt','mapnav','navfom','psgnav','rl4vla','streamvln','susa','uninavid','polarmem','progressthink','profocus'}
for ident,n in notes.items():
 for b in n['bottlenecks']:
  if ident in uncertain:b['status']='Uncertain because evidence is insufficient'
  elif 'Shifted' in b['status']:b['status']='Shifted into another cost or limitation'
  else:b['status']='Partially addressed'
  b['description']=b['description'].strip()
 if isinstance(n.get('pages'),list):n['pages']=', '.join(n['pages'])
 audits.setdefault(ident,{})
 if not audits[ident].get('status'):
  audits[ident]['status']='Paper review and scoped source audit complete; access and reproduction limits documented'
 # No implementation defect is promoted to a methodological bottleneck here.
 if 'code' not in audits[ident] and 'code' not in n:
  audits[ident]['code']='No official executable implementation was verified in the scoped source search. Method and training details above are paper evidence. No code inspection or execution is claimed.'

for p in paths:
 ids=[n['id'] for n in json.loads(p.read_text(encoding='utf-8'))]
 p.write_text(json.dumps([notes[i] for i in ids],ensure_ascii=False,indent=2),encoding='utf-8')
(E/'code_audits.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
print('Closeout edits saved for',len(notes),'papers')
