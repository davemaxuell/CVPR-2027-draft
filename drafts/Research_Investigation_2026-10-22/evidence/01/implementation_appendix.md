# Assignment 01: implementation inspection and reproducibility appendix

Cutoff: 2026-10-22. This appendix indexes the actual static source inspection. No downloaded model code, training job, simulator episode, checkpoint validation or API generation call was executed by assignment01. The coordinator’s separate DUET routing toy check is recorded under experiments/duet_routing_check_results.json and is not a navigation benchmark.

Files were read at the commit identifiers below. Locators specify focused functions/regions, not an assertion that every line, every task branch, every dependency, or every configuration was audited. Full pipeline reproduction remains narrower than source availability. HTTP403 from the shared GitHub API was treated as rate limiting; immutable raw-source retrieval and git references were used instead.

| Paper | Actual phase coverage | Remaining scope or reproducibility limit |
|---|---|---|
| [duet](../../dossiers/duet.md) | Model, pretraining losses, teacher labels, graph routing, final goal return, data inputs, R2R driver/configuration and graph evaluator | No simulator/checkpoint run; other task branches are not assumed byte-identical. Source inspection and coordinator routing toy check are different evidence types. |
| [scalevln](../../dossiers/scalevln.md) | R2R and REVERIE DUET branches, teacher loss, feature/annotation input, pretraining launcher and graph evaluator | Full graph generation and GAN image-repair pipeline not reconstructed; recipe/data differences prevent interpreting a benchmark delta as language reliance. |
| [srdf](../../dossiers/srdf.md) | Full paper including appendix, official repository tree/README, release-status check | Original repository README-only; later specialist checkpoints/implementations do not establish full flywheel reproduction. |
| [gaussianmap](../../dossiers/gaussianmap.md) | Main method, objectives, ablations and official repository tree/README | No model implementation in pinned release; semantic compression and action-probability normalization cannot be resolved from code. |
| [uagm](../../dossiers/uagm.md) | Full method and appendix, exact arXiv PDF checksum, official repository tree/README | No model implementation in pinned release; Fisher regularization, VI details and statistical test unit remain unresolved. |
| [bevbert](../../dossiers/bevbert.md) | BEV projection/pretraining heads, discrete teacher loss/data targets, graph metric functions and configuration | Continuous and discrete branches differ; only the specifically linked source functions were inspected. No depth model was trained. |
| [gridmm](../../dossiers/gridmm.md) | Grid feature construction/aggregation, discrete action loss, feature targets, graph metric functions and launch script | Continuous policy subset inspected; full simulator integration unexecuted. Commented STOP reweighting is not an active method. |
| [kerm](../../dossiers/kerm.md) | Fact database/retrieval scripts, knowledge-fusion model, action loss, pretraining inputs, evaluator and launcher | External features/knowledge data are needed; released facts are not certified observations of the current scene. |
| [ver](../../dossiers/ver.md) | Volumetric detector/head, focal/L1 losses, feature loading, 3D evaluation and configuration | Release inspection did not locate full VLN policy/evaluator. Volumetric perception evaluation does not reproduce navigation claims. |
| [hamt](../../dossiers/hamt.md) | Hierarchical/history model, proxy objectives, imitation/RL agent, instruction/features, R2R evaluator and script | No end-to-end ViT training or checkpoint execution. Supplement ablation removes ITM as well as history. |
| [vlnbert](../../dossiers/vlnbert.md) | Recurrent model variants, actor-critic/imitation action path and task-specific code recorded in dossier | Pretrained OSCAR/PREVALENT variants and search/reranking systems are separate models; no execution. |
| [airbert](../../dossiers/airbert.md) | Multimodal model, BnB preprocessing/objectives, training driver and reranking output conversion | Complete beams and exploration observations are external inputs. Generative recurrent policy variant is not the same inference protocol as the inspected reranker. |
| [prevalent](../../dossiers/prevalent.md) | Top-level release links plus separate PREVALENT_R2R pretraining heads, listener loss, train/eval driver and launcher | Application repository pinned separately; HANNA application and original large pretraining run not executed. |
| [speakerfollower](../../dossiers/speakerfollower.md) | Speaker/follower model and search code, supervised train driver and evaluator | Augmented instructions and image features required; no independent search-cost measurement. |
| [selfmonitoring](../../dossiers/selfmonitoring.md) | Co-grounding/progress model, positive loss, rollout, training driver and evaluator | Printed negative MSE sign differs from released positive MSE. No checkpoint verification. |
| [regretful](../../dossiers/regretful.md) | Regret gate, progress detach, visit marker, rollout/trainer and evaluator | Detached progress is a static code finding; its performance effect was not measured. |
| [tacticalrewind](../../dossiers/tacticalrewind.md) | Frontier/ending queues, backtracking, goal reranking and stopping code | Heuristic score calibration remains unproven; mathematical shift counterexample is not benchmark evidence. |
| [rcm](../../dossiers/rcm.md) | Main method/objectives and primary author homepage repository search | No official executable implementation located in bounded search; no fetched model-code inspection claim. |
| [r2r](../../dossiers/r2r.md) | Dataset/simulator baseline, tokenization/connectivity, train-vs-submission driver and evaluator | Dataset assets not downloaded or simulated; test-submission training intentionally uses validation data and cannot support held-out validation claims. |
| [crossmodalmap](../../dossiers/crossmodalmap.md) | Semantic map/goal model, position and coverage losses, offline train driver and online stopping/evaluation path | Perception and local controller are external/pretrained components. Timeout and synthetic stop_signal handling require protocol parity. |
| [weakmap](../../dossiers/weakmap.md) | Map projection/fusion, localization/waypoint losses, DAgger trainer and STOP-aware metrics | Fine/semantic/fused maps have distinct actual tensor sizes; no LMDB collection or simulator execution. |
| [goat](../../dossiers/goat.md) | Confounder extraction/intervention model, DAgger loss, dictionary/data inputs, metrics and script | Implementation attention/gating does not by itself prove causal identification or eliminate all spurious associations. |
| [gavln](../../dossiers/gavln.md) | Frozen VGGT/SigLIP geometry, BEV scatter/pooling, supervised text labels, action decoding/evaluation and launcher | Official README supplies evaluation entry point; full training-driver/configuration reproduction remains unverified. External checkpoints and backbones required. |
| [hsgm](../../dossiers/hsgm.md) | Subtask state/STOP/rollback agent, mapper/path planner, runner metrics and sensor configuration | No external API or simulator call. First STOP is persistent; final-distance metrics differ from an explicit STOP predicate at timeout. |
| [ivln](../../dossiers/ivln.md) | Continuous Map-CMA model, iterative DAgger and masks, oracle transit environment, iterative evaluator and config | Separate discrete IVLN implementation not inspected; paper findings and inspected continuous implementation are distinguished. |
| [overnav](../../dossiers/overnav.md) | Main method, module supervision, experiment/ablation and primary project/repository search | No official executable implementation located in bounded search; mechanism is paper evidence. |
| [marval](../../dossiers/marval.md) | Full main mechanism, expert-label rules and conference slides; official Marky data link checked | No verified complete executable MARVAL implementation; component/data repositories do not establish model release. |
| [navillm](../../dossiers/navillm.md) | Image/graph embedding, navigation classifier, CE training path, separate generation losses, data and evaluator | Paper uniform generation narrative differs from released classifier branch. No full multitask checkpoint execution. |
| [navrag](../../dossiers/navrag.md) | Hierarchical generator, shortest-path dataset conversion, DUET loss/model, pretraining driver and evaluator | API model/version configured externally; regeneration not deterministic from code alone.10× STOP loss is active and should be controlled. |
| [roomtour3d](../../dossiers/roomtour3d.md) | NaviLLM fork, video-frame candidates, teacher loss, summary generation, data/configuration and train driver | Geometry/data-curation pipeline not rerun; video-action source contains commented DAgger branch, not active sampled video training. |
| [topologicalplanning](../../dossiers/topologicalplanning.md) | Main planning/controller objectives, BCE STOP alternative, evaluation protocol and project release search | No verified public executable model repository located; pre-exploration is excluded from reported SPL. |
| [reverie](../../dossiers/reverie.md) | Dataset/supplement metrics, follower positive CE+MSE and target-ID/navigation evaluator | Object proposals/perception are external; ID match and endpoint visibility are separate evaluator branches. |
| [rxr](../../dossiers/rxr.md) | Official data README/schema, dense pose/text alignment interface and full relevant datasheet | No exact RxR baseline model code inspected; related VALAN README and Pangea toolkit lead do not substitute for that inspection. |

## Pinned source records

### duet

Repository: [https://github.com/cshizhe/VLN-DUET](https://github.com/cshizhe/VLN-DUET); commit `93e8b233164bc079a6db48b8a0a78d123ec8de41`.

- [map_nav_src/models/vilmodel.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/models/vilmodel.py) (forward_navigation_per_step, lines 750–817): dual branches, learned fusion, masks and shared backtrack contribution
- [map_nav_src/r2r/agent.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/agent.py) (lines 239–285, 295–303, 394–402, 452–478): teacher policy, shortest-path action execution, stop cache, train/test stop difference, terminal return
- [map_nav_src/models/graph_utils.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/models/graph_utils.py) (FloydGraph.path lines 76–92): routes recursively include intermediary visited nodes
- [pretrain_src/model/pretrain_cmt.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/pretrain_src/model/pretrain_cmt.py) (forward_mlm, forward_mrc, forward_sap): Task heads implement token NLL, soft-label region KL and action classification; these proxy objectives differ from rollout fine-tuning.
- [pretrain_src/data/dataset.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/pretrain_src/data/dataset.py) (376–466): R2R STOP labels equal index0 at the exact final path node; otherwise labels target the next reference node, with unavailable actions ignored; constructs truncated instruction and graph/panorama inputs.
- [map_nav_src/scripts/run_r2r.sh](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/scripts/run_r2r.sh) (19–67): Released script fixes DAgger, dynamic fusion, SPL expert, 15 action steps, 200 instruction tokens, AdamW and behavioral-cloning weight.2; checkpoints/data remain external.
- [map_nav_src/r2r/eval_utils.py](https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/eval_utils.py) (6–25): DTW uses graph shortest distances; nDTW=exp(−DTW/(3×reference_length)); SDTW multiplies endpoint success.

### scalevln

Repository: [https://github.com/wz0919/ScaleVLN](https://github.com/wz0919/ScaleVLN); commit `3bdff0da2216066cade2e7923c4ad0b5c36d6592`.

- [VLN-DUET/map_nav_src/r2r/agent.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/map_nav_src/r2r/agent.py) (407–424;469–503): Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [VLN-DUET/pretrain_src/data/dataset.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/pretrain_src/data/dataset.py) (382–472): Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [VLN-DUET/map_nav_src/r2r/eval_utils.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/map_nav_src/r2r/eval_utils.py) (6–25): Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [VLN-DUET/pretrain_src/run_r2r_h14.sh](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET/pretrain_src/run_r2r_h14.sh) (2–14): Two-GPU launcher selects CLIP-H14 model and HM3D+MP3D+Gibson pretraining configuration.
- [VLN-DUET-RVR/map_nav_src/reverie/agent_obj.py](https://github.com/wz0919/ScaleVLN/blob/3bdff0da2216066cade2e7923c4ad0b5c36d6592/VLN-DUET-RVR/map_nav_src/reverie/agent_obj.py) (rollout and training loss): REVERIE-specific navigation/object grounding branch was inspected separately from R2R.

### srdf

Repository: [https://github.com/wz0919/VLN-SRDF](https://github.com/wz0919/VLN-SRDF); commit `1835eca4bd90958b9b42c210236006e40ed6aa32`.

No model source inspected; only explicitly described paper, project or release metadata.


### gaussianmap

Repository: [https://github.com/Gaozzzz/3D-Gaussian-Map-VLN](https://github.com/Gaozzzz/3D-Gaussian-Map-VLN); commit `3a42711d7449280d4eda89d257e5891947a1ced2`.

No model source inspected; only explicitly described paper, project or release metadata.


### uagm

Repository: [https://github.com/Gaozzzz/Uncertainty-Aware-VLN](https://github.com/Gaozzzz/Uncertainty-Aware-VLN); commit `43748013cc34376b68b159cca6226535ed84ccd0`.

No model source inspected; only explicitly described paper, project or release metadata.


### bevbert

Repository: [https://github.com/MarSaKi/VLN-BEVBert](https://github.com/MarSaKi/VLN-BEVBert); commit `ee40002e9eb75ca6d587e8746990e91b0463f5ca`.

- [bevbert_ce/pretrain/pretrain_src/model/bev_utils.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/bevbert_ce/pretrain/pretrain_src/model/bev_utils.py) (pixel_to_world_mapping200;PointCloud.forward349;project_bev382): depth/pose projection and BEV pooling
- [bevbert_ce/pretrain/pretrain_src/model/vilmodel.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/bevbert_ce/pretrain/pretrain_src/model/vilmodel.py) (LocalBEVEncoder581–609;GlocalTextPathCMT712–730): cell embedding and CE grid setting
- [bevbert_ce/pretrain/pretrain_src/model/pretrain_cmt.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/bevbert_ce/pretrain/pretrain_src/model/pretrain_cmt.py) (347–350): global/local/fused action cross-entropy losses
- [map_nav_src/r2r/agent.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/map_nav_src/r2r/agent.py) (539–555;585–633): Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/pretrain_src/data/dataset.py) (397–440;471–578): Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [map_nav_src/r2r/eval_utils.py](https://github.com/MarSaKi/VLN-BEVBert/blob/ee40002e9eb75ca6d587e8746990e91b0463f5ca/map_nav_src/r2r/eval_utils.py) (6–25): Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.

### gridmm

Repository: [https://github.com/MrZihan/GridMM](https://github.com/MrZihan/GridMM); commit `713139f0be045f339bd041a7caefe10b5c4fc6eb`.

- [VLN_CE/vlnce_baselines/models/Policy_ViewSelection_GridMap.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/VLN_CE/vlnce_baselines/models/Policy_ViewSelection_GridMap.py) (get_rel_position1263;get_gridmap_pos_fts1321;getGlobalMap1377): depth-driven map construction
- [VLN_CE/vlnce_baselines/models/gridmap/vilmodel.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/VLN_CE/vlnce_baselines/models/gridmap/vilmodel.py) (forward_navigation_per_step1419–1595): map-conditioned action path and local/global masked logits
- [map_nav_src/r2r/agent.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/map_nav_src/r2r/agent.py) (330–370;401–448): Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/pretrain_src/data/dataset.py) (636–751): Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [map_nav_src/r2r/eval_utils.py](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/map_nav_src/r2r/eval_utils.py) (6–25): Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [map_nav_src/scripts/run_r2r.sh](https://github.com/MrZihan/GridMM/blob/713139f0be045f339bd041a7caefe10b5c4fc6eb/map_nav_src/scripts/run_r2r.sh) (19–67): Default script uses four GPUs, batch 4 and15 action steps; test invocation and training command/checkpoint selection must be distinguished.

### kerm

Repository: [https://github.com/XiangyangLi20/KERM](https://github.com/XiangyangLi20/KERM); commit `b51504334bebb2805b4608b7774bac076d85432a`.

- [knowledge_nav_src/models/vilmodel.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/models/vilmodel.py) (forward_panorama_per_step1465–1467;forward_navigation_per_step1747;masking1823–1833): knowledge-conditioned DUET pipeline
- [preprocess/retrieve_facts.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/preprocess/retrieve_facts.py) (FeaturesDB22–28): HDF5 feature loading for retrieval
- [preprocess/get_knowledge_base.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/preprocess/get_knowledge_base.py) (VisualGenomeCaptions15–21): Visual Genome annotation parsing
- [knowledge_nav_src/r2r/agent.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/r2r/agent.py) (426–442;515): Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/pretrain_src/data/dataset.py) (419–512): Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [knowledge_nav_src/r2r/eval_utils.py](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/r2r/eval_utils.py) (6–25): Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [knowledge_nav_src/scripts/run_r2r.sh](https://github.com/XiangyangLi20/KERM/blob/b51504334bebb2805b4608b7774bac076d85432a/knowledge_nav_src/scripts/run_r2r.sh) (19–68): Pinned script selects dynamic fusion and external trained checkpoint; training command is commented out in this file.

### ver

Repository: [https://github.com/DefaultRui/VLN-VER](https://github.com/DefaultRui/VLN-VER); commit `d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a`.

- [projects/mmdet3d_plugin/bevformer/detectors/voxelformer.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/mmdet3d_plugin/bevformer/detectors/voxelformer.py) (231–326): Loads six horizontal-view precomputed ViT patch feature tensors, ground-truth geometry/occupancy and invokes volumetric losses; HDF5 drops CLS to retain196×768 patch features.
- [projects/mmdet3d_plugin/bevformer/dense_heads/voxelformer_occupancy_head.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/mmdet3d_plugin/bevformer/dense_heads/voxelformer_occupancy_head.py) (282–350;903–991;1505–1540): Learned voxel queries and positional embeddings; matched detection classification/L1 plus occupancy loss; focal-loss inference adds a0.25 empty-space threshold before argmax.
- [projects/configs/verformer/vocc.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/configs/verformer/vocc.py) (9–20;55–207;209 onward): 15×15×4 initial volume, 768 channels, 17 object and16 occupancy classes; focal-loss and L1 weights; external feature/annotation paths remain placeholders.
- [projects/mmdet3d_plugin/datasets/mp3docc_dataset.py](https://github.com/DefaultRui/VLN-VER/blob/d925dfa3c815779c3b6b131ccc3ccf9df1fbfc3a/projects/mmdet3d_plugin/datasets/mp3docc_dataset.py) (304–345;485–579): Indoor box evaluation and occupancy IoU using sparse labels/optional invalid-voxel masks; these are perception metrics, not VLN SR/SPL.

### hamt

Repository: [https://github.com/cshizhe/VLN-HAMT](https://github.com/cshizhe/VLN-HAMT); commit `c8b9ee12125f9fe36c51d2ab928fde38f7d846bd`.

- [finetune_src/models/vilmodel_cmt.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/models/vilmodel_cmt.py) (HistoryEmbeddings lines 523–559; NavCMT lines 624–629): History encodes temporal and panoramic structure and is passed alongside current visual observations into cross-modal navigation.
- [finetune_src/r2r/agent_cmt.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/r2r/agent_cmt.py) (training paths lines 549 and584–602): Imitation and reinforcement updates are explicit; gradients are clipped at40 before optimizer steps.
- [pretrain_src/model/pretrain_cmt.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/pretrain_src/model/pretrain_cmt.py) (task-dispatch and prediction-head definitions): Separate proxy heads implement the paper’s multi-task pretraining; this is training-time supervision, not an inference-time planner.
- [finetune_src/r2r/env.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/r2r/env.py) (evaluation methods): Checked final and oracle graph-distance evaluation, supporting distinction between endpoint SR and route fidelity.
- [finetune_src/r2r/data_utils.py](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/r2r/data_utils.py) (feature and instruction loaders): Inspected precomputed feature/encoded-instruction loading, not an executed preprocessing pipeline.
- [finetune_src/scripts/run_r2r.sh](https://github.com/cshizhe/VLN-HAMT/blob/c8b9ee12125f9fe36c51d2ab928fde38f7d846bd/finetune_src/scripts/run_r2r.sh) (released training/evaluation flags): Configuration snapshot inspected; scripts select task data, model dimensions and checkpoint paths.

### vlnbert

Repository: [https://github.com/YicongHong/Recurrent-VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT); commit `ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e`.

- [r2r_src/vlnbert/vlnbert_PREVALENT.py](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e/r2r_src/vlnbert/vlnbert_PREVALENT.py) (forward lines 322–327 and mode dispatch near384): The recurrent state is concatenated with current visual features for the visual/cross-modal branch.
- [r2r_src/agent.py](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e/r2r_src/agent.py) (lines 527–544): Imitation and RL losses are optimized with gradient clipping40.
- [r2r_src/eval.py](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/ad147f464ae81ef0a9eeb29dc0a0cbbcddcb1f3e/r2r_src/eval.py) (Evaluation scoring functions): Scoring computes endpoint and path measures; no instruction counterfactual evaluation is implemented in the inspected evaluator.

### airbert

Repository: [https://github.com/airbert-vln/airbert](https://github.com/airbert-vln/airbert); commit `34f533dba3f2a78cc111e40e5ffd46315a349f1b`.

- [airbert.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/airbert.py) (forward nearline62): The model consumes text tokens and image-region features for multimodal scoring.
- [utils/dataset/bnb_dataset.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/utils/dataset/bnb_dataset.py) (load_shuffler284–286; caption construction288 onward; negatives536 and587–593): Shufflers produce negative captions/images and return them with positives, confirming the supervision mechanism.
- [train_bnb.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/train_bnb.py) (training entrypoint and loss aggregation): Pretraining is an optimization stage; inference is not training-free use of an unchanged general VLM.
- [test.py](https://github.com/airbert-vln/airbert/blob/34f533dba3f2a78cc111e40e5ffd46315a349f1b/test.py) (125–192): Reranks externally supplied beams; test conversion explicitly prepends exploration_path. This confirms long exploration belongs in comparison cost.

### prevalent

Repository: [https://github.com/weituo12321/PREVALENT](https://github.com/weituo12321/PREVALENT); commit `4ccf59be11a9f7fb6b28f43631fd147edbaecf37`.

- [r2r_src/r2rpretrain_class.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/r2rpretrain_class.py) (104–145): Application repository has MLM/action heads with positive CE sum; this is inspected source, not a full reproduction of original pretraining.
- [r2r_src/agent.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/agent.py) (167–174;353–503): Listener training combines teacher CE and actor-critic policy/value/entropy terms.
- [r2r_src/eval.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/eval.py) (51–107): Final/nearest goal distances and graph path length yield SR/OSR/SPL.
- [r2r_src/train.py](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/r2r_src/train.py) (181–250;444–479): Training loop, validation-SPL checkpoint selection and split/vocabulary construction.
- [run/agent.bash](https://github.com/weituo12321/PREVALENT_R2R/blob/868fb53d6b7978bbb10439a59e65044c811ee5c2/run/agent.bash) (1–9): Sample-feedback listener configuration, RMS optimizer and80k iterations; exact pretrained model branch must be checked before reproduction.

### speakerfollower

Repository: [https://github.com/ronghanghu/speaker_follower](https://github.com/ronghanghu/speaker_follower); commit `e4dcdf923ecacf7893397bd9821377e71e32a75a`.

- [tasks/R2R/speaker.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/speaker.py) (lines 179–182): Speaker score sums token log likelihoods and trains with token NLL.
- [tasks/R2R/follower.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/follower.py) (rollout lines 430–538; state-factored search): Follower trains on action CE and maintains recurrent state during rollout/search.
- [tasks/R2R/rational_follower.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/rational_follower.py) (lines 85–106, 131–150): Physical path reconstruction is distinct from weighted speaker/follower candidate reranking; tested speaker weights include0.95.
- [tasks/R2R/data_augmentation_from_speaker.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/data_augmentation_from_speaker.py) (entry_point lines 35–83): Speaker-generated and scored candidates form augmentation output.
- [tasks/R2R/eval.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/eval.py) (56–138): Scores endpoint/oracle distances and full trajectory length; candidate reranking and search cost should not be collapsed into greedy performance.
- [tasks/R2R/train.py](https://github.com/ronghanghu/speaker_follower/blob/e4dcdf923ecacf7893397bd9821377e71e32a75a/tasks/R2R/train.py) (68–138;174–233): Training driver configures data augmentation, training split, vocabulary and validation separately.

### selfmonitoring

Repository: [https://github.com/chihyaoma/selfmonitoring-agent](https://github.com/chihyaoma/selfmonitoring-agent); commit `28174ef695fb2483130cb00708ac306e465613ef`.

- [tasks/R2R-pano/agents/pano_agent.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/agents/pano_agent.py) (lines 156–181): Target is normalized start-to-goal distance improvement, with≤3m override and positive MSELoss.
- [tasks/R2R-pano/agents/pano_agent.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/agents/pano_agent.py) (lines 311–329 and602–634): Positive weighted CE/MSE training; inference compares progress scores and records exploration/rollback.
- [tasks/R2R-pano/models/policy_model.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/models/policy_model.py) (co-grounding forward definitions lines 57 onward): Visual candidate and language attentions feed the recurrent action policy.
- [tasks/R2R-pano/trainer.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/trainer.py) (28–70): Loads training ground-truth paths, selects the model-specific rollout, and backpropagates the returned loss.
- [tasks/R2R-pano/eval.py](https://github.com/chihyaoma/selfmonitoring-agent/blob/28174ef695fb2483130cb00708ac306e465613ef/tasks/R2R-pano/eval.py) (43–95): Final endpoint and nearest visited viewpoint determine success/oracle success; explicit path distances are scored.

### regretful

Repository: [https://github.com/chihyaoma/regretful-agent](https://github.com/chihyaoma/regretful-agent); commit `5caf7b500667981bc7064e4d31b49e83db64c95a`.

- [tasks/R2R-pano/models/policy_model.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/models/policy_model.py) (lines 126–157): Detached progress difference drives2-way rollback attention; current-minus-stored progress features affect action logits.
- [tasks/R2R-pano/agents/pano_agent.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/agents/pano_agent.py) (lines 381–437): Per-episode dictionary tracks node progress; positive MSE/CE plus negative-entropy regularization trains the agent.
- [tasks/R2R-pano/trainer.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/trainer.py) (27–70): Loads training ground-truth paths, selects the model-specific rollout, and backpropagates the returned loss.
- [tasks/R2R-pano/eval.py](https://github.com/chihyaoma/regretful-agent/blob/5caf7b500667981bc7064e4d31b49e83db64c95a/tasks/R2R-pano/eval.py) (43–95): Final endpoint and nearest visited viewpoint determine success/oracle success; explicit path distances are scored.

### tacticalrewind

Repository: [https://github.com/Kelym/FAST](https://github.com/Kelym/FAST); commit `c4f1dabb388a16d96a388d4d6c3a309047bc41e3`.

- [follower.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/follower.py) (lines 844–917): Tracks logit/log-prob summaries; selectively expands topK actions; routes STOP candidates to ending_queue separately from frontier nodes.
- [follower.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/follower.py) (lines 966–986): Goal reranker uses length/logit/log-prob/progress/speaker features, then realistic_jumping returns through the visited graph.
- [follower.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/follower.py) (lines 542–551 and658–666): Progress MSE and action CE training remain separate from search heuristics.
- [run_search.py](https://github.com/Kelym/FAST/blob/c4f1dabb388a16d96a388d4d6c3a309047bc41e3/run_search.py) (search setup and sweep entrypoints): Search settings are explicit experiment choices and must be matched in comparisons.

### rcm

No verified executable repository in the bounded search record.

No model source inspected; only explicitly described paper, project or release metadata.


### r2r

Repository: [https://github.com/peteanderson80/Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator); commit `589d091b111333f9e9f9d6cfd021b2eb68435925`.

- [tasks/R2R/agent.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/agent.py) (Seq2SeqAgent action definitions and rollout): Low-level turn/look/forward/end actions are distinct from later panoramic neighbor actions.
- [tasks/R2R/model.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/model.py) (EncoderLSTM and attention decoder): Sequence encoder and image-conditioned recurrent decoder implement the original baseline.
- [tasks/R2R/eval.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/eval.py) (Evaluation._score_item and score): Uses shortest-path graph distance for endpoint and oracle proximity.
- [tasks/R2R/utils.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/utils.py) (19–120): Loads connectivity with Euclidean edge weights; tokenizes/lowercases, reverses words and constructs training vocabulary.
- [tasks/R2R/train.py](https://github.com/peteanderson80/Matterport3DSimulator/blob/589d091b111333f9e9f9d6cfd021b2eb68435925/tasks/R2R/train.py) (45–101;105–151): Distinct train-validation and test-submission routines; the latter intentionally combines train/validation data, so it must not be reused to report held-out validation results.

### crossmodalmap

Repository: [https://github.com/ggeorgak11/CM2](https://github.com/ggeorgak11/CM2); commit `4730cfdf49c468d6632fff03ecc9cee9955e7dee`.

- [models/vln_goal_predictor.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/models/vln_goal_predictor.py) (coverage_loss89–102; position_loss106–126): Two-class coverage CE and visibility-masked heatmap MSE.
- [models/semantic_grid.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/models/semantic_grid.py) (spatial-transform and map update methods): Map construction uses pose-aligned spatial transformations.
- [datasets/util/map_utils.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/datasets/util/map_utils.py) (projection and heatmap helpers): Training labels and observations use explicit metric coordinates.
- [trainer_vln.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/trainer_vln.py) (training loss aggregation): Waypoint and map objectives are optimized together.
- [trainer_vln.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/trainer_vln.py) (24–75): Offline dataset, Adam optimization, and summed waypoint-position plus coverage losses.
- [tester.py](https://github.com/ggeorgak11/CM2/blob/4730cfdf49c468d6632fff03ecc9cee9955e7dee/tester.py) (260–298): Stops by predicted goal distance or time limit; local-policy STOP is replaced by random motion, and final metric helper is called with stop_signal=True. This protocol must be replicated before comparisons.

### weakmap

Repository: [https://github.com/PeihaoChen/WS-MGMap](https://github.com/PeihaoChen/WS-MGMap); commit `e6af3c40c19010ecf07ea1d62c18c8196e37150d`.

- [vlnce_baselines/models/mg_map_policy.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/models/mg_map_policy.py) (forward180–251): Projects RGB features, predicts semantic map, concatenates map features, computes text-conditioned map attention, and updates the second recurrent state.
- [vlnce_baselines/common/rgb_mapping.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/common/rgb_mapping.py) (mapping update methods): Builds pose-aligned projected latent-feature memory.
- [vlnce_baselines/dagger_trainer.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/dagger_trainer.py) (collection/training paths): DAgger is a substantive training stage rather than an inference-only map addition.
- [vlnce_baselines/config/CMA_AUG_DA_TUNE.yaml](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/config/CMA_AUG_DA_TUNE.yaml) (training configuration): Defines the augmented/DAgger setup needed for matched reproduction.
- [vlnce_baselines/dagger_trainer.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/vlnce_baselines/dagger_trainer.py) (505–536): Applies tanh to2D waypoint predictions and weighted MSE to target waypoints, then adds auxiliary map/localization losses and backpropagates.
- [habitat_extensions/measures.py](https://github.com/PeihaoChen/WS-MGMap/blob/e6af3c40c19010ecf07ea1d62c18c8196e37150d/habitat_extensions/measures.py) (320–323): Success requires task.is_stop_called and target distance below threshold.

### goat

Repository: [https://github.com/CrystalSixone/VLN-GOAT](https://github.com/CrystalSixone/VLN-GOAT); commit `c083bf458e6d2041fbdf91c2a8036f458faa95b4`.

- [map_nav_src/models/vilmodel_GOAT.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/models/vilmodel_GOAT.py) (lines 94–162): Frequency-weighted or attention-based dictionary features are added/gated with original text embeddings, including direct residual paths.
- [map_nav_src/do_utils/do_intervention.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/do_utils/do_intervention.py) (dictionary construction/extraction routines): Confounder representations are estimated from training samples, not externally randomized assignments.
- [map_nav_src/do_utils/extract_room_type.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/do_utils/extract_room_type.py) (BLIP room-type extraction): Room labels are model predictions and can carry perception errors.
- [map_nav_src/r2r/agent.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/r2r/agent.py) (306–350;616–621;697): Inspected action-label/rollout loss path: categorical teacher action supervision with policy/teacher feedback; comparison requires matching expert and STOP handling.
- [pretrain_src/data/dataset.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/pretrain_src/data/dataset.py) (68–132;616–721): Inspected annotation/feature and action-target preparation; geometric/map/confounder inputs are additional data requirements where used.
- [map_nav_src/r2r/eval_utils.py](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/r2r/eval_utils.py) (6–25): Graph-distance DTW, exponential normalization and success-weighted DTW use the inherited discrete evaluator.
- [map_nav_src/scripts/run_r2r_goat.sh](https://github.com/CrystalSixone/VLN-GOAT/blob/c083bf458e6d2041fbdf91c2a8036f458faa95b4/map_nav_src/scripts/run_r2r_goat.sh) (15–72): Activates front/back-door branches, updated text dictionary, generated speaker labels and augmented data; these additions must be matched in causal-module comparisons.

### gavln

Repository: [https://github.com/jahhaoyang/GA-VLN](https://github.com/jahhaoyang/GA-VLN); commit `cc6086b7081a346695abecf6821f827e2db44a43`.

- [gavln/model/model_gavln.py](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/gavln/model/model_gavln.py) (encode_images127–261): VGGT aggregation runs under no_grad; SigLIP/VGGT features share grid accumulators and are averaged only in occupied cells before MLLM projection.
- [gavln/dataset/vln_action_dataset.py](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/gavln/dataset/vln_action_dataset.py) (83–154;237–318): Conversation preprocessing masks prompt tokens with IGNORE_INDEX and supplies assistant/action text labels for autoregressive supervision.
- [gavln/gavln_eval.py](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/gavln/gavln_eval.py) (178–215;412–454;456 onward): Greedy model generation is parsed into actions; Habitat metrics are recorded after rollout. Training driver was not verified from the official entry-point documentation.
- [scripts/gavln_eval.sh](https://github.com/jahhaoyang/GA-VLN/blob/cc6086b7081a346695abecf6821f827e2db44a43/scripts/gavln_eval.sh) (6–25): Evaluation requires external checkpoints, SigLIP and VGGT paths plus distributed-launch settings.

### hsgm

Repository: [https://github.com/Teacher-Tom/HSGM_public](https://github.com/Teacher-Tom/HSGM_public); commit `e2df6e70df869af57d77a32ba658fba04e6814eb`.

- [src/agent/agent.py](https://github.com/Teacher-Tom/HSGM_public/blob/e2df6e70df869af57d77a32ba658fba04e6814eb/src/agent/agent.py) (804–979;1082–1094;1646–1884): Instruction subtask state, persistent first/second STOP flags, no-movement guard, replanning and timeout rollback.
- [src/run_experiments.py](https://github.com/Teacher-Tom/HSGM_public/blob/e2df6e70df869af57d77a32ba658fba04e6814eb/src/run_experiments.py) (445–529;979–1120): Final-distance success/SPL and Euclidean fast-DTW; stopping/saving occurs on second_stop or maximum steps.
- [config/vlnce_test.yaml](https://github.com/Teacher-Tom/HSGM_public/blob/e2df6e70df869af57d77a32ba658fba04e6814eb/config/vlnce_test.yaml) (camera and simulator configuration): Sensor FOV135 degrees, pitch−30 degrees, height 1.2m, maximum 200 steps and radius 0.1m are part of the released operating conditions.

### ivln

Repository: [https://github.com/jacobkrantz/IVLN-CE](https://github.com/jacobkrantz/IVLN-CE); commit `14417fc0930fc273183667c09e3e3564adc823a0`.

- [ivlnce_baselines/models/map_cma_policy.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/models/map_cma_policy.py) (276–368): Map/depth/text attention and episodic recurrent-state masks; optional tanh progress monitor with positive MSE.
- [ivlnce_baselines/trainers/iterative_dagger_trainer.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/trainers/iterative_dagger_trainer.py) (33–94): Tour memory versus episodic resets, truncated gradient state, weighted action CE and auxiliary losses.
- [ivlnce_baselines/common/environments.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/common/environments.py) (36–110): Iterative environment includes agent, oracle_goal and oracle_start phases, explicitly using oracle transit between episodes.
- [ivlnce_baselines/common/base_il_trainer.py](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/common/base_il_trainer.py) (173–218;585–914): Weighted imitation update and iterative evaluation maintaining tour/episode masks and aggregating tour-nDTW.
- [ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/1_ftune_dagger.yaml](https://github.com/jacobkrantz/IVLN-CE/blob/14417fc0930fc273183667c09e3e3564adc823a0/ivlnce_baselines/config/map_cma/pred_semantics/iterative_maps/1_ftune_dagger.yaml) (1–38): Predicted-semantics iterative mapper, 10 DAgger iterations, 5000 collected trajectories and progress auxiliary enabled.

### overnav

No verified executable repository in the bounded search record.

No model source inspected; only explicitly described paper, project or release metadata.


### marval

No verified executable repository in the bounded search record.

No model source inspected; only explicitly described paper, project or release metadata.


### navillm

Repository: [https://github.com/LaVi-Lab/NaviLLM](https://github.com/LaVi-Lab/NaviLLM); commit `e069f46ed98affb221d58715a785613622e11145`.

- [models/nav_model.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/models/nav_model.py) (forward_navigation129–247): Builds global graph embeddings, masks visited nodes, merges local views, permutes candidates, applies out_head to LLM CLS states and returns fused logits.
- [models/nav_model.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/models/nav_model.py) (summarization251–344;object grounding407 onward): Other tasks route through language-model losses/heads; training objectives depend on task mode.
- [models/image_embedding.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/models/image_embedding.py) (multi-view embedding classes): Encodes visual and geometric inputs before the language model.
- [tasks/agents/mp3d_agent.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/tasks/agents/mp3d_agent.py) (rollout/teacher-action paths): Graph navigation and supervision sit outside the LLM module.
- [tasks/agents/mp3d_agent.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/tasks/agents/mp3d_agent.py) (734–756;845–904): Navigation criterion over candidate logits is backpropagated separately from QA/summarization token-generation loss.
- [train.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/train.py) (207–231): Constructs summed categorical cross entropy with ignored padding targets; driver uses task-specific agents.
- [tasks/datasets/r2r.py](https://github.com/LaVi-Lab/NaviLLM/blob/e069f46ed98affb221d58715a785613622e11145/tasks/datasets/r2r.py) (10–60;108–129): Expands instructions into samples, truncates encodings and prepares fine-grained labels; evaluation flattens graph subpaths and uses final/nearest geodesic errors, path length and SPL.

### navrag

Repository: [https://github.com/MrZihan/NavRAG](https://github.com/MrZihan/NavRAG); commit `7dd133e88797a6c4baacf820f3edf905740077fa`.

- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py) (teacher/rollout/navigation routines): Downstream policy is a DUET-family graph navigator.
- [VLN-DUET-NAVRAG/pretrain_src/train_rag.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/pretrain_src/train_rag.py) (training loop and data configuration): Dataset pretraining is a distinct optimization stage.
- [instruction_generator/convert_to_dataset.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/instruction_generator/convert_to_dataset.py) (instruction/path conversion): Generated annotations are converted into navigation training records.
- [VLN-DUET-NAVRAG/map_nav_src/scripts/rag_h14_envedit_mix.sh](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/scripts/rag_h14_envedit_mix.sh) (experiment configuration): Feature/data mixture choices are explicit and must be controlled.
- [instruction_generator/generate_instruction.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/instruction_generator/generate_instruction.py) (22–70;74–150): API model is configured externally; hierarchical role→zone→viewpoint→direction→instance prompts construct demand instructions. Responses are parsed without independent semantic verification.
- [instruction_generator/convert_to_dataset.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/instruction_generator/convert_to_dataset.py) (67–148): Uses seed 0, held-out building lists and shortest paths; samples starts with replacement from the nearest third of graph distances, 10 per training instruction versus 1 for held-out instructions.
- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent.py) (394–425): Shared softmax STOP score is cached; active training multiplies STOP-target loss by 10.
- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent_base.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/agent_base.py) (141): CrossEntropyLoss uses reduction=none, confirming per-example STOP weighting is active.
- [VLN-DUET-NAVRAG/map_nav_src/NavRAG/env.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/map_nav_src/NavRAG/env.py) (314–335): Scores flattened graph paths with final/nearest target distances and SPL.
- [VLN-DUET-NAVRAG/pretrain_src/train_rag.py](https://github.com/MrZihan/NavRAG/blob/7dd133e88797a6c4baacf820f3edf905740077fa/VLN-DUET-NAVRAG/pretrain_src/train_rag.py) (180–260;349–428): Multi-task loader and optimizer; masked-token CE, region KL and action CE validation.

### roomtour3d

Repository: [https://github.com/roomtour3d/roomtour3d-NaviLLM](https://github.com/roomtour3d/roomtour3d-NaviLLM); commit `eac6c0753621e92ebc7ac8e92c3e25f14285e31c`.

- [tasks/agents/tour3d_agent.py](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/tasks/agents/tour3d_agent.py) (579–604;828–865;907–944): Teacher-forced video navigation criterion and separate summary generation loss; sampled DAgger path is commented out in this video agent.
- [tasks/datasets/tour3d_dataset.py](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/tasks/datasets/tour3d_dataset.py) (121–189;370–462): Filters short trajectories, expands instruction records, and constructs video-frame navigation candidates with angle/3D-position metadata.
- [configs/pretrain.yaml](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/configs/pretrain.yaml) (Feature and Dataset sections): Explicit external EVA feature files and synthetic/MP3D task data paths.
- [configs/multi.yaml](https://github.com/roomtour3d/roomtour3d-NaviLLM/blob/eac6c0753621e92ebc7ac8e92c3e25f14285e31c/configs/multi.yaml) (Feature and Dataset sections): Fine-tuning data mix includes video features; one path has a.bak suffix, requiring local data-path validation.

### topologicalplanning

No verified executable repository in the bounded search record.

No model source inspected; only explicitly described paper, project or release metadata.


### reverie

Repository: [https://github.com/YuankaiQi/REVERIE](https://github.com/YuankaiQi/REVERIE); commit `207e90cdde48acf2f2f3fc912528966a062fc432`.

- [tasks/REVERIE/follower.py](https://github.com/YuankaiQi/REVERIE/blob/207e90cdde48acf2f2f3fc912528966a062fc432/tasks/REVERIE/follower.py) (304–305;685–712;798): Positive MSE and action CE, including an explicit equally weighted combined loss.
- [tasks/REVERIE/eval_release.py](https://github.com/YuankaiQi/REVERIE/blob/207e90cdde48acf2f2f3fc912528966a062fc432/tasks/REVERIE/eval_release.py) (56–109): Grounding target ID match; endpoint/anywhere target visibility for navigation and oracle success; graph-distance trajectory length.

### rxr

Repository: [https://github.com/google-research-datasets/RxR](https://github.com/google-research-datasets/RxR); commit `d22df5940a0486a42b703cd445937210ce21e9f5`.

- [README.md](https://github.com/google-research-datasets/RxR/blob/d22df5940a0486a42b703cd445937210ce21e9f5/README.md) (Guide/Follower schemas; Pose Traces; Text Features; Changelog): Data release documents dense alignments, missing word timestamps in a small subset, and seven corrected training pose-trace archives. This is data-interface inspection, not baseline model execution.

