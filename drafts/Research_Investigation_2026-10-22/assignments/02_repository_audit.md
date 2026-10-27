# Assignment 02: implementation and access audit

Cutoff: 2026-10-22. This appendix links exact source snapshots. The narrative identifies substantive inspected lines; the extraction log also includes broader search excerpts and is not a claim that every line of every fetched repository was read. No fetched project code, checkpoint, simulator or training command was executed. Independent arithmetic checks are recorded separately in `logic_check_results.json`.

## sc2wm

INSPECTED CODE: `vlnce_baselines/models/etp/vilmodel_cmt.py:598–621,934–967` implements a 2304→1536→768 MLP over navigation, mixed-candidate and raw-candidate features. The immediate refinement uses detached reconstructed features and a learned residual/gate. This is not a literal implementation of the paper's Gaussian posterior/prior RSSM equations. Crucially, `vlnce_baselines/ss_trainer_ETP.py:2246–2308` ALSO compares a previous predicted candidate feature with the next real panoramic observation using one minus cosine similarity. This delayed observed residual drives test-time adaptation; a claim that SC²-WM never corrects using observed outcomes is false. It also has drift and percentile-KL controls. The active branch does not show the paper's entropy gate at this location. `TTA_setup_trainable_params` at 1030–1035 enables all global-encoder parameters, including action/cross-modal components, whereas its nearby docstring describes a narrower component. Lines 2325–2371 recompute features and replace graph embeddings after adaptation. The shipped R2R configuration uses one environment, 30,000 iterations, learning rate 1e-5, adaptation learning rate 6e-7 and zero drift-decay coefficient. Its default backtracking option is teleport, so a reproduction must explicitly select and document the continuous-control protocol. README documents assets and the SS-ETP trainer. No checkpoint execution establishes which released branch produced each published table. The code supports delayed residual adaptation, but does not justify treating endogenous immediate refinement as an observed prediction error.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b](https://github.com/sunrise-yuze/SC2_WM/tree/f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b).

- [README.md](https://github.com/sunrise-yuze/SC2_WM/blob/f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b/README.md#L1)
- [run_r2r/iter_train.yaml](https://github.com/sunrise-yuze/SC2_WM/blob/f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b/run_r2r/iter_train.yaml#L1)
- [vlnce_baselines/models/etp/vilmodel_cmt.py](https://github.com/sunrise-yuze/SC2_WM/blob/f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b/vlnce_baselines/models/etp/vilmodel_cmt.py#L1)
- [vlnce_baselines/ss_trainer_ETP.py](https://github.com/sunrise-yuze/SC2_WM/blob/f2f2f5ff2e1f21a08f7d7c51fbbd5c17b0d6778b/vlnce_baselines/ss_trainer_ETP.py#L320)

## navmorph

INSPECTED CODE: `vlnce_baselines/models/etp/vilmodel_cmt.py:741–759,942–966` concatenates 768-dimensional visual and position embeddings, learns 1536-dimensional Gaussian prior/posterior projections, samples with bounded standard deviation, and decodes a three-dimensional displacement. `utils_p/losses.py:56–98` implements Gaussian KL with separate stop-gradient terms. `vlnce_baselines/ss_trainer_ETP.py:1352–1392` constructs the real-state posterior and invokes probability/action losses; notably the probability-loss call precedes assignment of the current posterior, so time indexing requires verification before equating it with the printed current posterior/prior objective. Lines 1389–1394 push and retrieve memory during evaluation. Lines 1421–1454 use nearest reference-path coordinates during training and retrieve memory before imagination. `utils_p/memory.py:125–175,220–246` confirms finite-capacity nearest-neighbor updates and cosine retrieval. This is forward memory adaptation, not evidence that model weights are optimized at deployment. The release includes its training/evaluation trainer and configs, but no benchmark was run and the detailed training-time alignment warrants an execution audit.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [7fb0f4aa44ff3e042335d8de0d1b2197c6f1f657](https://github.com/Feliciaxyao/NavMorph/tree/7fb0f4aa44ff3e042335d8de0d1b2197c6f1f657).

- [utils_p/memory.py](https://github.com/Feliciaxyao/NavMorph/blob/7fb0f4aa44ff3e042335d8de0d1b2197c6f1f657/utils_p/memory.py#L125)
- [vlnce_baselines/models/etp/vilmodel_cmt.py](https://github.com/Feliciaxyao/NavMorph/blob/7fb0f4aa44ff3e042335d8de0d1b2197c6f1f657/vlnce_baselines/models/etp/vilmodel_cmt.py#L560)
- [vlnce_baselines/ss_trainer_ETP.py](https://github.com/Feliciaxyao/NavMorph/blob/7fb0f4aa44ff3e042335d8de0d1b2197c6f1f657/vlnce_baselines/ss_trainer_ETP.py#L1348)

## navq

INSPECTED CODE: `code/Q_pretrain/GPT_VL.py:125–240` forms visual/text/action sequences and predicts 512-dimensional future features with squared-error supervision, rather than a scalar reward Q-function. `code/Q_pretrain/dataset.py:180–244` explicitly constructs the target from stored shortest paths: immediate caption features plus gamma-discounted downstream features weighted by branch probabilities. Only one stored shortest path per destination is used, resolving ties operationally without proving graph-theoretic uniqueness. Earlier dataset code averages 36 views into 12 and optionally adds ScaleVLN scenes; `train.py` contains optimizer and MSE updates. This supports the successor-feature interpretation and identifies a concrete data/architecture confound. The audit covers Q-model construction, supervision, training and dataset preparation; it does not certify every downstream navigation/evaluation branch or replicate the main tables.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [9465fe43c9c1583bd7ba390b6be30996ea4f5a5c](https://github.com/woyut/NavQ_ICCV25/tree/9465fe43c9c1583bd7ba390b6be30996ea4f5a5c).

- [code/Q_pretrain/GPT_VL.py](https://github.com/woyut/NavQ_ICCV25/blob/9465fe43c9c1583bd7ba390b6be30996ea4f5a5c/code/Q_pretrain/GPT_VL.py#L125)
- [code/Q_pretrain/dataset.py](https://github.com/woyut/NavQ_ICCV25/blob/9465fe43c9c1583bd7ba390b6be30996ea4f5a5c/code/Q_pretrain/dataset.py#L1)

## dreamer

INSPECTED CODE: `dreamer.py:145–275` optimizes image/reward likelihoods and KL with free nats, then actor and value objectives over imagined latent trajectories using lambda returns and separate optimizers. `models.py:25–110` contains recurrent deterministic state plus diagonal-Gaussian stochastic posterior/prior; standard deviation is softplus plus 0.1. `dreamer.py:136–145` updates the posterior using a real observation before actor inference. `wrappers.py:1–80` supplies DeepMind Control images and action/reward handling; `dreamer.py:329–340,441–449` supplies normalization and evaluation/training separation. The model supports gradients through latent dynamics for actor optimization. This is inspected TensorFlow source, not an executed control experiment; historical dependencies and assets remain prerequisites.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [56d4d444dfd0582b0e79dab80aebbea74c0ce40d](https://github.com/danijar/dreamer/tree/56d4d444dfd0582b0e79dab80aebbea74c0ce40d).

- [dreamer.py](https://github.com/danijar/dreamer/blob/56d4d444dfd0582b0e79dab80aebbea74c0ce40d/dreamer.py#L32)
- [models.py](https://github.com/danijar/dreamer/blob/56d4d444dfd0582b0e79dab80aebbea74c0ce40d/models.py#L25)
- [wrappers.py](https://github.com/danijar/dreamer/blob/56d4d444dfd0582b0e79dab80aebbea74c0ce40d/wrappers.py#L1)

## planet

INSPECTED CODE: `planet/models/rssm.py:96–139` implements recurrent belief, Gaussian prior, and observation-conditioned posterior; decoder features concatenate stochastic state and belief. `planet/training/define_model.py:52–78` connects posterior/prior inference to configured objectives and active episode collection. `planet/scripts/configs.py:180–198` exposes KL/free-nats, optional overshooting, reward-loss scaling and Adam settings; overshooting is not enabled merely because its code exists. `planet/control/planning.py:24–60` samples action sequences, rolls out latent dynamics, scores predicted reward, and refits CEM elites. `planet/control/mpc_agent.py:57–83` executes only the first selected action and updates from the next real observation. `planet/tools/preprocess.py:22–30` reduces images to five bits by default and adds dequantization noise. The source provides training and control interfaces; this audit does not execute them or verify historical checkpoint compatibility.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [c04226b6db136f5269625378cd6a0aa875a92842](https://github.com/google-research/planet/tree/c04226b6db136f5269625378cd6a0aa875a92842).

- [planet/control/mpc_agent.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/mpc_agent.py#L38)
- [planet/control/planning.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/control/planning.py#L22)
- [planet/models/rssm.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/models/rssm.py#L45)
- [planet/scripts/configs.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/configs.py#L39)
- [planet/scripts/objectives.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/scripts/objectives.py#L20)
- [planet/tools/preprocess.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/tools/preprocess.py#L20)
- [planet/training/define_model.py](https://github.com/google-research/planet/blob/c04226b6db136f5269625378cd6a0aa875a92842/planet/training/define_model.py#L35)

## worldmodels

INSPECTED CODE: the article website links the actual author implementation, `hardmaru/WorldModelsExperiments`, inspected at the separate immutable snapshot recorded here. `carracing/vae/vae.py:14,64–87` confirms a 32-dimensional VAE with reconstruction MSE plus KL floor and Adam. `carracing/rnn/rnn.py:39–45,139–168` confirms 256 recurrent units, five Gaussian mixtures and negative log likelihood of next latents conditioned on actions. `carracing/series.py:35–39` normalizes 64×64 RGB and samples VAE latents. `carracing/model.py:98–114,154–239` implements the controller and episode evaluation; `carracing/train.py:89–93,354–360` implements CMA-ES and evaluation. `doomrnn/doomrnn.py:24,624–655` samples the learned simulator with temperature 1.25 and learned termination. This release setting differs from the paper's highlighted 1.15 setting; reproduce the intended configuration explicitly. No historical TensorFlow/Gym code or model checkpoint was executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [fd982b9691a941b52c6addbde29bc801ca6202c8](https://github.com/hardmaru/WorldModelsExperiments/tree/fd982b9691a941b52c6addbde29bc801ca6202c8).

- [carracing/model.py](https://github.com/hardmaru/WorldModelsExperiments/blob/fd982b9691a941b52c6addbde29bc801ca6202c8/carracing/model.py#L96)
- [carracing/rnn/rnn.py](https://github.com/hardmaru/WorldModelsExperiments/blob/fd982b9691a941b52c6addbde29bc801ca6202c8/carracing/rnn/rnn.py#L17)
- [carracing/series.py](https://github.com/hardmaru/WorldModelsExperiments/blob/fd982b9691a941b52c6addbde29bc801ca6202c8/carracing/series.py#L33)
- [carracing/train.py](https://github.com/hardmaru/WorldModelsExperiments/blob/fd982b9691a941b52c6addbde29bc801ca6202c8/carracing/train.py#L20)
- [carracing/vae/vae.py](https://github.com/hardmaru/WorldModelsExperiments/blob/fd982b9691a941b52c6addbde29bc801ca6202c8/carracing/vae/vae.py#L12)
- [doomrnn/doomrnn.py](https://github.com/hardmaru/WorldModelsExperiments/blob/fd982b9691a941b52c6addbde29bc801ca6202c8/doomrnn/doomrnn.py#L22)

## dinowm

INSPECTED CODE: `models/visual_world_model.py:1–180,190–309` encodes images into DINO patch features, concatenates/replaces action tokens, and trains prediction against detached future embeddings. The optional decoder has a separate reconstruction role. `planning/cem.py:106–118` samples candidate actions, calls world-model rollout and refits elites; it does not require a pixel decoder for planning. `conf/train.yaml` freezes the encoder and uses a three-frame history, one-step prediction and predictor learning rate 5e-4. `planning/evaluator.py:112–163` denormalizes actions and evaluates actual environment rollouts. Model/loss, planning, configuration and evaluator paths were inspected; all task-specific dataset preprocessing and checkpoints were not executed or exhaustively audited.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [0a9492fa12044b852ae9e001cc74604b79c8bb0c](https://github.com/gaoyuezhou/dino_wm/tree/0a9492fa12044b852ae9e001cc74604b79c8bb0c).

- [conf/train.yaml](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/train.yaml#L33)
- [models/visual_world_model.py](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py#L1)
- [planning/cem.py](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/planning/cem.py#L10)
- [planning/evaluator.py](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/planning/evaluator.py#L110)

## cvaml

INSPECTED CODE: `mad_td/training_util/model_loss_functions.py:120–171` evaluates sampled model next states using the critic, subtracts a variance correction, and minimizes squared mismatch of sample-mean values. Lines 156–158 use population sample variance (`ddof=0`) divided by N. Under the restricted assumption of independent samples and a fixed value function, an unbiased correction would use unbiased sample variance divided by N; this implementation leaves a sigma²/N² residual. Its action may depend on the first sampled next state, so the simple independent-sample derivation is not a general diagnosis of training behavior. Lines 390–425 show objective dispatch and target-critic use. `config/algo/corrected_itervaml.yaml` uses four stochastic samples and retains likelihood-related model losses. `mad_td/trainers/mad_td_model.py:40–95` initializes encoder, actor, critic and latent-model train states from replay data. `mad_td/utils/eval_utils.py:25–50` evaluates on-policy one-step tuples using saved physics state. No JAX training or numerical reproduction of paper performance was executed; the estimator discrepancy is a scoped mathematical deduction requiring controlled comparison.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [e522ef376992475e57982bccf489338e8ff0be7a](https://github.com/adaptive-agents-lab/CVAML/tree/e522ef376992475e57982bccf489338e8ff0be7a).

- [config/algo/corrected_itervaml.yaml](https://github.com/adaptive-agents-lab/CVAML/blob/e522ef376992475e57982bccf489338e8ff0be7a/config/algo/corrected_itervaml.yaml#L1)
- [mad_td/trainers/mad_td_model.py](https://github.com/adaptive-agents-lab/CVAML/blob/e522ef376992475e57982bccf489338e8ff0be7a/mad_td/trainers/mad_td_model.py#L1)
- [mad_td/training_util/model_loss_functions.py](https://github.com/adaptive-agents-lab/CVAML/blob/e522ef376992475e57982bccf489338e8ff0be7a/mad_td/training_util/model_loss_functions.py#L100)
- [mad_td/utils/eval_utils.py](https://github.com/adaptive-agents-lab/CVAML/blob/e522ef376992475e57982bccf489338e8ff0be7a/mad_td/utils/eval_utils.py#L20)

## wmnav

INSPECTED CODE: `src/WMNav_agent.py:1244–1302` parses VLM scores, updates curiosity with elementwise minimum, and chooses the highest mean direction. There is an important exception: identified goal-mask cells are reset to 10 at lines 1286–1287. Thus ordinary merge updates cannot recover an underestimated exploration cell, but goal identification can override it. `src/WMNav_env.py:356–389` collects 12 views at roughly 30-degree rotations and applies directional masks, then carries subtask text forward. `src/WMNav_agent.py:752–764` uses 0.8 m internal goal stopping; `config/WMNav.yaml:20–43` ships Gemini 2.0 Flash, depth-sensor navigability, 40 macro steps and a 1 m evaluation success threshold. These are not the paper's headline Gemini 1.5 Pro configuration. No task-specific training optimizer or learned transition loss is part of this prompt-based method. API access, Habitat assets and exact model/version settings are required to reproduce it.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [57b74cee7a00b669733e8ca917f34a8aadfd6fd0](https://github.com/B0B8K1ng/WMNavigation/tree/57b74cee7a00b669733e8ca917f34a8aadfd6fd0).

- [config/WMNav.yaml](https://github.com/B0B8K1ng/WMNavigation/blob/57b74cee7a00b669733e8ca917f34a8aadfd6fd0/config/WMNav.yaml#L1)
- [src/WMNav_agent.py](https://github.com/B0B8K1ng/WMNavigation/blob/57b74cee7a00b669733e8ca917f34a8aadfd6fd0/src/WMNav_agent.py#L13)
- [src/WMNav_env.py](https://github.com/B0B8K1ng/WMNavigation/blob/57b74cee7a00b669733e8ca917f34a8aadfd6fd0/src/WMNav_env.py#L99)

## eoe

INSPECTED CODE: `eval_ood_detection.py:23–49` loads a frozen image-text model, constructs ID and envisioned OOD labels, and evaluates scores. `utils/detection_util.py:98–174` computes normalized image/text similarity, joint softmax, and the maximum ID probability minus beta times maximum envisioned-OOD probability for the EOE branch. AUROC/FPR95 are computed by its metric utilities, not navigation metrics. Lines 203–236 implement semantic filtering at cosine 0.85 specifically for near-OOD ImageNet20, while the default threshold is 1.0. `utils/generate_llm_class.py:187–227` loads, lowercases and deduplicates generated class names. Thus it is inaccurate to describe all filtering as exact-string-only. This method requires no detector fine-tuning, but does rely on pretrained CLIP and LLM class generation; the released candidate lists improve repeatability without guaranteeing coverage of unknown classes.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [20fa95e5baff9242b93f0957bb4564db3d36d1f2](https://github.com/tmlr-group/EOE/tree/20fa95e5baff9242b93f0957bb4564db3d36d1f2).

- [eval_ood_detection.py](https://github.com/tmlr-group/EOE/blob/20fa95e5baff9242b93f0957bb4564db3d36d1f2/eval_ood_detection.py#L1)
- [utils/detection_util.py](https://github.com/tmlr-group/EOE/blob/20fa95e5baff9242b93f0957bb4564db3d36d1f2/utils/detection_util.py#L9)
- [utils/generate_llm_class.py](https://github.com/tmlr-group/EOE/blob/20fa95e5baff9242b93f0957bb4564db3d36d1f2/utils/generate_llm_class.py#L13)

## pathdreamer

INSPECTED CODE: `models/pathdreamer_models.py:193–258` projects RGB/depth/semantics into persistent point-cloud memory; lines 260–377 reproject into a requested pose, complete structure, synthesize RGB and optionally add predictions back into memory. The default `add_preds_to_memory=False` is an explicit provenance choice. `models/point_cloud_models.py:205–220` creates Gaussian prior/posterior modules; `models/pathdreamer_config.py:30–38` sets panoramic dimensions and circular padding. README provides two pretrained checkpoints and an inference Colab. The inspected release contains inference/model tests, but no full training launcher or downstream beam-search navigation implementation in its source listing; therefore the paper's training losses and navigation experiments cannot be fully reproduced from this repository alone. No checkpoint was downloaded or run.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [351cc2ff79e079d8e280aff95666e2323eb4ee9b](https://github.com/google-research/pathdreamer/tree/351cc2ff79e079d8e280aff95666e2323eb4ee9b).

- [README.md](https://github.com/google-research/pathdreamer/blob/351cc2ff79e079d8e280aff95666e2323eb4ee9b/README.md#L23)
- [models/pathdreamer_config.py](https://github.com/google-research/pathdreamer/blob/351cc2ff79e079d8e280aff95666e2323eb4ee9b/models/pathdreamer_config.py#L30)
- [models/pathdreamer_models.py](https://github.com/google-research/pathdreamer/blob/351cc2ff79e079d8e280aff95666e2323eb4ee9b/models/pathdreamer_models.py#L13)
- [models/point_cloud_models.py](https://github.com/google-research/pathdreamer/blob/351cc2ff79e079d8e280aff95666e2323eb4ee9b/models/point_cloud_models.py#L160)

## hnr

INSPECTED CODE: `NeRF/models/etp/nerf.py:148–195` computes alpha/transmittance weights and normalized rendered features; sparse neighbor search uses geometric distances and peak selection. `NeRF/ss_trainer_ETP.py:1065–1106` crops and encodes CLIP regions, preprocesses depth, and invokes rendering supervision. Lines 1162–1177 explicitly use 2×2 rendering for fast feature evaluation versus 224×224 for visualization. `run_r2r/iter_train_nerf.yaml` exposes 15,000 iterations, batch one and learning rate 1e-5. The available snapshot is centered on NeRF training/rendering evaluation; the complete lookahead navigation tree/policy reproduction was not established from the inspected files. Its 11.5 ms paper timing concerns rendering, not a certified end-to-end robot control cycle. Historical CUDA neighbor-search and Habitat dependencies remain prerequisites.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [6c935c1f6f0d715b1c6bef28724bdfabb48b159e](https://github.com/MrZihan/HNR-VLN/tree/6c935c1f6f0d715b1c6bef28724bdfabb48b159e).

- [NeRF/models/etp/nerf.py](https://github.com/MrZihan/HNR-VLN/blob/6c935c1f6f0d715b1c6bef28724bdfabb48b159e/NeRF/models/etp/nerf.py#L17)
- [NeRF/ss_trainer_ETP.py](https://github.com/MrZihan/HNR-VLN/blob/6c935c1f6f0d715b1c6bef28724bdfabb48b159e/NeRF/ss_trainer_ETP.py#L1065)
- [run_r2r/iter_train_nerf.yaml](https://github.com/MrZihan/HNR-VLN/blob/6c935c1f6f0d715b1c6bef28724bdfabb48b159e/run_r2r/iter_train_nerf.yaml#L3)

## chasing_ghosts

INSPECTED CODE: `tracker/modules/navigators.py:142–189` builds spatial and language-conditioned motion kernels, renormalizes their product, and propagates log belief. Lines 279–286 combine motion with language-conditioned observation likelihood. `tracker/modules/loss.py:20–36` implements KL supervision against target belief maps. `tracker/train_filter.py:181–262` creates target heatmaps from reference coordinates and uses truncated backpropagation; lines 51–103 validate prediction on supervised paths. This confirms that the filter predicts the instruction's ideal agent, not a learned photorealistic transition model. The repository also supplies policy and evaluator modules, but their complete checkpoint-level reproduction and preprocessing dependency chain were not executed; source inspection here is strongest for the learned filter rather than the complete deployed policy.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [f819aea21b94d9d3e23d9b6b9264054ee50c007b](https://github.com/batra-mlp-lab/vln-chasing-ghosts/tree/f819aea21b94d9d3e23d9b6b9264054ee50c007b).

- [tracker/modules/loss.py](https://github.com/batra-mlp-lab/vln-chasing-ghosts/blob/f819aea21b94d9d3e23d9b6b9264054ee50c007b/tracker/modules/loss.py#L3)
- [tracker/modules/navigators.py](https://github.com/batra-mlp-lab/vln-chasing-ghosts/blob/f819aea21b94d9d3e23d9b6b9264054ee50c007b/tracker/modules/navigators.py#L26)
- [tracker/train_filter.py](https://github.com/batra-mlp-lab/vln-chasing-ghosts/blob/f819aea21b94d9d3e23d9b6b9264054ee50c007b/tracker/train_filter.py#L16)

## learning_to_map

INSPECTED CODE: `models/predictors/map_predictor_model.py:45–62` trains occupancy and semantic maps with separate cross-entropy terms. `tester.py:245–282,433–442` preprocesses RGB/depth, runs the ensemble, and registers semantic means and per-class variance. `trainer_active.py:344–352` computes classwise ensemble variance and averages it for active collection; Adam initialization is at 61–65. `tester.py:348–367` reports distance, success, SPL and soft-SPL from the navigation trajectory. This is an ensemble-disagreement heuristic, without a code mechanism proving probabilistic coverage. Data, semantic labels, pose assumptions and the frozen point-navigation controller remain reproduction prerequisites. No simulator episode was executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [afa7081241b4e2407286ebe7dc41d2a32f10e906](https://github.com/ggeorgak11/L2M/tree/afa7081241b4e2407286ebe7dc41d2a32f10e906).

- [models/predictors/map_predictor_model.py](https://github.com/ggeorgak11/L2M/blob/afa7081241b4e2407286ebe7dc41d2a32f10e906/models/predictors/map_predictor_model.py#L6)
- [tester.py](https://github.com/ggeorgak11/L2M/blob/afa7081241b4e2407286ebe7dc41d2a32f10e906/tester.py#L50)
- [trainer_active.py](https://github.com/ggeorgak11/L2M/blob/afa7081241b4e2407286ebe7dc41d2a32f10e906/trainer_active.py#L41)

## sgm

INSPECTED CODE: `SGM_train/main_pretrain.py:38,170–198` selects `models_mae_cross` and AdamW; `engine_pretrain.py:75–99` applies its masked-map loss and optimizer update. A material discrepancy appears in `models_mae_cross.py:254–295`: the objective includes channel-weighted squared error and BCE, and its IoU term is computed after hard thresholding and `.detach()`. Consequently the IoU addend provides no gradient in this released path, despite the paper describing IoU as part of the optimization objective. `SGM_train/sgm/dataset.py:819–859` reads precomputed semantic maps and samples masks. `semexp/eval_sgm.py:178–184,604–606,944–947` reconstructs resized map crops and passes them to target selection; lines 1073–1080 aggregate ObjectNav metrics. Ground-truth map helper functions also exist for visualization/oracle paths; their presence alone does not establish test leakage. Released assets and exact flags need checking in an executable reproduction.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [579982dae2fe040d2fa26e82778352d950e31a19](https://github.com/sx-zhang/SGM/tree/579982dae2fe040d2fa26e82778352d950e31a19).

- [SGM_train/engine_pretrain.py](https://github.com/sx-zhang/SGM/blob/579982dae2fe040d2fa26e82778352d950e31a19/SGM_train/engine_pretrain.py#L50)
- [SGM_train/main_pretrain.py](https://github.com/sx-zhang/SGM/blob/579982dae2fe040d2fa26e82778352d950e31a19/SGM_train/main_pretrain.py#L36)
- [SGM_train/models_mae_cross.py](https://github.com/sx-zhang/SGM/blob/579982dae2fe040d2fa26e82778352d950e31a19/SGM_train/models_mae_cross.py#L26)
- [SGM_train/sgm/dataset.py](https://github.com/sx-zhang/SGM/blob/579982dae2fe040d2fa26e82778352d950e31a19/SGM_train/sgm/dataset.py#L18)
- [semexp/eval_sgm.py](https://github.com/sx-zhang/SGM/blob/579982dae2fe040d2fa26e82778352d950e31a19/semexp/eval_sgm.py#L9)
- [semexp/sgm_adds.py](https://github.com/sx-zhang/SGM/blob/579982dae2fe040d2fa26e82778352d950e31a19/semexp/sgm_adds.py#L52)

## nwm

INSPECTED CODE: `models.py:233–241` adds diffusion-time, relative-time and action embeddings and attends to encoded visual context. `train.py:127–138,188–258,290–303` instantiates CDiT, combines robot datasets and optimizes the diffusion loss with AdamW. Its periodic helper at 366–405 evaluates only the first sampled batch; this is training-time monitoring, not necessarily the paper's full evaluation. `planning_eval.py:217–299` implements LPIPS-driven CEM, but its candidate displacement is repeated across the predicted horizon, with a final-yaw adjustment. Therefore this released planner is more restricted than unconstrained optimization of every action in an arbitrary eight-step trajectory. Lines 366–376 evaluate offline ATE/RPE against logged trajectories. README records unavailable redistributable high-resolution data. No online robot navigation, training or checkpoint evaluation was executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [3f6cd8e70d6f2d1e2b9684acff510710135f0f41](https://github.com/facebookresearch/nwm/tree/3f6cd8e70d6f2d1e2b9684acff510710135f0f41).

- [models.py](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/models.py#L137)
- [planning_eval.py](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/planning_eval.py#L13)
- [train.py](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/train.py#L104)

## genex

INSPECTED CODE: `code/train/explorer/train_world_explorer.py:162–169,1233–1301` normalizes panoramic data, computes diffusion MSE, and optionally decodes/rotates/re-encodes frames for spherical consistency. The released mixing schedule ramps its weight and disables the spherical contribution if it exceeds ten times video loss; this is more specific than a fixed two-term objective. `code/inference/explorer.py:122–158` rotates panoramas and recursively feeds generated last frames into further motion generation. `code/evaluate/IECC/iecc_score.py:12–42,73–77` computes Inception-v4 feature MSE after 299×299 normalization. This tests generated loop consistency, not truth about unobserved physical states. The audited code does not establish a calibrated Bayesian belief updater or prove imagined observations are independent evidence.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [eda604588d0a205c57e2b2599eec566d97162c2e](https://github.com/Beckschen/GenEx/tree/eda604588d0a205c57e2b2599eec566d97162c2e).

- [code/evaluate/IECC/iecc_score.py](https://github.com/Beckschen/GenEx/blob/eda604588d0a205c57e2b2599eec566d97162c2e/code/evaluate/IECC/iecc_score.py#L9)
- [code/inference/explorer.py](https://github.com/Beckschen/GenEx/blob/eda604588d0a205c57e2b2599eec566d97162c2e/code/inference/explorer.py#L3)
- [code/train/explorer/train_world_explorer.py](https://github.com/Beckschen/GenEx/blob/eda604588d0a205c57e2b2599eec566d97162c2e/code/train/explorer/train_world_explorer.py#L162)

## fantasyvln

INSPECTED CODE: `trainer.py:145–178,201–217` forms hard losses for reasoning tokens and soft cross-entropy for action tokens using detached non-CoT teacher logits. Lines 571–626 first update the non-CoT branch, then refresh teacher outputs and update CoT branches. This confirms the direction is non-CoT→CoT for the explicit action-alignment loss; shared parameters may still transfer useful representation learning back to the direct mode. `train.py:33–37,100–129` passes alignment options and prepared datasets into the trainer. `lhvln/NavModel/VLMModel.py:170–217` processes history/current images and generates action text through the common VLM. No separate online VAR rollout appears in this inference function. Benchmark metrics/config and all external visual-token assets were not independently executed; code inspection does not establish faithful chain-of-thought reasoning.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [01be27681184b1b878d79ac9fd434f032051427a](https://github.com/Fantasy-AMAP/fantasy-vln/tree/01be27681184b1b878d79ac9fd434f032051427a).

- [lhvln/NavModel/VLMModel.py](https://github.com/Fantasy-AMAP/fantasy-vln/blob/01be27681184b1b878d79ac9fd434f032051427a/lhvln/NavModel/VLMModel.py#L170)
- [train.py](https://github.com/Fantasy-AMAP/fantasy-vln/blob/01be27681184b1b878d79ac9fd434f032051427a/train.py#L5)
- [trainer.py](https://github.com/Fantasy-AMAP/fantasy-vln/blob/01be27681184b1b878d79ac9fd434f032051427a/trainer.py#L122)

## lcvn

INSPECTED CODE: `lcvn-wm/algorithms/ldit/ldit_video_social.py:148–235` encodes language, samples per-frame noise and optimizes conditional diffusion. `lcvn-ac/lcvn_ac/behavior_models/lumos.py:1449–1503` uses detached lambda-return targets for the critic, KL-balanced latent plans, and REINFORCE (`-log_prob * detached_return`) for the actor when log probabilities are supplied. Its comment explicitly says diffusion rollouts are no-grad; the legacy negative-return fallback therefore cannot supply a dynamics path gradient. This is not automatically Dreamer-style differentiable actor learning. `lcvn-uni/train.py:338–450` performs supervised token training and distinct single-step/task/rollout evaluations. `lcvn-uni/scripts/navigation.py:59–89` defaults to random 80/10/10 trajectory splits within a data root; held-out robot-source results require their separate dataset configuration. `scripts/metrics.py:20–63` computes unaligned ATE/RPE, and 67–100 preprocesses images for visual metrics. The complete multimodal decoder and all benchmark split manifests were not exhaustively audited or executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [de1781720f40e4a3a3745878fa202db3385c74fb](https://github.com/UWMILab/LCVN/tree/de1781720f40e4a3a3745878fa202db3385c74fb).

- [lcvn-ac/lcvn_ac/behavior_models/lumos.py](https://github.com/UWMILab/LCVN/blob/de1781720f40e4a3a3745878fa202db3385c74fb/lcvn-ac/lcvn_ac/behavior_models/lumos.py#L12)
- [lcvn-uni/scripts/metrics.py](https://github.com/UWMILab/LCVN/blob/de1781720f40e4a3a3745878fa202db3385c74fb/lcvn-uni/scripts/metrics.py#L1)
- [lcvn-uni/scripts/navigation.py](https://github.com/UWMILab/LCVN/blob/de1781720f40e4a3a3745878fa202db3385c74fb/lcvn-uni/scripts/navigation.py#L1)
- [lcvn-uni/train.py](https://github.com/UWMILab/LCVN/blob/de1781720f40e4a3a3745878fa202db3385c74fb/lcvn-uni/train.py#L14)
- [lcvn-wm/algorithms/ldit/ldit_video_social.py](https://github.com/UWMILab/LCVN/blob/de1781720f40e4a3a3745878fa202db3385c74fb/lcvn-wm/algorithms/ldit/ldit_video_social.py#L66)

## vid2world

INSPECTED CODE: `main/utils_train.py:342–366` implements three-tap extrapolative transfer as weights [0, w0-w2, w1+2w2], matching a linear future-feature approximation. `configs/navigation/config_recon_train.yaml` selects extrapolative transfer, diffusion-forcing loss, causal attention, three-dimensional actions and 0.2 action dropout. `lvdm/models/ddpm3d.py:1144–1158` supports independent per-time-step action masks. `lvdm/data/reconvid.py:178–210` reads video/action pairs, applies transforms and normalizes frames to [-1,1]. Sampler code implements conditional guidance; the full action-specific evaluation branch was not exhaustively traced. Causal masking prevents future-frame leakage in this architecture, but neither these files nor the paper's guidance identity establish intervention identification from observational data. Exact video/checkpoint benchmarks were not executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [ae24077adbd0783bc6e1b79f705d9b8fad5888ce](https://github.com/thuml/Vid2World/tree/ae24077adbd0783bc6e1b79f705d9b8fad5888ce).

- [configs/navigation/config_recon_train.yaml](https://github.com/thuml/Vid2World/blob/ae24077adbd0783bc6e1b79f705d9b8fad5888ce/configs/navigation/config_recon_train.yaml#L1)
- [lvdm/data/reconvid.py](https://github.com/thuml/Vid2World/blob/ae24077adbd0783bc6e1b79f705d9b8fad5888ce/lvdm/data/reconvid.py#L9)
- [lvdm/models/ddpm3d.py](https://github.com/thuml/Vid2World/blob/ae24077adbd0783bc6e1b79f705d9b8fad5888ce/lvdm/models/ddpm3d.py#L349)
- [lvdm/models/samplers/ddim_multiplecond.py](https://github.com/thuml/Vid2World/blob/ae24077adbd0783bc6e1b79f705d9b8fad5888ce/lvdm/models/samplers/ddim_multiplecond.py#L78)
- [lvdm/modules/networks/openaimodel3d.py](https://github.com/thuml/Vid2World/blob/ae24077adbd0783bc6e1b79f705d9b8fad5888ce/lvdm/modules/networks/openaimodel3d.py#L58)
- [main/utils_train.py](https://github.com/thuml/Vid2World/blob/ae24077adbd0783bc6e1b79f705d9b8fad5888ce/main/utils_train.py#L342)

## visualplanning

INSPECTED CODE: `train_rl_maze.py:79–106` invokes a symbolic action parser with the true layout and distance map, rewarding a distance-decreasing valid transition by 1, a valid nonprogressing transition by 0, and invalid output by -5. Lines 171–210 configure GRPO/LoRA training. `evaluation/visual_planning_evaluator.py:418–432` checks parsed validity and target coordinates at the expected shortest length; MiniBehavior uses task-specific pick/drop logic at 596–602. The release therefore depends on privileged rule-based validation even though its model emits only image tokens. Some imported custom trainer/parser/tokenizer infrastructure is external or absent from the small snapshot; it is not a self-contained reproduction package. No training or planning benchmark was executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [081e67ca6926318623a73b279d45cf0f7dc2b8fa](https://github.com/yix8/VisualPlanning/tree/081e67ca6926318623a73b279d45cf0f7dc2b8fa).

- [evaluation/visual_planning_evaluator.py](https://github.com/yix8/VisualPlanning/blob/081e67ca6926318623a73b279d45cf0f7dc2b8fa/evaluation/visual_planning_evaluator.py#L74)
- [train_rl_maze.py](https://github.com/yix8/VisualPlanning/blob/081e67ca6926318623a73b279d45cf0f7dc2b8fa/train_rl_maze.py#L1)

## visualimaginations

INSPECTED CODE: `VLN-DUET/map_nav_src/models/vilmodel.py:562–655` adds an imagination type embedding and aligns projected image features with mean noun-phrase embeddings by one minus cosine similarity. Lines 1154–1166 concatenate imagination and text tokens for both global/local encoders. `VLN-DUET/map_nav_src/r2r/agent.py:317–382` loads cached instruction-keyed image features and masks; 412–449 constructs alignment inputs and 621–623 adds its weighted auxiliary loss. Thus deployment uses static text-conditioned prototypes rather than predicting the actual next view conditioned on a candidate action. The repository contains broader HAMT/DUET training/evaluation scaffolds; all scripts, generated-image provenance and multi-stage optimizer flags were not exhaustively checked, and no benchmark was run.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [31816e52e9913e9993a5ab05cf1e99b56044e84a](https://github.com/akhilperincherry/VLN-Imagine/tree/31816e52e9913e9993a5ab05cf1e99b56044e84a).

- [VLN-DUET/map_nav_src/models/vilmodel.py](https://github.com/akhilperincherry/VLN-Imagine/blob/31816e52e9913e9993a5ab05cf1e99b56044e84a/VLN-DUET/map_nav_src/models/vilmodel.py#L560)
- [VLN-DUET/map_nav_src/r2r/agent.py](https://github.com/akhilperincherry/VLN-Imagine/blob/31816e52e9913e9993a5ab05cf1e99b56044e84a/VLN-DUET/map_nav_src/r2r/agent.py#L315)

## vlnsig

INSPECTED CODE: `pretrain_src/model/pretrain_cmt.py:475–520,595–663` selects a dynamic vocabulary, applies log-softmax and KL loss against future visual-token distributions, and returns detached target/difficulty statistics. This verifies auxiliary semantic prediction rather than an inference-time image planner. The repository contains pretraining and finetuning entrypoints, but their complete dynamic-vocabulary update schedule, preprocessing caches and every HAMT/DUET evaluation branch were not exhaustively audited. The separately reviewed supplement documents the task mixture, vocabulary settings, and the fact that displayed decoded images retain 70% ground-truth tokens. No cached features, checkpoints or navigation evaluation were executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [603bbd2f40fe47b62c787af567d0cef5a462b301](https://github.com/jialuli-luka/VLN-SIG/tree/603bbd2f40fe47b62c787af567d0cef5a462b301).

- [pretrain_src/model/pretrain_cmt.py](https://github.com/jialuli-luka/VLN-SIG/blob/603bbd2f40fe47b62c787af567d0cef5a462b301/pretrain_src/model/pretrain_cmt.py#L295)

## womap

INSPECTED CODE: `src/models/worldmodel.py:260–300,677–700` uses sampled Gaussian next latents with smooth-L1 reconstruction and KL regularization, alongside confidence/bounding-box losses. This is more specific than a generic KL between learned and true transition distributions. `joint_gradient_planner` at 1180–1250 samples next latent and reward distributions, optionally divides reward by predicted standard deviation, and minimizes abs(1 - weighted_reward) by differentiating through the model. No action-change L1 smoothing term is present in this inspected planner, so the paper's printed positive sign cannot be explained as a matching implementation. Other planner branches and all deployment dispatches were not exhaustively traced; this is a precise source discrepancy, not proof that every reported run used this branch. Detector-reward calibration, stochastic uncertainty coverage, preprocessing of every Gaussian scene, and actual hardware evaluation remain unexecuted. The model/loss/planner audit is substantive, but complete end-to-end reproduction is not certified.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [3f75333d788ac3fe0cd05bed3781104b7a0b859a](https://github.com/irom-princeton/womap/tree/3f75333d788ac3fe0cd05bed3781104b7a0b859a).

- [src/models/worldmodel.py](https://github.com/irom-princeton/womap/blob/3f75333d788ac3fe0cd05bed3781104b7a0b859a/src/models/worldmodel.py#L260)

## monodream

INSPECTED CODE: current primary-source search found the official MonoDream implementation inside HorizonRobotics/RoboOrchardLab, correcting the initial no-code search result. `projects/monodream/README.md:18–27,54–80,108–118` links author project/checkpoints and describes Habitat 0.1.7 plus an external VLN-CE dependency. `robo_orchard_lab/models/monodream/navigation_arch.py:380–396,478–496` runs direct language-model generation; `model.py:99–106` wraps training/inference calls. `processor.py:115–148` prepares multimodal inputs. `projects/monodream/infer.py:338–380` executes a queue of up to three primitive actions before another prediction, while 142–158 steps Habitat and collects metrics. `analyze_results.py:49–65` aggregates success, SPL and distance from episode logs. These files establish released inference/model support, not a complete implementation of the paper's four panoramic reconstruction losses: the monodream project listing does not include its training launcher or auxiliary-objective pipeline. This limitation is narrower than no implementation. No checkpoint or Habitat episode was executed.

Repository status: Scoped source audit recorded; no execution; remaining coverage limits stated.

Snapshot: [48ac43987df9ac1ba40fef72dd7a2b4679c592b4](https://github.com/HorizonRobotics/RoboOrchardLab/tree/48ac43987df9ac1ba40fef72dd7a2b4679c592b4).

- [projects/monodream/README.md](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/projects/monodream/README.md#L1)
- [projects/monodream/analyze_results.py](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/projects/monodream/analyze_results.py#L41)
- [projects/monodream/infer.py](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/projects/monodream/infer.py#L65)
- [robo_orchard_lab/models/monodream/model.py](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/robo_orchard_lab/models/monodream/model.py#L97)
- [robo_orchard_lab/models/monodream/navigation_arch.py](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/robo_orchard_lab/models/monodream/navigation_arch.py#L380)
- [robo_orchard_lab/models/monodream/processor.py](https://github.com/HorizonRobotics/RoboOrchardLab/blob/48ac43987df9ac1ba40fef72dd7a2b4679c592b4/robo_orchard_lab/models/monodream/processor.py#L59)

## dreamwalker

The author repository was pinned and its README inspected; the snapshot contains a README rather than the model, losses, trainer, preprocessing or evaluator. Consequently the paper mechanism is reviewed, but its implementation is unavailable for this audit.

Repository status: No model source available or verified; access limitation closed explicitly.

Snapshot: [ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5](https://github.com/hanqingwangai/Dreamwalker/tree/ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5).

- [README.md](https://github.com/hanqingwangai/Dreamwalker/blob/ab7c9a5c9d07742e30e4cf99132cf1f61bb83de5/README.md#L1)

## worldvln

The official snapshot is a project website with a video-asset generation helper, not released world-action training/inference code. Its GRPO density, reward standardization, action decoder and benchmark protocol cannot be reconciled with implementation. No model-code inspection is claimed.

Repository status: No model source available or verified; access limitation closed explicitly.

Snapshot: [8912d62f363ef739ea78864b2e37cb630060792c](https://github.com/EmbodiedCity/WorldVLN/tree/8912d62f363ef739ea78864b2e37cb630060792c).


## cotvla

The official CoT-VLA project page and full paper were reviewed. No official model/training implementation link was verified in this search. It is not valid to treat a third-party reimplementation or related VILA repository as the paper implementation.

Repository status: No model source available or verified; access limitation closed explicitly.

## unemo

The full paper and AAAI publication source were reviewed; no verified official implementation was located. The provisional/final prediction feedback equations are therefore author-described, not code-confirmed.

Repository status: No model source available or verified; access limitation closed explicitly.

## sali

The full paper and its ablations were reviewed; no verified official model/training repository was located. Different headline and ablation configurations remain unresolved from source.

Repository status: No model source available or verified; access limitation closed explicitly.

## imaginenav

No verified official implementation of the original paper was located. The later ImagineNav++ project links a repository whose recorded README says code is coming; it is not evidence of an available original implementation.

Repository status: No model source available or verified; access limitation closed explicitly.

## lookbeforeleap

The original ECCV paper was reviewed; no verified official repository implementing its complete lookahead policy was located. Related later code must not be substituted without labeling it.

Repository status: No model source available or verified; access limitation closed explicitly.

## navforesee

The supplied CVPR paper and current preprint identity were checked; no official model/training repository was verified. Future-query attention and inverse-dynamics loss remain paper evidence.

Repository status: No model source available or verified; access limitation closed explicitly.
