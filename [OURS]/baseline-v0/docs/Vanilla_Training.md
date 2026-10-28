# Vanilla Training Formulation

23 October 2026 · Training specification · SmolVLM2 execution: [Step 2 complete](../runs/step2_smolvlm2/RESULTS.md), [Step 3 complete](../runs/step3_smolvlm2/RESULTS.md), [selected refinement and Step 4 complete](Step4_Run.md), [Step 5A DINO pair complete](Step5_Run.md), [Step 6 labels complete](Step6_Run.md), [Step 7 frozen features complete](Step7_Run.md). [Steps 8–9](Step9_Run.md) are complete with conservative base-behavior settings; [Step 10](Step10_Run.md) is complete, with 21/224 successes and zero overrides for all five systems. The [NWM/CDiT repeat](Step5B_Run.md) has started automatically.

**Goal:** establish our existing navigation-policy → world-model → scorer pipeline using ordinary supervised objectives. Train the components in sequence, freeze the upstream components, then connect them for one-action navigation. This document defines the baseline before adding any research contribution.

The three policy trials and four world-model trials remain as recorded in [Backbone Trials](../manifest/Backbone_Trials.json) and [World Model Trials](../manifest/World_Model_Trials.json). Shapes and interfaces follow [Architecture Compatibility](Architecture_Compatibility.md); the scorer follows [Scorer Design](Scorer_Design.md).

Start the implementation with **SmolVLM2-500M and the DINO-feature branch**, then extend the same procedure to the other agreed trials. This is an implementation order, not a claim that this pairing is the best model or already fits 8 GB. A transfer run requires a compatible released checkpoint; otherwise record the run as the fresh-dynamics baseline.

## Training order

| Step | Work | What learns | Output |
| --- | --- | --- | --- |
| 1 | Fix datasets, action definitions, and run settings | Nothing | Experiment manifest |
| 2 | Expert navigation supervision | Navigation-policy adapters | Initial navigation policy |
| 3 | DAgger collection and supervised refinement | Same policy adapters | Frozen reference policy |
| 4 | Collect actual next-state transitions | Nothing | World-model training data |
| 5 | Train a world model on those transitions | Dynamics and command conditioning | Frozen CE world model |
| 6 | Measure alternative-action outcomes | Nothing | Scorer labels |
| 7 | Extract real and predicted features | Nothing | Frozen-feature dataset |
| 8 | Train the scorer | Projections, memory, attention, and heads | Frozen scorer |
| 9 | Select the override margin on development scenes | No gradient updates | Deployment settings |
| 10 | Run the full pipeline and controls | Nothing | Baseline results |

At deployment, use one selected policy, one selected world model, and its associated scorer. The trials are separate systems.

## Notation

| Symbol | Meaning |
| --- | --- |
| $I$ | Full navigation instruction |
| $o_t$ | Actual RGB observation before action $t$ |
| $h_t=(o_0,a_0,\ldots,a_{t-1},o_t)$ | Actual observation/action history |
| $\mathcal A=\{F,L,R,\mathrm{STOP}\}$ | Four canonical CE actions |
| $\pi_\psi$ | Navigation policy; $\psi$ denotes its trainable parameters during policy fitting |
| $\pi_{\mathrm{ref}}$ | Final trained policy, frozen after Step 3 |
| $E_D,\ E_V,\ D_V$ | Frozen DINO encoder, VAE encoder, and VAE decoder |
| $F_\phi$ | Action-conditioned dynamics model |
| $S_\theta$ | Scorer, including its real-history memory update |
| $Z_t\in\mathbb R^{196\times384}$ | Raw DINO patch features |
| $V_t\in\mathbb R^{49\times384}$ | Spatially pooled DINO features used by the scorer |
| $M_t\in\mathbb R^{8\times256}$ | Scorer memory after observing the real prefix |
| $B_{\max}$ | Configured maximum number of actions in an episode |
| $\operatorname{sg}(\cdot)$ | Stop-gradient: treated as a fixed target/input |

Here **CE means continuous environments**. The policy's cross-entropy loss is written explicitly to avoid confusing the two uses of “CE.”

## 1. Fix the experiment before collecting data

Use **R2R-CE and RxR-CE** for our added training and evaluation. Record dataset versions, language subsets, scene IDs, camera settings, action sizes, collision/sliding settings, and episode budgets. External pretraining in the selected VLMs, DINOv2, VAE, and transferred dynamics remains disclosed.

Use the existing pilot split:

| Scene group | Permitted use |
| --- | --- |
| A: model fitting | Policy supervision, DAgger, and world-model fitting |
| B: scorer fitting | Collect branch labels and optimize the scorer |
| C: development | Select checkpoints, settings, and the override margin |
| E: final evaluation | Report results after choices are fixed |

Construct A/B/C from official training scenes. Group shared R2R/RxR scene IDs together, so the same environment cannot cross these groups. Keep final evaluation scenes separate. Development results are tuning results. This pilot uses less training data than a full-data benchmark protocol; label it accordingly.

For the first vanilla implementation, use **one primitive action per policy response and one action per environment step**. This makes policy supervision, DAgger, branch continuation, and deployment share the same schedule. It is an explicit implementation choice for our StreamVLN-style adaptation; the original StreamVLN scripts use four future steps. An original-chunk-schedule comparison can be recorded separately. [StreamVLN training script](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/scripts/streamvln_train_slurm.sh)

Map internal F/L/R/STOP IDs to the live task IDs. Read forward distance and turn angle from that task. World-model transitions span **one executed CE command**; do not inherit a source dataset's frame skip as five CE actions.

**Output:** a manifest containing scene partitions, model revisions, action mapping, image transforms, policy context window, response format, seeds, and training budgets. Values that still require profiling are listed at the end.

## 2. Train the navigation policy on expert trajectories

**Purpose:** teach the selected lightweight VLM to produce legal navigation actions from an instruction and observed history.

Initialize one of the three locked VLMs using its own pretrained weights and native processor/tokenizer. The initial lightweight recipe trains language-decoder LoRA adapters while freezing the pretrained language weights, policy vision encoder, and pretrained multimodal projector. Treat projector unfreezing as a separately recorded recipe if later needed. LoRA target names must match each native implementation.

For an expert trajectory, build a prompt

$$
x_t=\operatorname{Render}_i(I,h_t;W_i),
$$

where $i$ identifies the backbone and $W_i$ is its recorded history/window budget. Include the full instruction and only observations/actions available by time $t$. The target $y_t$ is that backbone's tokenization of the expert's next action and response terminator.

Train with masked autoregressive cross-entropy:

$$
\mathcal L_{\mathrm{policy}}(\psi)
=-\frac{\sum_{t,j}m_{t,j}
  \log p_\psi(y_{t,j}\mid x_t,y_{t,<j})}
 {\sum_{t,j}m_{t,j}},
$$

where $m_{t,j}=1$ only for target response tokens. Instruction, image, padding, and already observed history tokens receive no target loss. An action name can occupy multiple tokenizer tokens; it need not be a dedicated single token.

Use the same prompt/window conventions at training and deployment. Preserve complete instructions within supported limits and record any excluded overlength cases. The native policy image processor is separate from the DINO/VAE world-model preprocessing.

**Train:** policy LoRA adapters.  
**Freeze:** pretrained policy components outside those adapters. The world model and scorer do not participate.  
**Output:** an SFT navigation checkpoint with its processor, adapter configuration, prompt format, and action parser.

This uses StreamVLN's supervised navigation training pattern with a different backbone, dataset scope, and trainable parameter set. It does not reproduce the original full-module, multi-dataset training recipe. [StreamVLN first-stage script](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/scripts/streamvln_train_slurm.sh)

## 3. Refine the policy with DAgger, then freeze it

**Purpose:** teach recovery at states the learned policy actually visits.

1. Run the SFT policy on group-A episodes using one-action replanning.
2. At each visited state, query the configured CE expert for the corrective action.
3. Save the actual prefix and that expert label; record the expert/recovery rule and its reference-path progress handling.
4. Aggregate these examples with the original expert examples.
5. Continue supervised optimization with the same masked action-token loss.

$$
\mathcal D_{\mathrm{policy}}
=\mathcal D_{\mathrm{expert}}\cup\mathcal D_{\mathrm{DAgger}},
\qquad
\psi^\star\leftarrow
\operatorname{Fit}(\mathcal L_{\mathrm{policy}},\mathcal D_{\mathrm{policy}}).
$$

Start with one collection/refinement round, then record any additional rounds as additional rollout/expert-query budget. The expert may use privileged state for training labels. The deployed policy receives no reference path or goal coordinates.

**Keep `expert_action` and `executed_action` separate.** A corrective label is not necessarily the action that produced the next image. Step 5 must use executed actions for dynamics targets.

Select the policy checkpoint using group C, then freeze it:

$$
\pi_{\mathrm{ref}}=\pi_{\psi^\star}.
$$

Freeze its weights, LoRA adapters, decoding rule, prompt/window convention, and action mapping. Its checkpoint and inference schedule define all later continuation labels. Replacing it requires versioning/recollecting those labels and regenerating policy features.

**Output:** a frozen reference policy and a policy-only navigation baseline. The original StreamVLN second stage also uses DAgger data; our added data remains within the CE restriction. [StreamVLN second-stage script](https://github.com/InternRobotics/StreamVLN/blob/e48f6ff7e9201d93aae003e8f64b04c00cec13bc/scripts/streamvln_stage_two_train_slurm.sh)

## 4. Collect world-model transition data

**Purpose:** obtain real examples of what each movement command does.

Use group A only. Combine legal expert trajectories, recorded DAgger transitions, and frozen-policy visits. At selected actual roots, restore the same root and execute F, L, and R separately to improve command coverage. For dynamics fitting, each branch needs its first actual next observation; a complete continuation is unnecessary.

Store

$$
\mathcal D_{\mathrm{WM}}
=\{(o_{t-m+1:t},a^{\mathrm{exec}}_{t-m+1:t-1},
      a^{\mathrm{exec}}_t,o_{t+1})\}.
$$

Keep episode/scene identity, frame ordering, command duration, and preprocessing metadata. Retain blocked-motion examples with their actual observations. A commanded forward action can produce little or no movement.

Use the same transition corpus for the paired transfer/fresh world-model trials. STOP is an identity feature input to the scorer and is excluded from dynamics fitting.

For the cross-family comparison, use the same eligible root/action IDs with at least four real observations: DINO takes the last three and CDiT takes four. If a DINO-only run also trains on earlier one/two-frame roots, record those extra examples separately from the matched comparison.

**Output:** an actual-transition dataset with no imagined targets. Branch outcomes from group B must not be reused to fit the world model in this split.

## 5. Train the world model with ordinary prediction losses

The policy remains frozen. Train each world-model trial separately.

### 5A. DINO-WM transfer and DINOv2 with fresh dynamics

Freeze one shared DINOv2-S/14 encoder. Apply the corrected preprocessing: resize/crop 224, mean/std 0.5, then resize to 196 for the encoder.

$$
Z_t=E_D(o_t)\in\mathbb R^{196\times384}.
$$

Use up to three actual frames, the intervening executed commands, and the candidate/current training command:

$$
\widehat Z_{t+1}
=F_\phi(Z_{t-m+1:t},
        a^{\mathrm{exec}}_{t-m+1:t-1},a_t),\qquad 1\le m\le3.
$$

The implemented adapter has a learned 10-dimensional F/L/R embedding and 10 zero-valued proprio channels. The predictor sees 404 channels per patch. Its last-frame visual output has 384 channels. Train on the same short-history pattern that inference accepts.

Use ordinary feature prediction:

$$
\mathcal L_{\mathrm{DINO}}(\phi)
=\mathbb E_{\mathcal D_{\mathrm{WM}}}
  \left[
  \frac{1}{196\cdot384}
  \left\|\widehat Z_{t+1}
       -\operatorname{sg}\!\left(E_D(o_{t+1})\right)\right\|_F^2
  \right].
$$

Apply the loss to **raw visual patches before pooling**. The CE adapter supervises the final predicted visual frame; its placeholder proprio channels and command output channels are not targets. No DINO image decoder is required for this feature-only pipeline. This is our ordinary CE adaptation objective, not a claim to reproduce every source-task loss in DINO-WM. [DINO-WM prediction and loss code](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py)

| Trial | Predictor initialization | Trainable components |
| --- | --- | --- |
| DINO-WM transfer | Strictly compatible released predictor weights | Predictor and new CE command embedding |
| DINOv2 with new dynamics | Random predictor weights, same architecture | Same components |

The source command/proprio encoders are replaced in both cases. Match grid, history, encoder, trainable modules, transitions, and update budget when comparing initialization. Select a checkpoint on group-C feature prediction error and command-specific diagnostics, then freeze it.

For the scorer, pool each 2 × 2 raw patch region:

$$
\widehat V_{t+1}=\operatorname{Pool}_{2\times2}(\widehat Z_{t+1})
\in\mathbb R^{49\times384}.
$$

**Output:** frozen DINO dynamics, a frozen CE command embedding, and an exact encoder/preprocessing identity.

### 5B. NWM transfer and CDiT with fresh dynamics

Freeze the shared pretrained VAE. With the declared VAE posterior-sampling convention, encode each actual image as

$$
z_t=0.18215\,E_V(o_t)\in\mathbb R^{4\times28\times28}.
$$

Here $E_V(o_t)$ denotes a sampled latent, not the posterior distribution object. Record posterior sampling/caching seeds. Both trials use the same VAE and CE resize/crop convention.

Condition CDiT on four actual latent frames, a nominal command $u(a_t)\in\mathbb R^3$, and the declared relative time:

$$
c_t=(z_{t-3:t},u(a_t),\Delta t),\qquad \Delta t=1/128.
$$

One time unit denotes one CE command in this adaptation. The command table expresses nominal local x/y/yaw with recorded units, signs, and normalization. It does not contain the realized future displacement or collision.

For diffusion index $\ell$ and Gaussian noise $\epsilon$, form

$$
z^{(\ell)}_{t+1}
=\sqrt{\bar\alpha_\ell}\,z_{t+1}
 +\sqrt{1-\bar\alpha_\ell}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,\mathbf I).
$$

CDiT returns noise and variance outputs:

$$
(\widehat\epsilon_\phi,\widehat v_\phi)
=F_\phi(z^{(\ell)}_{t+1},\ell,c_t).
$$

For the inspected epsilon/learned-variance configuration, use the official training loss:

$$
\mathcal L_{\mathrm{CDiT}}
=\mathbb E\left[
 \operatorname{MSE}(\epsilon,\widehat\epsilon_\phi)
 +\mathcal L_{\mathrm{VB}}
   \bigl(\operatorname{sg}(\widehat\epsilon_\phi),\widehat v_\phi\bigr)
\right].
$$

The VB term fits variance using the source's variational-bound calculation; its noise-output input is detached. Preserve the source's normalization and variance parameterization rather than adding an invented coefficient. Training uses the inspected 1,000-step diffusion schedule; diffusion index $\ell$ and navigation time $\Delta t$ are different variables. [NWM diffusion factory](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/diffusion/__init__.py), [training loss implementation](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/diffusion/gaussian_diffusion.py#L719-L797)

Transfer initializes compatible CDiT weights from NWM; the fresh trial initializes the same-sized CDiT randomly. Train the dynamics and its command/time-conditioning modules; freeze the VAE. Keep the same training data, size, action convention, and update budget. Select on group-C diffusion/prediction diagnostics, then freeze.

At feature extraction time, a declared inference sampling budget produces a completed four-channel latent:

$$
\widehat z_{t+1}
\xrightarrow{\ D_V(\widehat z_{t+1}/0.18215)\ }
\widehat o_{t+1}
\xrightarrow{\ \text{RGB conversion},\,E_D,\,\operatorname{Pool}\ }
\widehat V_{t+1}\in\mathbb R^{49\times384}.
$$

The eight-channel denoiser output cannot enter the VAE directly. Choose and record inference denoising steps separately from training's diffusion schedule. The existing two-step CPU smoke test is not a production sampling choice. [NWM sampling and decoding](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/isolated_nwm_infer.py)

**Output:** a frozen CE CDiT, frozen VAE, command table, diffusion settings, and common DINO adapter.

## 6. Collect the scorer's usefulness labels

**Purpose:** measure whether an action helps finish the instruction under a fixed continuation policy.

Use group-B roots for fitting and separate group-C roots for development. Begin with the existing **64-root, at-least-eight-episode pilot** to debug collection; it is not a sufficient final training set.

For every root $(I,h_t)$:

1. Restore identical simulator state, actual history, metric prefix, remaining budget, and random-state protocol.
2. Execute one candidate $a\in\mathcal A$.
3. For F/L/R, continue with frozen $\pi_{\mathrm{ref}}$ on real observations until STOP or the original episode limit.
4. For STOP, terminate immediately.
5. Record the actual first-next image, immediate forward collision, whole-path metrics, and remaining costs.

The scorer and world model do not select continuation actions during this collection. After the forced first action, rebuild the policy prompt using what actually happened.

Define the measured target:

$$
U_t^a
=0.70\,S_t^a+0.30\,N_t^a
-0.01\,\frac{n_t^a}{B_{\max}}
-0.02\,\frac{c_t^a}{B_{\max}}.
$$

| Label | Definition |
| --- | --- |
| $S_t^a$ | Official terminal success under the configured CE evaluator |
| $N_t^a$ | Official nDTW of the complete actual trajectory, including the root's earlier prefix |
| $n_t^a$ | Executed actions from the root through termination, including the candidate and any executed STOP |
| $c_t^a$ | Forward collision events over that same remaining trajectory |
| $C_{t,\mathrm{imm}}^a$ | Collision on the candidate action itself; used only for the FORWARD auxiliary loss |

For STOP, use the current view as its feature input, no continuation, $n=1$, and $c=0$. Let the actual evaluator determine success/nDTW. Every branch retains the original budget; it does not receive a fresh episode.

Conceptually, the scorer estimates

$$
Q_{\mathrm{ref}}(I,h_t,a)
=\mathbb E[U(\tau)\mid h_t,\ a,\ \text{then }\pi_{\mathrm{ref}}].
$$

A single measured branch is one outcome sample when the continuation is stochastic. Use the configured CE evaluator for metrics and [labels.py](../Scorer/labels.py) for the current utility arithmetic. Reference paths/poses are label-construction resources, never deployed model inputs.

**Output:** versioned branch labels tied to the frozen policy, action schedule, scene partition, and evaluator configuration.

## 7. Build the scorer's frozen-feature dataset

At each real root, extract:

| Feature | Shape | Source |
| --- | --- | --- |
| Instruction $T_I$ | $L\times d_i$ | Instruction-only pass through frozen trained policy; cache once per episode |
| Context $q_t$ | $d_i$ | Last valid hidden state of the actual multimodal prefill, before response generation |
| Current view $V_t$ | $49\times384$ | Shared frozen DINO encoder and pooling |
| Previous action / elapsed budget | ID and $t/B_{\max}$ | Actual execution log |
| Candidate future $\widehat V_{t+1}^a$ | $49\times384$ | Frozen world model; current-view identity for STOP |

The policy widths $d_i$ are 960, 896, and 2048 for SmolVLM2, OneVision, and Qwen3-VL. Use the native tokenizer/processor and an expanded-sequence mask when extracting $q_t$.

Generate one recorded future sample for each F/L/R candidate. Also encode each actual branch's first-next observation for the diagnostic/warm-up stage. Actual and predicted features share the same measured outcome label.

For a four-world-model comparison, sample predictions equally from the available frozen models and fit **one scorer per navigation policy**. Do not provide a world-model identity input. Once trained, reuse that exact scorer across those world-model trials. If initially only one world model is available, label the initial run accordingly; adding models requires a newly versioned pooled scorer and rerunning comparisons.

Cache frozen inputs and checkpoint/preprocessing identities. Recompute trainable scorer memory from actual prefixes during training. Reject roots with insufficient world-model context instead of silently padding. For matched comparisons involving NWM, begin reranking only after four actual observations for every compared system.

**Output:** root/action feature records, labels, and the actual prefix needed for memory replay.

## 8. Train the small scorer

**Train:** policy/visual projections, action/type embeddings, memory attention and GRU, candidate attention, and the three output heads.  
**Freeze:** navigation policy, DINO, VAE, world model, and all extracted external features.

First reconstruct real memory:

$$
M_t=G_\theta(M_{t-1},T_I,V_t,q_t,a_{t-1},t/B_{\max}).
$$

Then evaluate each candidate independently:

$$
(\widehat U_t^a,s_t^a,c_t^a)
=S_\theta(T_I,V_t,q_t,M_t,a,\operatorname{sg}(\widehat V_{t+1}^a)).
$$

Here $s_t^a$ and $c_t^a$ are success and immediate-collision logits. Only $\widehat U_t^a$ selects actions. Keep the existing width 256, eight memory slots, two attention blocks, four heads, and 128-wide readout.

For $R$ roots, each with all four actions, use the implemented loss:

$$
\begin{aligned}
\mathcal L_{\mathrm{scorer}}(\theta)
={}&\frac{1}{4R}\sum_{r=1}^{R}\sum_{a\in\mathcal A}
\left[
  (\widehat U_r^a-U_r^a)^2
  +0.1\,\operatorname{BCEWithLogits}(s_r^a,S_r^a)
\right]\\
&+\frac{0.1}{R}\sum_{r=1}^{R}
\operatorname{BCEWithLogits}(c_r^F,C_{r,\mathrm{imm}}^F).
\end{aligned}
$$

The forward-only collision term has its own normalization, matching [scorer.py](../Scorer/scorer.py). Train direct outcome regression; the labels come from Step 6.

Use the already proposed simple training sequence:

1. **Actual-next-view stage:** fit/debug with observed next features to check that labels and scorer capacity support meaningful ranking.
2. **Mixed stage:** use actual or predicted movement features with the same labels. A 50/50 mix is an initial recipe, not a validated optimum.
3. **Predicted-only stage:** finish fitting and select the deployed checkpoint using predicted features. STOP remains identity in all stages.

Record the update budget of each stage and match it in comparisons. Actual-next-view evaluation is privileged and must be reported separately from deployable results.

Replay each root's actual prefix under current scorer weights. Process the earlier prefix without gradients, detach at the boundary, and backpropagate through at most the last **32 real observation updates**. Reset memory at episode boundaries; imagined candidates never update it. Do not reuse cached memory produced by older scorer weights.

Starting scorer settings from [config.json](../Scorer/config.json):

| Setting | Initial value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate / weight decay | $10^{-4}$ / 0.01 |
| Effective batch | Eight roots, all four candidates per root |
| Gradient clipping | Global norm 1.0 |
| Recurrent gradient window | At most 32 real updates |
| External-feature gradients | Detached; `future_grad=False` |

Use accumulation if eight roots do not fit. These are proposed training settings, not measured CE results. Select using group-C utility regret and closed-loop navigation checks, with prediction-only inputs for the deployable checkpoint.

**Output:** one frozen scorer checkpoint per reference policy and declared set of world-model predictions. There is no scorer gradient back into the world model or policy in this vanilla training procedure.

## 9. Calibrate action selection, then freeze all settings

Let $a_{\mathrm{base}}$ be the frozen policy's preferred action and $a^\star$ the scorer's highest-utility action. On group C, select a finite nonnegative margin $\delta$:

$$
a_t=
\begin{cases}
a^\star,&\widehat U_t^{a^\star}
          -\widehat U_t^{a_{\mathrm{base}}}>\delta,\\
a_{\mathrm{base}},&\text{otherwise}.
\end{cases}
$$

Use both measured branch regret and real closed-loop episodes. A margin is an empirical choice, not a guarantee of improvement. Until selected, keep `override_margin=null`: prediction is skipped and the base action is executed.

Keep the base action on a tie, prediction failure, nonfinite scores, or insufficient real context. Record fallback events. Lock the chosen margin, diffusion steps, seeds, warm-up rule, and precision before final evaluation.

**Output:** frozen model identities and a complete deployment configuration.

## 10. Run the vanilla pipeline and its controls

At every real step:

```text
read actual RGB and executed-action history
obtain base action and real-prefill context from the frozen policy
encode the current image and update scorer memory once

if overrides are enabled and sufficient real context exists:
    predict one next state for FORWARD, LEFT, RIGHT
    convert predictions to the common 49 x 384 feature representation
    use the current view for STOP
    score all four actions and apply the calibrated override rule
else:
    keep the base action

map the selected action to the live CE task and execute once
record only that executed action and the actual next observation
rebuild the policy's bounded real-history prompt
terminate/reset when the task ends
```

Start with these controls on identical episodes and action schedules:

| System | Purpose |
| --- | --- |
| Frozen policy alone | Establish the one-action navigation baseline |
| Same-history, same-capacity scorer with a learned null future | Check what direct action/history scoring achieves without prediction |
| Scorer supplied actual branch-next views | Privileged diagnostic separating prediction error from scoring error |
| Policy + frozen world model + scorer | Measure the deployable vanilla pipeline |

Train the null-future control with the same branch labels, prefix information, capacity, and update budget; removing futures only at test time would be an input-corruption diagnostic. Compare transfer/fresh pairs with matched architecture and data. Hold exact scorer weights fixed across a policy's world-model comparisons.

Report SR, SPL, navigation error, nDTW/SDTW where supported, collision and action costs, branch action regret, fallback frequency, and complete pipeline latency/peak memory. Resample episodes/scenes for uncertainty estimates rather than treating correlated branches as independent episodes.

**Completion criterion:** the baseline trains, runs complete CE episodes with consistent histories/interfaces, and produces reproducible controlled results. An operational pipeline does not by itself establish that predicted futures improve navigation.

## Settings to record before an actual run

| Fixed in this formulation | Still requires selection/profiling |
| --- | --- |
| SFT → DAgger → freeze policy → train/freeze WM → train/freeze scorer | Exact policy LoRA targets/rank, learning rate, visual/history budget, and microbatch |
| One action response/execution; one-step world-model target | Compatible source checkpoint files and frozen encoder/VAE revisions |
| Ordinary DINO feature MSE or native CDiT diffusion loss | World-model optimizer settings, epochs/update counts, and precision |
| Existing utility weights and scorer heads/loss | CE command normalization and production diffusion sampling steps |
| Same real/predicted feature convention | Final scene partitions, collection counts, and expert recovery configuration |
| CE-only added data and held-out final evaluation | Training-stage durations, development-selected checkpoints, and override margin |

Profile each stage separately. Cached frozen features can avoid keeping all large models resident during scorer fitting; full inference still requires measuring all components and transfers. Neither successful CPU checks nor these equations establish an 8 GB training/inference fit.

For each run, save the manifest, policy/WM/scorer checkpoints, scene and collection identifiers, feature-cache identities, optimizer settings, selected margin, metric outputs, and resource measurements together.

## Current implementation status

The [scorer](../Scorer/scorer.py), [label helper](../Scorer/labels.py), [interfaces](../Scorer/interfaces.py), [world-model adapters](../Scorer/world_models.py), and [runtime](../Scorer/runtime.py) already have CPU interface checks. The executed [design checks](../Scorer/check_results.json) and [compatibility checks](../Scorer/compatibility_results.json) are not navigation training results.

Native policy training adapters, CE collection/replay, complete training loops, selected pretrained world-model/encoder integration, and GPU profiling remain to be implemented/validated. This document is the training specification for that work. It introduces no new research objective or novelty claim.
