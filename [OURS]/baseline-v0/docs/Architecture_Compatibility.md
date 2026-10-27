# Architecture compatibility

23 October 2026. Scope: repair the agreed CE baseline before exploring new research ideas.

The current implementation fixes the identified tensor, representation, action, and runtime mismatches. It has executable CPU checks, including the actual pinned DINO predictor and NWM denoiser/sampler source. **It is not yet a complete, checkpoint-validated Habitat agent.** Trained navigation policies, selected world-model weights, native processor integration, and GPU measurements are still required.

## Corrected connections

| Connection | Problem found | Implemented correction |
| --- | --- | --- |
| DINO-WM → scorer | The old scorer assumed 256 raw patches and 64 pooled tokens. The inspected DINO-WM profile produces 196 raw patches. | Default 14 × 14 raw patches → 7 × 7 pooled patches → **49 × 384** scorer features. Grid-derived positions and sequence lengths replace fixed sizes. |
| DINO preprocessing | A generic 224-pixel DINO pipeline would differ from the transferred predictor's source convention. | Resize/crop 224, normalize mean/std 0.5, resize to 196; use the same frozen encoder for real and decoded future views. |
| DINO output | Native prediction can include time, action, and proprioception dimensions. | Explicit layout extraction takes the last predicted frame and only its visual features before pooling. |
| DINO CE actions | Released source tasks have different command/proprioception encoders. | New three-command CE embedding; zero proprio channels; compatible predictor-only transfer followed by CE adaptation. No simulator pose input. Fresh dynamics use the same interface. |
| DINO device handling | The inspected attention mask is constructed directly on CUDA and is not a buffer. | A hash-checked loader changes that one assignment to a nonpersistent registered buffer in memory. Source files and checkpoint keys are unchanged. |
| CDiT → VAE | The eight-channel denoiser output is epsilon plus learned variance, not a decoded future latent. | Run the matching official diffusion sampler to obtain a four-channel latent first. |
| VAE → DINO | VAE scaling, RGB range, and DINO preprocessing differ. | Encode with scale 0.18215; decode after division by that scale; clamp/convert pixels to [0,1]; use the common DINO encoder. |
| NWM conditioning | Context count, latent grid, command width, and timestep conventions must agree. | Validate four real context frames, 4 × 28 × 28 latents, three command coordinates, and explicit relative time for the inspected 224-pixel CDiT-S/2 profile. |
| Policy → scorer | The policies have different text widths; padding and generated-token states can select the wrong context. | Validate widths 960/896/2048; capture one final-norm prefill; select the last valid expanded-sequence position; pack full instruction features with a mask. |
| Precision | BF16/FP16 policy/WM features can enter FP32 scorer layers. NWM's timestep embedding also produces FP32. | Convert cached features at the scorer boundary; use autocast around a reduced-precision CDiT forward. |
| Episode history | Branch scoring, duplicate observations, or an old action chunk can corrupt real history. | Read-only candidate scoring; one real observation update and executed-action commit per step; per-row training resets; STOP requires episode reset. Native policy prompts must be rebuilt from actual history. |
| Action selection | A null margin or prediction failure previously lacked executable fallback. | Null margin skips prediction and keeps the base action. Invalid predictions and nonfinite scores produce logged fallback. Ties preserve the base action. |
| Action IDs | Internal F/L/R/STOP IDs need not equal Habitat's IDs. | Explicit bijective `ActionMap`, populated from the live task configuration. |
| Configuration and gradients | Architecture/loss settings were descriptive; unconditional detachment also blocked a future differentiable critic interface. | `NavigationScorer.from_config` consumes dimensions/grid/loss settings. Default external features stay detached; an explicit frozen-critic mode permits future-feature gradients only. |

The scorer's parameter counts remain 2,890,115 / 2,873,603 / 3,170,819 for SmolVLM2 / OneVision / Qwen3-VL. Patch positions are buffers, so the grid correction changes token cost without changing those counts. The tested default DINO predictor has 20,122,600 parameters, plus 30 in the new CE command embedding. The tested CDiT-S/2 has 49,227,680 parameters. These are module counts, not complete-system memory estimates.

## Executable pipeline boundaries

```text
Real RGB and actual action history
  ├─ native frozen navigation policy → instruction states, q_t, base action
  └─ shared frozen DINO encoding → current patches

DINO branch:
  real raw DINO history + past commands + candidate F/L/R
    → CE command adapter + predictor
    → last predicted visual frame → spatial pooling → [B,49,384]

NWM/CDiT branch:
  four real views → frozen VAE encode × 0.18215
    + normalized nominal command + relative time
    → CDiT epsilon/variance → official diffusion sampler
    → completed latent [B,4,28,28]
    → VAE decode / 0.18215 → RGB → same DINO/pooling → [B,49,384]

Three movement predictions + current-view STOP
  → scorer [B,4] → validated override or base action
  → map to live task ID → execute once → record actual action and observation
```

The runtime callback receives `[B,3]` F/L/R IDs and must return `[B,3,49,384]` movement predictions; the implemented runtime currently runs one episode at a time (`B=1`). The scorer itself supports batched roots. World-model history is read-only during this callback. No STOP prediction is generated. The callback or task adapter must reject insufficient real context; NWM's shape validator already rejects fewer than four frames. Use the same warm-up rule in matched world-model comparisons.

The DINO adapter accepts one to three real frames; train CE on the same available-history pattern. The NWM command table and sampler budget remain explicitly unset in the configuration. Populate commands from the chosen task's nominal step distance and turn angle, with a documented local coordinate convention, units, signs, and normalization. A forward collision changes the observed target, not the commanded input. Do not substitute realized displacement or future pose.

The shared resize/crop convention is a deliberate CE adaptation for nonsquare images: NWM's source preprocessing instead resizes directly to a square. Keep the CE convention fixed across training and inference and across transfer/fresh pairs. The source predictor's exact checkpoint metadata can override a proposed profile only through a separately declared compatible configuration, never silent reshaping.

## What the checks establish

Run from `Scorer` in an environment with compatible PyTorch/torchvision, NumPy, einops, and timm:

```powershell
python -B check_design.py
python -B check_compatibility.py
```

The local execution uses Python 3.13.7 and PyTorch 2.9.0+cpu. Additional test dependencies `einops==0.8.1` and `timm==0.9.16` were installed under `Scorer/.test_dependencies`; the test runner adds that directory locally. They do not alter the shared torch/CUDA installation. The tests use the pinned source snapshots in the sibling research investigation folder; they do not download weights. The project uses a single process for deterministic NWM sampling because the upstream sampler uses global RNG; its RNG state is restored afterward.

| Evidence | Executed coverage | Does not establish |
| --- | --- | --- |
| [13 original check groups](../Scorer/check_results.json) | Scorer dimensions, parameter counts, gradients, candidate independence, masking, memory, STOP, selection, pooling, and utility bounds | Navigation performance |
| [10 compatibility check groups](../Scorer/compatibility_results.json) | All three policy widths × two explicit grids, configurable architecture/loss, padding, mixed feature precision, resets, gradient routes, selection fallback, feature hooks, action mapping, and lifecycle | Actual pretrained VLM/processor integration |
| Actual DINO-WM predictor source, random weights | Native six-layer CPU forward/backward; CE command gradients; source-mask repair; strict state loading; predictor → pooling → scorer/runtime | Released source-weight transfer or useful CE predictions |
| Actual NWM CDiT-S/2 and diffusion source, random weights | Native 224-pixel profile; eight-channel forward; four-channel completed sampling; repeatable seed; CPU BF16 forward | Quality, speed, or stability of a trained checkpoint |
| Explicit VAE/DINO test doubles | Pixel range, scaling, grid checks, frozen parameters, optional input gradients, and rejection of raw denoiser output | Actual pretrained VAE reconstruction or DINO feature quality |
| Saved official VLM metadata | Selected model family, revision, and text width | Running the policy checkpoints |

The diffusion test uses **two sampling steps solely to exercise the interface**. It is not a proposed production setting or an accuracy result. Source/config hashes and exact outcomes are recorded in the linked JSON files. The earlier independent review remains historical evidence; its files were not rewritten to hide the issues it identified.

## Remaining integration requirements

1. **Select actual weights.** Check the DINO predictor's grid/history/conditioning metadata and encoder revision; strictly load it. NWM access and exact checkpoint are still pending. Verify VAE revision and the checkpoint's diffusion objective/schedule. Keep transfer/fresh pairs matched.
2. **Wire the three native policy implementations.** Validate their processor/tokenizer, final decoder norm, expanded attention masks, prompt limits, and action parser. The locked OneVision model is native `LlavaQwenForCausalLM`; it is not automatically interchangeable with a stock HF OneVision class. Initialize each actor from its own VLM, not StreamVLN weights of a different architecture.
3. **Replay actual history first.** Rebuild the bounded prompt after a forced/reranked action and discard unexecuted chunks. Match that window convention in policy training. Optimize KV reuse only after feature and action outputs agree with fresh replay. A hook test cannot validate a backbone's cache semantics.
4. **Connect Habitat.** Read action IDs, step size, turn angle, camera configuration, and episode budget from the live CE task. Test left/right signs, blocked forward motion, STOP, reset, and branch replay including metric prefix. Habitat and CE assets are not installed in this environment.
5. **Run a real-checkpoint episode and profile.** Measure the complete actor + predictor + VAE/DINO + scorer, including candidates and diffusion steps. There is no CUDA device in this execution environment and no demonstrated 8 GB fit. Overrides remain disabled until CE development evidence supplies a margin.

Configuration fields for optimizer, data collection, and comparison protocols describe the planned training; no optimizer or CE training loop is implemented here. Cache files must be keyed by actual policy/encoder/WM checkpoint identity, preprocessing/grid, history/action convention, and feature-extraction version. Cached trainable memory is never a substitute for replay under current scorer weights.

These are concrete prerequisites for an end-to-end run. They do not change the locked three policies, four world-model families, one-action horizon, or current scorer target. No new novelty claim is introduced by these compatibility repairs.

## Source references

- DINO-WM commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`: [ViTPredictor and attention mask](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/vit.py), [visual model/grid/conditioning](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/models/visual_world_model.py), [preprocessing](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/datasets/img_transforms.py), [training configuration](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/conf/train.yaml).
- NWM commit `3f6cd8e70d6f2d1e2b9684acff510710135f0f41`: [CDiT definitions](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/models.py), [VAE and sampling flow](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/isolated_nwm_infer.py), [official diffusion factory](https://github.com/facebookresearch/nwm/blob/3f6cd8e70d6f2d1e2b9684acff510710135f0f41/diffusion/__init__.py).
- Exact policy metadata: [saved official configurations and revision identifiers](../manifest/source_pins.json), [locked trials](../manifest/Backbone_Trials.json).
- Files implementing these boundaries: [interfaces.py](../Scorer/interfaces.py), [world_models.py](../Scorer/world_models.py), [scorer.py](../Scorer/scorer.py), [runtime.py](../Scorer/runtime.py), [config.json](../Scorer/config.json).
