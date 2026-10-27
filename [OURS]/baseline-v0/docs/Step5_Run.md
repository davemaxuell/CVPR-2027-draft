# Step 5A — matched DINO dynamics training

Run: `runs/step5_smolvlm2_refined512/`; [status](../runs/step5_smolvlm2_refined512/status.json),
[locked plan](../runs/step5_smolvlm2_refined512/plan.json).
The first pair follows the earlier sequence: DINO-WM transfer and DINOv2 with fresh dynamics.
The NWM/CDiT pair remains a later repeat. **This DINO pair completed on 26 October 2026 at
10:00 KST.** Feature extraction, both training passes, checkpoint selection and frozen-bundle
verification are complete. See [results and learning curves](../runs/step5_smolvlm2_refined512/RESULTS.md).

| Model | Selected update | Group-C macro MSE |
| --- | ---: | ---: |
| PointMaze transfer | 36,864 | 1.347414 |
| Fresh dynamics | 36,864 | 1.320767 |
| Copy current features | — | 4.611210 |

Both models beat the copy-current baseline in every dataset/action group. Fresh dynamics had
lower error in this single matched run. Each complete training pass took about 57.7 minutes;
the two models are both retained for downstream comparison.

The navigation policy is the separately versioned [Step 4 reference](Step4_Run.md): refined
SmolVLM2 checkpoint 512 with STOP logit bias −0.25. Its identity is
`729480bf5c8402b07cad653a7ef0f7124be212d50255077290316ba686777ca7`.
The deliberate selection despite the failed stopping safeguard and the preserved Step 2 control
remain part of the experiment. Step 5 does not update the policy.

## Released predictor and encoder

The author’s [Hugging Face mirror](https://huggingface.co/datasets/gaoyuezhou/dino_wm) supplied
the checkpoint archive after the OSF download endpoint returned HTTP 502. Its SHA-256 matches
the OSF release: `425b0d8c2a4194d3ec996b553575f69409aa5714176b063d8401789db9179482`.
[Transfer selection](../runs/step5_smolvlm2_refined512/assets/transfer_selection.json) records
the archive revision, individual checkpoint hashes, extracted predictor and compatibility checks.

PointMaze was selected before CE validation. Its three-frame, 196-patch, 404-channel predictor
loads strictly into the declared six-layer architecture. The navigation source task is closer
to this experiment than PushT; this is a selection rationale, not a measured transfer advantage.
Wall’s released checkpoint has one history frame and is incompatible with the fixed interface.
Only predictor tensors transfer. Both trials receive the same freshly initialized three-action,
10-dimensional CE embedding and ten zero proprio channels. Source action/proprio encoders,
image decoder and optimizer are excluded.

Both trials share frozen DINOv2-S/14 weights. All 175 native tensors exactly match the previously
pinned Hugging Face weights. The pinned native source retains position-interpolation offset 0.1;
the installed Hugging Face implementation instead uses size-based interpolation. See
[encoder equivalence](../runs/step5_smolvlm2_refined512/assets/encoder_weight_equivalence.json).

Preprocessing: RGB/255, tensor resize shortest side to 224 with antialiasing, center crop 224,
normalize mean/std 0.5, then resize to 196 with antialiasing. These operations run in CPU float32.
The native encoder runs in float32 without gradients. Final 196×384 features are stored as float16.
The runtime check measured float16 storage MSE of 2.62e−7 versus float32 features. Half-precision
encoder execution was excluded after producing substantially larger drift. These choices are
identical in both trials and recorded in the encoder/cache identity.

## Matched data and optimization

Both models train on all **2,422,469 matched group-A transitions**, in the same seeded shuffled
order, exactly once. No early DINO-only examples are added. At each eligible root there are at
least four actual observations; DINO receives the last three, their two intervening commands,
and the current movement command. Targets are the actual next raw visual patches. STOP is
excluded; failed policy visits and blocked movement remain in training.

The index maps the original Step 4 sample IDs to 875,174 byte-distinct RGB images across training
and validation. Every image is hash-checked when encoded. A single immutable feature cache serves
both trials. Its rows include group-C images only because the encoder is already frozen; no
weights or statistics are fitted to those images.

Validation fixes **3,072 group-C expert-path samples**, 512 per dataset/action combination,
cycled across the available scenes with seed 5. The same quality filter as policy validation is
used. These validation examples contain no collisions or blocked forward motion, so they cannot
establish failure-state prediction quality. Group-A training includes those cases.

| Setting | Both trials |
| --- | --- |
| Trainable components | Predictor and CE command embedding |
| Objective | Final-frame raw 196×384 visual-patch MSE; no command/proprio output loss |
| Updates | 37,852; one complete pass including the final short batch |
| Batch | 64 |
| Optimizer | AdamW, peak LR 5e−4, weight decay 0.01, betas (0.9, 0.999), epsilon 1e−8 |
| Schedule | 512-update linear warm-up, then cosine decay to 10% of peak |
| Gradient clipping | Global norm 1.0 |
| Precision | FP32 parameters, BF16 dynamics autocast |
| Reproducibility | Seed 5; same data order and dropout seed; deterministic CUDA algorithms |
| Selection | Lowest mean group-C MSE across six dataset/action groups; earliest-update tie break |
| Checkpoints | Initialization, every 4,096 updates, and final update |

The initial speed check measured 762 samples/second and about 6.5 GB peak allocated GPU memory
at batch 64, before input-loading/validation overhead and the deterministic restart setting.
This suggested roughly 53 minutes of model compute per trial. Actual throughput is logged in
each trial’s `train.jsonl`. This is a training measurement, not a complete navigation-agent memory claim.

## Verification and restart

[Runtime verification](../runs/step5_smolvlm2_refined512/check_runtime.json) passed native/SDPA
output and gradient comparisons, a block-causal future-view test, frozen encoder checks,
preprocessing comparisons and an actual training profile. Optimized attention preserves the
source frame mask and trainable tensor layout.

[Training verification](../runs/step5_smolvlm2_refined512/check_training.json) checked 272
training samples against the original transition catalog, all 3,072 validation group assignments,
and dataset/action balance. A saved/reloaded AdamW update reproduced the uninterrupted parameters
bit for bit. Checkpoint loading preserves CPU optimizer step counters; deterministic CUDA algorithms
and `CUBLAS_WORKSPACE_CONFIG=:4096:8` remove nondeterministic backward differences.

Resume with the `rolevln` environment: `python tools/run_step5.py`. The supervisor locks duplicate
jobs, pins source snapshots and plan/data identities, and restricts GPU work to physical GPU 0.
Completed checkpoints retain model, optimizer, RNG state and sample position. The two trials share
the exact update budget and feature cache.

Selected bundles are saved under [frozen/dinowm_transfer/](../runs/step5_smolvlm2_refined512/frozen/dinowm_transfer/config.json)
and [frozen/dinov2_fresh_dynamics/](../runs/step5_smolvlm2_refined512/frozen/dinov2_fresh_dynamics/config.json),
with [frozen/encoder/](../runs/step5_smolvlm2_refined512/frozen/encoder/encoder_identity.json) shared between them.
`wm.frozen.FrozenDinoWorldModel` checks the policy and encoder identities and rejects prediction
before four real observations. Command indices are 0=FORWARD, 1=LEFT, 2=RIGHT; STOP uses the current
features without a dynamics call.

[Final verification](../runs/step5_smolvlm2_refined512/final_verification.json) passed. Reloading
each packaged model reproduced its selected validation score exactly. All model and encoder
parameters are frozen. Re-encoding 64 source images with the packaged encoder differed from the
stored float16 features by MSE 2.55e−7. The selected policy identity, required STOP bias, original
Step 2 control and failed-safeguard selection record were verified unchanged. An additional
[inference-mode check](../runs/step5_smolvlm2_refined512/check_inference_modes.json) confirmed
identical predictions with the training evaluator and frozen loader settings.

The DINO pair is ready for Step 6 group-B scorer-label collection. Feature-prediction results
are not navigation success results; scorer labels and navigation evaluation come later.
