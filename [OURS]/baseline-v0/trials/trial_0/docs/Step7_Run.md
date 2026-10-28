# Step 7 — frozen scorer features

Run: `runs/step7_smolvlm2_refined512_dino_pair_v1/`.
Status: complete, 27 October 2026 at 05:44 KST (26 October 20:44 UTC).
[Results](../runs/step7_smolvlm2_refined512_dino_pair_v1/RESULTS.md) and the
[full audit](../runs/step7_smolvlm2_refined512_dino_pair_v1/verification.json) passed.
The input is the [verified Step 6 corpus](Step6_Run.md): 1,024 B fitting roots and 256 C
development roots, with 5,120 measured action outcomes. The [source index](../runs/step7_smolvlm2_refined512_dino_pair_v1/index/manifest.json)
contains 161 episodes, 28,859 actual observation updates and 25,020 distinct PNG images.
All roots satisfy the four-observation requirement; no roots are excluded.

The selected SmolVLM2 policy remains checkpoint 512 with STOP logit bias −0.25, native saved
configuration/preprocessing and batch-one decoding. Its deliberate selection despite the failed
stopping safeguard remains part of the identity. Step 2 remains the control.

## Feature contract

Instruction features use a native instruction-only user chat through the trained frozen policy.
Token offsets identify the full instruction span; no instruction is truncated. Language, token IDs,
token count, text span and prompt are recorded. The 161 instructions contain 9–329 tokens.

The concrete final decoder norm is `base_model.model.model.text_model.norm`. A hook captures
exactly one native observation prefill and selects the last valid token using the fully expanded
multimodal attention mask. It never captures generated action states. Instruction and context
states have width 960 and are stored in BF16. The saved raw action logits and STOP-corrected
decisions are checked against Step 6 during extraction.

Every real observation from episode start through each root has a context vector, visual input,
actual previous canonical action (START=4 at t=0), and t/original-budget value. These inputs allow
Step 8 to rebuild memory under the current scorer weights. No trainable memory is cached.

The shared frozen DINOv2-S/14 encoder uses the Step 5 weights and preprocessing: RGB float32,
short-side resize/crop to 224, mean/std 0.5 normalization, then resize to 196. Raw 196×384 features
are stored in FP16; 2×2 average pooling is performed in FP32 and stored as 49×384 FP32 features.

The [batch diagnostic](../runs/step7_smolvlm2_refined512_dino_pair_v1/dino_batch_diagnosis.json)
found that cuDNN can choose TF32 patch convolution for large batches. Production extraction uses
one image per native forward, matching online encoding and preserving per-image repeatability.
Policy precision flags, encoder weights and preprocessing are unchanged. The failed initial
large-batch invariance check and its source are retained in the run directory.

For each root, both frozen DINO models receive the last three actual raw feature frames, their
two intervening executed canonical commands, and one F/L/R candidate. Each candidate has one
stored prediction. Only the last predicted frame's 384 visual channels are pooled. STOP is exact
current-view identity and makes no dynamics call. Each measured first-next image is also encoded
for the privileged actual-view diagnostic and warm-up stage, with the same measured labels.

## Scope and validation

This initial version pools the DINO transfer/fresh pair. Step 8 should sample them equally and fit
one scorer for this policy, without a world-model identity input. NWM/CDiT requires a newly
versioned pooled feature set and scorer, followed by rerunning comparisons.

[Native runtime checks](../runs/step7_smolvlm2_refined512_dino_pair_v1/check_runtime.json) cover full
instruction spans, final-norm parity, exact saved policy logits, native processor parity, masks,
the frozen encoder, command predictions, and insufficient-context rejection. Full verification
checks every stored shape, dtype, hash, STOP identity, action/time prefix, label mapping and
pooling result. It also reruns sampled native policy/encoder/prediction forwards and exercises
the actual scorer API with discarded random weights. No scorer is trained in Step 7.

The [source-label audit](../runs/step7_smolvlm2_refined512_dino_pair_v1/source_label_verification.json)
also verifies all 1,280 source label files and 5,120 branch file hashes, exact preservation of the
upstream catalog fields, and canonical-action utility/success mappings.

The completed audit checked 22,680 saved policy-trace logits and all 1,280 root logits. Fresh
native policy forwards matched exactly on 16 roots spanning B/C, R2R/RxR and expert/policy
histories; both world models' predictions matched exactly on those same roots. All array hashes,
shapes, dtypes, finite values, pooling results and STOP identities passed. The real-prefix scorer
interface passed on eight roots. The published feature files and catalogs occupy 6.50 GB.

The extraction is resumable with `python tools/run_step7.py` in the `rolevln` environment.
[Status](../runs/step7_smolvlm2_refined512_dino_pair_v1/status.json) identifies the active stage.
Policy features are atomic per episode; visual and prediction arrays have durable chunk checkpoints.
The protocol pins source files, upstream catalogs, policy/encoder/model identities and precision.

Output cache:
`/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/scorer_features_smolvlm2_refined512_dino_pair_v1/`.
The completed cache provides B/C catalogs, real-prefix tensors, shared visual arrays, both prediction
arrays and a scorer configuration. `scorer_features.dataset.FrozenScorerDataset` exposes prefixes,
candidate features and labels separately. The next step is Step 8 scorer fitting.
