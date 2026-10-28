# Step 8 — scorer training and matched control

Run: `runs/step8_smolvlm2_refined512_dino_pair_v1/`.
The user authorized automatic progression through the remaining recorded steps on 27 October 2026.

The verified [Step 7 cache](Step7_Run.md) supplies 1,024 B fitting roots and 256 C development
roots. Refined policy checkpoint 512, STOP bias −0.25, the deliberate failed-safeguard selection,
and both frozen DINO models retain their identities. Group E is reserved for final evaluation.

Both scorers have the existing 2,890,115-parameter architecture. The main scorer pools predictions
from both DINO models without model identity as an input. The control has the same parameters,
initialization, root ordering and update budget. Its movement features are constant zeros before
the existing trainable projection biases and future-type embedding, providing a learned shared
null future. STOP keeps the already available current view. The control never consumes actual
or predicted movement views in candidate scoring.

The fixed budget is 16 epochs of 128 updates: four actual-view epochs, four mixed epochs, then
eight predicted-only epochs (512 / 512 / 1,024 updates). Mixed epochs contain exactly 512 actual,
256 transfer and 256 fresh-model examples. Predicted epochs contain 512 from each model. The
null control follows the same schedule with its null movement representation throughout.
Each update uses eight roots and all four actions, AdamW at 1e-4, weight decay 0.01 and gradient
clipping at norm 1.0. Loss weights and model architecture follow the existing specification.

Every batch rebuilds real memory under its current scorer weights. Earlier updates run without
gradients; at most the last 32 real updates receive recurrent gradients. Batched inactive rows
keep their prior memory. Within-batch projection reuse does not cache trainable memory or reuse
representations across optimizer updates. Native sequential forward and all-parameter gradient
checks passed on real roots with differing prefix lengths and instruction masks.

Training uses FP32 with matmul TF32 disabled and deterministic algorithms enabled. The MHA fast
path is disabled so no-gradient and gradient replay use the same attention implementation.
GPU 0 is used; upstream policy and model
weights remain read-only. The final deterministic pilot measured about 0.40 seconds per update and 5.13 GB peak
allocated memory for scorer fitting with resident frozen features, not full navigation inference.

Checkpoint/optimizer resumption is checked against uninterrupted updates. The initial test found
tiny differences in attention gradients and subsequent AdamW updates. Deterministic algorithms
and an explicit cuBLAS workspace configuration make resumed updates bit-identical. Checkpoints
load on CPU before state restoration to retain native optimizer scalar placement. The initial
failed tests, sources and gradient-difference diagnostic are retained in the run directory.

Every epoch is checkpointed and evaluated on C. Checkpoint selection considers only the final
predicted stage and uses mean measured branch utility regret, equally averaged over both DINO
models. Actual-view diagnostics are separate. The two best branch candidates per variant form
the closed-loop shortlist; C navigation checks and Step 9 margin calibration are required before
freezing deployment settings. Branch outcomes are not navigation success rates.

[Protocol](../runs/step8_smolvlm2_refined512_dino_pair_v1/protocol.json),
[status](../runs/step8_smolvlm2_refined512_dino_pair_v1/status.json),
[native checks](../runs/step8_smolvlm2_refined512_dino_pair_v1/check_training.json), and
[orchestration checks](../runs/step8_smolvlm2_refined512_dino_pair_v1/check_orchestration.json)
record execution. The next operations proceed automatically under the user's authorization.
