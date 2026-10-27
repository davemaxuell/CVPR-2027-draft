# Step 5B — NWM/CDiT transfer and fresh dynamics

Run: `runs/step5b_smolvlm2_refined512_nwm_pair_v1/`.
This is the next world-model repeat under the user's standing authorization to proceed
automatically. Its GPU work starts after the initial DINO [Step 10](Step10_Run.md) pilot finishes.

The released `facebook/nwm` CDiT-S/2 checkpoint at revision
`0821a1a7b1ae938539e32f13f5ad82465e0f3fde` passed strict CPU loading of all 219 EMA tensors
(49,227,680 parameters). The frozen `stabilityai/sd-vae-ft-ema` VAE is pinned at
`f04b2c4b98319346dad8c65879f680b1997b204a`. Source optimizer and scaler state are discarded.
The paired fresh model uses the exact same native architecture and trainable parameter set.
See the [asset record](../runs/step5b_smolvlm2_refined512_nwm_pair_v1/assets/selection.json).

CPU checks passed native command normalization, VAE posterior sampling, CE preprocessing,
denoiser output agreement, finite native-objective gradients, and repeatable RNG-preserving
sampling. The CPU sampler test uses two steps solely to exercise the interface. GPU throughput,
mixed-precision agreement, exact optimizer/EMA restart, and packaged three-candidate prediction
are mandatory checks before the training protocol is locked.

Both variants will use the original **2,422,469 matched group-A transitions**, the same seeded
sample order, and the same 3,072 group-C expert-path validation samples as the DINO pair. The
875,174 distinct source images are encoded with a frozen FP32 VAE. Mean/log-variance are stored
as FP16; fresh FP32 posterior samples are drawn during fitting. C images do not fit the VAE.

The declared recipe is one full pass, batch 64, native 1,000-step linear epsilon/learned-variance
diffusion loss, AdamW at constant 1e-4 with zero weight decay, gradient clipping at 1, and EMA
decay 0.9999. The same optimizer, random draws and budget apply to both variants. Selection
uses the lowest C dataset/action macro EMA loss, with an earlier-update tie break, evaluated
at initialization, every 4,096 updates, and update 37,852. The GPU speed test records the cost
before that recipe is locked.

The CE command is nominal local forward/left displacement plus left-positive yaw. Translation
is divided by 0.25 m and then uses the source x/y min/max normalization; yaw stays in radians.
Thus F/L/R are approximately `[-0.06667, 0, 0]`, `[-0.33333, 0, +pi/12]`, and
`[-0.33333, 0, -pi/12]`. Relative time is `1/128` for one CE command, without assigning physical
seconds. A collision changes the observed target, never the input command. Four real context
frames are required; future poses and realized displacements are excluded.

The agreed CE resize/center-crop 224 preprocessing differs deliberately from native NWM's
4:3-to-square rescaling. It matches the existing CE bridge and remains identical between the
transfer and fresh trials. The latent scale is 0.18215. The proposed deployment budget is 50
complete native ancestral steps, versus the upstream 250-step evaluation. The GPU check
profiles both budgets; the 50-step pilot choice is not selected from E results. F/L/R are
sampled together in that order, with shared real context and one noise sample per action.
Completed four-channel latents are decoded to RGB and passed through the original frozen DINO
encoder at batch one. STOP is the current view and has no world-model call.

After freezing the selected EMA weights, all four world models receive a common 49×384 pooled
feature diagnostic on the same 3,072 C transitions. This diagnostic cannot change checkpoint
or sampler selection. These expert-path samples have no collisions, so this is neither a
failure-state validation nor a navigation-success result.

The policy remains refined SmolVLM2 checkpoint 512 with STOP bias −0.25. Its deliberate selection
despite the failed stopping safeguard is retained; the Step 2 control remains intact. The
existing B/C measured outcome labels can be reused because the policy identity is unchanged.
The next scorer version will pool predictions from all four frozen world models and use one
shared scorer across them.

`tools/auto_nwm_repeat.py` waits for the DINO E pilot. `tools/run_nwm_pair.py` then runs checks,
caching, both training passes, freezing, common-feature diagnostics and final reporting.
Each stage records a status and log. Training restores model, EMA, optimizer, sample position
and RNG state. Source hashes are locked before caching/training; failures stop dependent stages.
