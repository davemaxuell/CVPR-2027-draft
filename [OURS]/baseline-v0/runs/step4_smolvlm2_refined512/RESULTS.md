# Step 4 completed — actual group-A world-model transitions

Completed: 2026-10-25T19:37:08.054936+00:00

Selected experimental policy: refined SmolVLM2 checkpoint 512, with 0.25 subtracted from the STOP logit before four-action argmax. The saved model configuration and native evaluation preprocessing are retained.
The user explicitly selected this policy despite its failed original stopping safeguard: 149 incorrect stops among 224 group-C confirmation episodes. It achieved 29/224 successes and a 9.6% forward collision rate. This decision does not relabel the failed criterion as passed. The original Step 2 reference is preserved as a control.

Versioned reference: `reference/`; policy identity: `729480bf5c8402b07cad653a7ef0f7124be212d50255077290316ba686777ca7`.
New selected-policy collection: 256 episodes, 88,306 executed actions, 2,848 actual branch roots and 8,544 F/L/R outcomes.
Combined corpus: 21,973 episodes, 2,488,382 movement transitions, 2,422,469 matched four-observation samples and 65,913 separately indexed early DINO-only samples.

| Source | Episodes | Movement transitions |
| --- | ---: | ---: |
| expert | 21,333 | 2,197,643 |
| dagger_round1 | 256 | 125,669 |
| dagger_round2 | 128 | 71,222 |
| selected_policy | 256 | 93,848 |

Corpus: `/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/wm_groupA_smolvlm2_refined512_v1` (`manifest.json`, `catalog.sqlite`).
Loader: `wm.data.TransitionDataset`; DINO and CDiT share matched sample IDs and observed targets, taking their last three or four real frames respectively.
All data comes from group A. STOP is excluded from dynamics fitting. Blocked forward motions are retained. The actual executed movement at a branch root is indexed once, with alternate movements added from real simulator branches. No imagined targets or future realized displacement are model inputs.
Integrity: all 2,513,438 images hashed and headers checked; 113,108 images fully decoded during indexing. Accessed images are hash-checked by the loader.
New visits/branches use lossless PNG; historical expert/DAgger RGB remains JPEG quality 95. The catalog references those existing files, so all four source corpora must be retained. Historical DAgger sources retain their original collecting-policy identities; they are not represented as visits from the new refinement.
[Final verification](final_verification.json) passed, including native preprocessing/decoder parity, branch root restoration, unchanged actual trajectories and metrics, left/right signs, blocked motion, and paired-family data loading.

The final audit checked 273 paired DINO/CDiT samples and retained 106,079 blocked-forward transitions. An additional [terminal-frame check](check_terminal_frames.json) fully decoded all 256 new final trajectory PNGs and verified their pixels against the live simulator hashes. [Runtime environment](runtime_environment.json) records the exact Python, package and GPU versions.

RxR episode 4838 had unavailable goal-distance/SPL diagnostics. A [metadata serialization correction](nonfinite_diagnostics_fix.json) records these as null with explicit nonfinite-value tags. The [recovery check](check_serialization_recovery.json) confirmed all 175 previously completed episode records were unchanged and all 767 images from that episode were reproduced byte for byte. Policy, decoding, simulation and branch semantics were unchanged; both collection configuration versions are retained.

The corpus is ready for Step 5. World-model training has not started; the transferred predictor checkpoint and training budget remain Step 5 choices.
