# Step 9 — C navigation calibration and frozen settings

Run: `runs/step9_smolvlm2_refined512_dino_pair_v1/`.
Execution proceeds automatically after the matched [Step 8](Step8_Run.md) runs finish.

The two checkpoints with lowest C branch regret from the predicted-only stage are shortlisted
for each scorer variant. For each checkpoint, the two lowest-regret margins from
`[0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]` are tested in complete navigation episodes.
Margin ties favor the larger margin. The main scorer uses one shared margin across both DINO
models; there is no model-identity input or separate fitting for each predictor.

Screening uses the first 32 R2R and 32 RxR episodes in the existing checkpoint-512 confirmation
set. Selection uses macro success, then SPL, then nDTW, averaged equally over the DINO pair for
the main scorer. Further ties use branch regret, the larger margin, then the earlier checkpoint.
Selected configurations and the base policy run on the remaining 160 episodes, producing the
same 224-episode C confirmation population. No episode is filtered by expert success or length.
The saved ordering puts 16 episodes from each of two scenes in the screen for each dataset.
The complete confirmation population covers seven C scenes, 16 episodes per scene per dataset.
It reproduces the user's authoritative confirmation population exactly; it is not all eight
C scenes. These coverage limits constrain how representative the screening result is.

If a selected system has lower C224 success than the base policy, or ties success with lower
SPL, a conservative margin of 1.0 is validated on all C224 episodes. It must make zero overrides
and reproduce the base actions exactly. If needed, the predeclared finite fallback margin
1,000,000 is checked the same way. This records absence of a validated navigation improvement
without changing the selected reference policy. The chosen finite settings are frozen before E.

Every real observation produces a native policy prefill and exactly one scorer-memory update.
Policy checkpoint 512 and STOP bias −0.25 remain unchanged. The first three observations keep
the base action. Thereafter F/L/R each have one world-model prediction; STOP reuses the current
view. Imagined candidates never update real memory. Errors/nonfinite scores fall back to the
base action, with an explicit event record. Native preprocessing and batch-one policy decoding
are preserved. The null control loads the same current-view encoder and no dynamics model.

Runtime checks compare live policy context, real memory, predicted/actual future features and
scores exactly against saved inputs across both DINO models and all B/C, R2R/RxR,
expert/policy strata. Separate complete-episode checks reproduce the selected reference's
actions, collisions, poses, scores and terminal metrics. The null controller is also checked
against native replay and altered future inputs.

The user deliberately selected checkpoint 512 despite its original stopping-safeguard failure;
this caveat remains in each frozen scorer bundle. The Step 2 control and all upstream frozen
artifacts stay intact. C is development data, so its selected results are not final benchmark
estimates. RxR uses the agreed unified 15-degree action protocol.

Step 10 follows automatically with the frozen policy, null control, privileged actual-view
control, and both DINO prediction systems on a scene-balanced 224-episode E pilot. This is a
pilot on official unseen scenes, not the complete 5,508-episode unseen benchmark. E metrics
cannot change the selected checkpoints, margins or preprocessing.
