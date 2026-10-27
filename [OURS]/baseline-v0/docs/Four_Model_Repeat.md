# Four-model SmolVLM2 repeat

Prepared on 27 October 2026 KST while the initial DINO-pair C confirmation is running. The GPU stages described here have not run yet. Automatic progression is authorized by the user.

The initial DINO-pair E pilot finishes first. The separately versioned [NWM/CDiT fit](Step5B_Run.md) then runs its GPU checks, fits both dynamics models, freezes their C-selected weights, and measures their common-DINO feature errors. The continuation below starts only after that result passes verification.

The selected policy remains refined SmolVLM2 checkpoint 512, with exactly −0.25 applied to the STOP logit before four-action argmax. Its saved preprocessing and configuration remain fixed. This is the user's deliberate selection despite 149 incorrect stops and the failed original stopping safeguard; the Step 2 control is preserved.

## Features and scorer training

Step 7B reuses the verified Step 6 labels and Step 7 instruction, native policy-context, real-image, and DINO-prediction caches. It adds NWM-transfer and fresh-CDiT futures for the same 1,024 B fitting roots and 256 C development roots. Each new model receives four actual context frames. It produces one completed diffusion sample per movement command in a fixed FORWARD/LEFT/RIGHT batch, with 50 sampling steps and explicitly recorded matched seeds. The decoded views use the common frozen DINO encoder; STOP is the exact current-view feature and never calls a world model. Native VAE preprocessing runs at batch one to match navigation. Fresh replay checks cover both ends of every group/dataset/prefix-source stratum.

Step 8B fits one pooled scorer and a matched null-future control for 2,048 updates each, using the original seed, root order, optimizer, architecture, and update budget. Four actual-only epochs precede four mixed epochs and eight predicted-only epochs. Mixed epochs use 512 actual roots and 128 roots per world model; predicted epochs use 256 roots per world model. The scorer receives no model-identity input. Full actual prefixes are replayed with current scorer weights; gradients cover at most the final 32 memory updates.

The pooled actual-only warm-up and every null-control checkpoint must reproduce the initial DINO fit bit for bit. This checks that adding frozen future representations does not silently change the shared supervision or control. New model-dependent updates must pass exact optimizer-resume checks. C branch diagnostics shortlist checkpoints; they are not navigation success rates.

## Calibration and repeated evaluation

The training recipe was declared at 22:40:48 UTC on 26 October, and the calibration/evaluation recipe at 22:54:24 UTC, before any initial E results. The latter fixes:

- The exact authoritative C224 confirmation population: 112 episodes per dataset across seven scenes. The original population omits one of the eight C scenes.
- A nested, scene-balanced C64 screen: four or five episodes per scene per dataset, selected without outcomes using seed 9. This new repeat differs from the initial two-scene C64 screen and is versioned accordingly.
- Two checkpoints per variant and two branch-selected margins per checkpoint. Closed-loop ranking uses macro SR, SPL, and nDTW, averaged equally over all four world models for the pooled scorer, then branch regret, larger margin, and earlier checkpoint.
- Confirmation on the remaining 160 episodes. If confirmed SR is below base, or tied SR has lower SPL, a conservative fallback must demonstrate zero overrides and exact base actions on all C224 before freezing.
- The same 224-episode E pilot as the initial DINO run, with seven systems: policy, learned null, privileged actual-next-view control, and the four frozen world models. All four use exactly the same pooled scorer weights and margin. Paired scene bootstrap uses 2,000 resamples and seed 10.

The initial E results will be available before the repeat executes. This is therefore a predeclared repeat on the same evaluation population, not a new independent test set. E outcomes cannot change the recorded recipe, checkpoint selection, or margin. The pilot is not the complete 5,508-episode unseen benchmark. The actual-next-view control is privileged. RxR retains the recorded unified 15-degree turns. H100 measurements do not establish deployment fit on an 8 GB device.

## Durable execution

The watchers are separate from the immutable initial runtime. Each stage verifies its predecessor and records failures instead of silently continuing with incomplete artifacts.

| Stage | Run directory | Entry point |
| --- | --- | --- |
| NWM/CDiT fitting | `runs/step5b_smolvlm2_refined512_nwm_pair_v1` | `tools/auto_nwm_repeat.py` |
| Feature extension | `runs/step7b_smolvlm2_refined512_four_models_v1` | `tools/auto_four_features.py` |
| Pooled scorer fit | `runs/step8b_smolvlm2_refined512_four_models_v1` | `tools/auto_four_training.py` |
| C calibration | `runs/step9b_smolvlm2_refined512_four_models_v1` | `tools/auto_four_calibration.py` |
| Repeated E pilot | `runs/step10b_smolvlm2_refined512_four_models_v1` | Same calibration watcher |

New native-runtime GPU checks, complete-prefix parity, cache hashes, immutable sources, and matched episode/action audits gate the runs. The CPU paired-reporting check has passed. Feature extraction, fitting, live runtime checks, and navigation results remain pending.
