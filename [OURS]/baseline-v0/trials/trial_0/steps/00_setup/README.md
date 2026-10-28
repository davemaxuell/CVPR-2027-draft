# Step 0 — setup and interface evidence

These are copied records from 23 October 2026, checked again for this archive;
the simulator and GPU checks were not rerun while organizing the trial.
Step 0 is the archive label for setup preceding the original steps 1–10.

| Check | Saved result | Evidence |
| --- | --- | --- |
| R2R-CE expert replay | 20/20 successes across 11 group-A scenes | [Report](check_ce_replay_r2r.json) |
| RxR-CE expert replay | 20/20 successes across 14 group-A scenes | [Report](check_ce_replay_rxr.json) |
| Native SmolVLM2 policy step | Forward/backward with 8 frames, LoRA r16, 8,683,520 trainable parameters | [Report](check_policy_step.json) |
| Scorer design | All 13 structural checks passed | [Report](scorer_design_check.json) |
| Model compatibility | All 10 check groups passed | [Report](model_compatibility_check.json) |

[Environment setup recipe](setup_env.sh) is preserved as provenance. The setup
script contains the original paths and is not a new trial launcher.
