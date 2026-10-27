# Step 2 completed — SmolVLM2 pilot

Completed: 2026-10-23T13:02:38.449346+00:00

Group A: 21,333 usable episodes / 2,218,976 available steps.
Group C: 600 rendered episodes; 596 usable / 46,373 steps for offline validation.
Training: 4,096 optimizer updates, effective batch 64, 262,144 sampled steps (0.1181 passes); decoder LoRA only.
Offline metrics use a fixed 2,000-step subset; simulator selection uses the same 32 R2R + 32 RxR episodes at every checkpoint.
Simulator observations are actual live RGB and executed-action history. Group E remains untouched.

| Checkpoint | Offline accuracy | Offline CE | Macro success | Macro SPL |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 0.7515 | 0.1646 | 0.0000 | 0.0000 |
| 2048 | 0.7565 | 0.1577 | 0.0156 | 0.0156 |
| 3072 | 0.7640 | 0.1523 | 0.0625 | 0.0472 |
| 4096 | 0.7615 | 0.1508 | 0.0156 | 0.0156 |

Selected checkpoint: `/data/team_a/dave-workspace/RoleVLN/[OURS]/baseline-v0/runs/step2_smolvlm2/train/checkpoints/step_003072` (also `train/best_checkpoint`).
Selection: highest dataset-macro success, then macro SPL, then lowest offline token CE, then earliest step.

These are pilot development results, not final benchmark results. RxR uses 15-degree turns, differing from its official 30-degree protocol.
Next step is the separately specified DAgger refinement; this checkpoint is the initial SFT policy, not the final frozen reference.
