# Step 3 completed — SmolVLM2 DAgger pilot

Completed: 2026-10-24T08:18:44.409889+00:00

Collected 256 group-A episodes across all available A buildings (44 R2R, 42 RxR): 125,730 policy actions, 125,730 usable corrective labels.
The Step 2 policy drove every action (beta=0). Collection used the validated Step 2 expert episode pool. Corrective labels followed the actual replay path, with monotonic progress, 0.5 m waypoint tolerance and 1 m final STOP tolerance. Invalid labels were excluded; failed policy episodes were retained.
Refinement: 2,048 updates × 64 examples = 131,072 samples; 65,536 from each of the original expert and DAgger sources. Decoder LoRA only, learning rate 5e-5 with a fresh AdamW optimizer and warmup/cosine schedule.
Selection: the same 32 R2R + 32 RxR group-C episodes, including the Step 2 checkpoint as a candidate. No group E use.

| Candidate | Offline action accuracy | Offline CE | Macro success | Macro SPL |
| --- | ---: | ---: | ---: | ---: |
| Step 2 parent (3072) | 0.7640 | 0.1523 | 0.0625 | 0.0472 |
| DAgger 512 | 0.7565 | 0.1620 | 0.0312 | 0.0065 |
| DAgger 1024 | 0.7550 | 0.1602 | 0.0000 | 0.0000 |
| DAgger 1536 | 0.7525 | 0.1603 | 0.0156 | 0.0042 |
| DAgger 2048 | 0.7510 | 0.1596 | 0.0625 | 0.0216 |

The final DAgger checkpoint tied the parent at 4/64 successes, but its macro SPL was lower (0.0215928 versus 0.0472389). The parent wins the declared tie-break. This round did not improve the selected navigation result. The final DAgger policy entered the 3 m goal region in 26/64 episodes but successfully stopped there in only four; navigation and stopping performance remain weak.

Frozen reference: **Step 2 parent (3072)**, in `frozen_reference/` (adapter, processor, policy configuration, prompt/decoding source and file hashes).
Selection rule: highest dataset-macro success, then macro SPL, then lowest offline token CE, then earliest update.

[Final verification](final_verification.json) passed: all five candidates used the same 64 group-C episodes and simulator/decoding settings; aggregate metrics were recomputed from episode records; adapter and bundle hashes match; the frozen model loaded on GPU with zero trainable parameters and all modules in evaluation mode. The Step 2 and Step 3 source snapshots also match their recorded hashes.

Collection checks passed for 88 expert replays, 16 recoveries after legal perturbations, and an actual refinement smoke run. A narrowly scoped [final-goal projection fix](goal_projection_fix.json) allowed collection to resume after 249 episodes; [integrity checks](check_collection_integrity.json) confirmed those records were unchanged and all 256 episodes have aligned actions/labels and T+1 frames. Both collector configurations remain versioned with the corpus.

These are small development-set pilot results, not final benchmark results. RxR uses 15-degree turns, differing from its official 30-degree protocol.
World-model fitting and scorer work have not started. The next step is group-A transition collection for the world model.
