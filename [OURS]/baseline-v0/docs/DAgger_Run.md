# SmolVLM2 Step 3 pilot

Completed on 24 October 2026. Run: `runs/step3_smolvlm2/`; see [results](../runs/step3_smolvlm2/RESULTS.md) and [final verification](../runs/step3_smolvlm2/final_verification.json).

The round collected 256 episodes with 125,730 valid corrective labels and completed 2,048 refinement updates. All four checkpoints were tested on the same 64 group-C episodes. Update 2,048 tied the Step 2 parent at four successes, but had lower macro SPL (0.0216 versus 0.0472). This round therefore retained **Step 2 update 3,072**, saved in `runs/step3_smolvlm2/frozen_reference/`. Its weights, processor, configuration and source copies passed verification; the loaded model has zero trainable parameters.

That bundle is now preserved as the Step 2 control. The user's 26 October handoff selected refined checkpoint 512 with STOP bias −0.25 for downstream experiments despite its failed stopping safeguard. The separate selected reference and collection protocol are recorded in [Step4_Run.md](Step4_Run.md); the original Step 3 results and bundle are unchanged.

The starting policy is Step 2's selected adapter, update 3,072. One collection round uses 128 R2R and 128 RxR group-A episodes, selected with seed 1 across all available scenes (44 R2R, 42 RxR). The episode pool is the same set with successful, untruncated expert replays and less than 0.5 m reference drift used for Step 2. This excludes known broken expert paths; **failures of the learned policy are retained**.

The policy drives every action (`beta=0`). At each actual state, the expert supplies a label but does not override the policy. The recovery route consists of the validated expert replay's actual positions, followed by the task goal. Consecutive repeated positions are removed. Waypoints are projected onto the starting navigation-mesh island, with a maximum 0.5 m snap distance; the dataset's final goal may be projected by up to 1 m, matching the final stopping tolerance. Progress advances only through consecutive waypoints within 0.5 m geodesic distance; it never jumps to the nearest later point or moves backward. After the route is completed, the expert labels STOP within 1 m of the task goal. Habitat's shortest-path follower is reset on every query because its previous recommendation may not have been executed. Unreachable targets or follower errors produce an invalid label, never an artificial STOP.

The policy sees only its instruction and actual RGB/action history. Each episode stores expert labels separately from executed actions, plus actual poses, collisions, progress and label-error records. For T actions, T+1 RGB frames are stored, including the final transition observation. Invalid labels are excluded from refinement while their actual history remains available. Collection stops for review if more than 1% of labels are invalid.

Refinement initializes the same decoder LoRA adapters from the selected SFT checkpoint. A fresh AdamW optimizer uses learning rate 5e-5 with warmup/cosine decay, 2,048 updates and effective batch 64. Each update samples exactly 32 original expert examples and 32 DAgger examples, uniformly with replacement within each source. The masked target-token cross-entropy, image processor, prompt, history window and action decoder remain unchanged.

Checkpoints at updates 512, 1,024, 1,536 and 2,048 are evaluated on the same 32 R2R + 32 RxR group-C episodes used in Step 2. The original SFT checkpoint is also a candidate. Selection uses dataset-macro success, then macro SPL, then lowest offline token cross-entropy, then earliest update. Group E is unused.

The selected reference is copied into `frozen_reference/` with its adapter, processor, policy configuration, prompt/decoding source, inference settings and file hashes. Its identity must accompany later continuation labels and policy features. Replacing it requires regenerating those dependent artifacts.

Checks cover action/history separation, retention of failed rollouts, skipped invalid labels without renumbering time, group-A enforcement, sampler resume, monotonic reference progress, expert-only navigation, recovery after legal perturbations, and an actual refinement smoke run. The existing Step 2 simulator evaluator is reused unchanged.

Run or resume with the `rolevln` environment:

```bash
python tools/run_step3.py
```

The pipeline rejects duplicate instances and source/configuration changes. Its training checkpoints and episode folders are written atomically. All work uses GPU 0.

The final-goal projection limit was widened from 0.5 m to 1 m after one valid dataset goal required a 0.655 m projection. Intermediate-waypoint limits and previously accepted recovery routes were unchanged. All 249 existing episodes were preserved byte for byte, and the seven remaining episodes passed expert checks before resuming. The original and corrected collector configurations remain versioned with the corpus; [the fix record](../runs/step3_smolvlm2/goal_projection_fix.json) and [collection integrity check](../runs/step3_smolvlm2/check_collection_integrity.json) document the recovery.
