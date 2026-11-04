# Receiving-agent task: independent Trial 1 reproduction with Qwen3.5-2B

Prepare, qualify, execute, supervise and record the complete sequence below. The experimental change is **Qwen/Qwen3.5-4B → Qwen/Qwen3.5-2B**. Preserve the established method; do not tune or redesign it.

```text
TARGET_MODEL = Qwen/Qwen3.5-2B
EXPERIMENT_ID = trial_1_qwen35_2b
RUN_EXTRA_STRATEGIES = false

pretrained 2B → S1 group-A SFT → fresh D1 collection → S2 experts+D1
             → F full official-training expert continuation → verified results
```

If the user explicitly chooses another 2B architecture, resolve compatibility before proceeding; never silently substitute Qwen3-VL-2B. E/G/J are additional branches, enabled only by `RUN_EXTRA_STRATEGIES=true`.

## 1. Establish the local workspace

Read this package's README, ENVIRONMENT.md, DATA.md, SOURCE_ORIGINS.json and reference recipes. Use these snapshots rather than older code elsewhere in the GitHub checkout.

Relevant references:
- `source/group_a/`: original native policy, simulator, group-A trainer/collector/auditor.
- `source/full/`: full-split collector/auditor/renderer, same optimization/native policy.
- `reference/trial_1/steps/02_policy_sft/` and `steps/03_dagger/`.
- `reference/trial_1/continuation_from_990_20261103/train_entry.py`: guarded F training and atomic checkpoints.
- `reference/trial_1/strategy_comparison_20261103/`: E/G/J definitions and guards.
- `reference/trial_1/orchestration_recovery_20261104/launch.py`: verified child-process identity repair.
- `reference/trial_1/rxr_unseen_comparison_20261104/`: full RxR audit and evaluation protocol.
- `reference_runs/S1,S2,F/`: actual 4B arguments/counts; historical evidence only.

Do not directly launch historical pipelines: they contain original server paths, queue references, authorizations and 4B checkpoint identities. Their old statuses are not new-run completion evidence. Do not revive the stopped full-data-from-pretrained experiment; its sequence differs from this task.

Discover the target machine's raw data root, two Python environments, Hugging Face cache and authorized GPU. Inspect current jobs before any GPU action. On the original server GPU 0 is the only authorized device; GPU 1 belongs to other services. Respect existing queue admission and never terminate unrelated jobs or change their holds.

Obtain the pinned target model and record its full immutable commit SHA. Use its own complete model, tokenizer and processor. Do not initialize from 4B weights, a quantized model, or smoke-training weights.

Run `python3 -B verify_package.py`, then prepare a new working directory outside this package:

```bash
python3 -B prepare_sources.py \
  --destination /absolute/path/to/trial1_2b_work \
  --data-root /absolute/path/to/RoleVLN/data \
  --policy-python /absolute/path/to/policy-env/bin/python \
  --sim-python /absolute/path/to/habitat-env/bin/python \
  --hf-home /absolute/path/to/hf_cache \
  --revision FULL_40_CHARACTER_2B_MODEL_COMMIT
# Inspect the dry-run output; repeat with --apply to create the working copies.
source /absolute/path/to/trial1_2b_work/environment.sh
```

The generated variables include WORK, PACKAGE, CODE_A, CODE_FULL, POL, SIM, RECORDS, S1, S2, F, D1, EXPERT_A, VAL_C, EXPERT_FULL, VAL_SEEN, EVAL_R2R, EVAL_RXR and EVAL_VENDOR. They are machine-specific bindings, not new recipe choices.

The preparation script changes model identity and local paths only; it launches no job and copies no historical run state. Inspect the resulting changes and record new source hashes. For a pre-existing working directory, verify its preparation/run identity and resume it rather than rerunning preparation or overwriting it.

## 2. Qualify the new backbone and data

The trainer has no general `--model` argument: it constructs StreamConfig directly. Confirm both source copies now bind `qwen35_2b`, `Qwen/Qwen3.5-2B` and the pinned revision. All non-model stream settings must match the reference.

Use the split environments described in ENVIRONMENT.md. Avoid modifying shared environments. Acquire only required model/dependencies, then operate offline. Verify actual installed versions and target CUDA/EGL compatibility.

Check:
- All language, vision and projection weights are trainable; measure the actual 2B parameter count.
- STOP/↑/←/→ each map to one distinct token in the new tokenizer.
- Image-token layout, shifted action labels, turn-end supervision and context lengths are correct.
- Training-window logits agree with inference prefixes; within-turn cached decoding agrees with fresh re-encoding.
- Real representative long windows have finite full-model gradients and fit the GPU.
- Save/load and same-run FP32 master-weight/optimizer/scheduler/RNG/sampler restoration work.
- Habitat can drive the new policy server; loaded-worker identities name the 2B model, checkpoint and intended source.

`tools/check_stream_layout.py` checks both layout and the StreamVLN controller and is CPU-only. `tools/check_stream_backbone.py` performs GPU checks; use full mode for this experiment. Make dimension-dependent probe names valid for the actual 2B architecture without altering the training algorithm. The legacy `check_dagger.py` is not the StreamVLN qualification test.

PolicyClient starts a separate interpreter from its own source root and removes PYTHONPATH. Verify actual subprocess imports/identities; changing PYTHONPATH alone is insufficient. Full evaluators also have a Python binding in common.py, adjusted by preparation.

Keep smoke outputs separate. Real S1 always starts from the pristine pinned pretrained model with seed 0. Never add smoke corrections to D1.

Audit the expert data with the new processor. Require the counts below, readable images, correct exclusions, STOP/frame/action alignment and no token overflow. Transfer or regenerate missing renders following DATA.md. Group-A and full-split auditors are not interchangeable.

| Scope | Usable episodes | Windows |
|---|---:|---:|
| Group-A R2R + RxR | 7,489 + 13,841 | 79,814 |
| Group-C offline validation | 596 | 1,736 |
| Full official train R2R + RxR | 10,819 + 19,402 | 109,784 |
| Official val-seen diagnostics | Audited selection | 1,860 |

Full raw sources contain 10,819 R2R and 19,996 English-guide RxR episodes. Preserve the 594 RxR exclusions: 450 failed/truncated, 138 action drift, six shorter than four actions. Preserve the established successful/non-truncated expert rule, deviation below 0.5 m, and four-action minimum for training.

## 3. Fixed format, optimization and evaluation conventions

Datasets are R2R-CE and RxR-CE English guide (en-US/en-IN) only. Preserve scene identities, source ordering, selected episode IDs and existing validity rules. Do not add EnvDrop, ScaleVLN/HM3D, external co-training data, Trial 2 data or custom Saved places behavior.

Format: 32-action windows, eight evenly spaced historical views, one new observation every four actions, up to four actions per assistant turn, current and memory images both 512×384, max 16,384 tokens. Preserve native Qwen markup without thinking output, exact prompt text, conjunction logic and actual executed-action history. Supervise every action and assistant `<|im_end|>`. Normalize summed cross-entropy by all supervised tokens across the accumulated optimizer batch.

Live action IDs are STOP=0, FORWARD=1, LEFT=2, RIGHT=3, represented by STOP/↑/←/→. Do not substitute the older scorer's internal mapping. Decode by greedy argmax over those four tokens with STOP bias 0. Preserve replanning after an executed-action override and existing within-turn cache behavior.

Simulator: RGB 640×480, HFOV 79, 0.25 m forward step, 15-degree turns, R2R cap 500, RxR cap 700, success distance 3 m; retain existing sensor/sliding settings. Existing RxR expert conversion splits each 30-degree turn into two 15-degree actions; do not convert twice.

All training stages use full fine-tuning, FP32 master weights, BF16 autocast, gradient checkpointing, fused AdamW with existing defaults, LR 2e-5 for language/projection and 5e-6 for the vision tower, zero weight decay, clip 1.0, micro-batch 1 × 128 accumulation, seed 0, one pass. No LoRA, quantization, source weights, artificial 50/50 mixture, added losses, curriculum or early stopping.

Preserve shuffled concatenation and per-draw augmentation/conjunction seeding. Augmentation remains ColorJitter(brightness=.2, saturation=.2), RandomPosterize(bits=4), RandomAdjustSharpness(1.5), RandomAutocontrast with the existing implementation/probabilities.

One pass is floor(windows/128) updates. Record the tail; do not round up or repeat windows. Log each update, validate initially/every 52 updates/finally on the deterministic set of up to 256 windows, and save every 156 updates plus the final update. Always select the final scheduled checkpoint.

## 4. S1: group-A expert SFT

After qualifying and acquiring an otherwise available GPU, launch detached:

```bash
"$POL" "$CODE_A/tools/train_stream_policy.py" \
  --recipe streamvln_stage1 --run-dir "$S1" \
  --train-root "$EXPERT_A" --val-root "$VAL_C"
```

Expected: 79,814 windows, 623 updates, 79,744 draws, 70-window tail; 47-update warmup, then the existing cosine-floor schedule toward 1.85e-5. Preserve the exact scheduler implementation.

LR rows report the NEXT update's rate: step 1 approximately 8.5106383e-7, step 2 approximately 1.2765957e-6. The first update itself uses approximately 4.2553191e-7. Do not repeat the historical erroneous LR stop.

Verify startup counts, model identity, actual 2B trainable count, step-0 validation and the first two updates. Do not impose 4B memory, speed, parameter-count or checkpoint-size expectations on 2B.

On completion verify status, all steps, samples, validation and checkpoint integrity; bind CK1 to S1/checkpoints/step_000623. Run a two-episode load test and C64 diagnostic after the GPU is free:

```bash
"$SIM" "$CODE_A/tools/eval_stream_policy.py" --run-dir "$S1" --weights "$CK1" \
  --datasets r2r rxr --episodes 1 --workers 1 --out "$S1/load_test_C2"
"$SIM" "$CODE_A/tools/eval_stream_policy.py" --run-dir "$S1" --weights "$CK1" \
  --datasets r2r rxr --episodes 32 --workers 4 --out "$S1/eval_C64_final"
```

Screens are operational/development diagnostics, never checkpoint selection. Preserve the original gate that final validation loss is below initial loss before DAgger.

## 5. D1: one fresh group-A DAgger round

Freeze CK1 throughout collection. Do not reuse the 4B corrective dataset.

```bash
"$SIM" "$CODE_A/tools/collect_stream_dagger.py" \
  --run-dir "$S1" --weights "$CK1" --episodes 0 --workers 6 --out "$D1"
```

Expected selected pool: 7,489 R2R + 13,844 RxR = 21,333 episodes. The extra three short trajectories remain eligible for collection under the established rule, though not expert SFT.

Preserve beta 0, collection seed 1, STOP bias 0, executed-action targets/final STOP labels, and controller defaults: waypoint radius 1.8 m, final radius .25 m, far STOP >=3 m, more than 12 disagreements or ratio >.8 triggers takeover until the next waypoint, existing accumulated-error/reset logic and premature STOP veto. Retain episodes ending within .5 m with path ratio <.93 after takeover or <.85 without. This deliberately retains recoveries/detours.

Require matching worker/checkpoint identities, completed selected-episode coverage, recorded abort reasons and a passed audit:

```bash
"$POL" "$CODE_A/tools/audit_stream_data.py" --train-root "$EXPERT_A" \
  --val-root "$VAL_C" --dagger-root "$D1" --out "$D1/data_audit.json"
```

Record actual kept episodes and usable windows. The old 4B values of 11,660 kept trajectories and 46,907 windows are observations, not constraints on 2B.

## 6. S2: group-A experts plus D1

Initialize CK1 weights with a fresh optimizer:

```bash
"$POL" "$CODE_A/tools/train_stream_policy.py" --recipe streamvln_stage2 \
  --run-dir "$S2" --init "$CK1" --train-root "$EXPERT_A" "$D1" --val-root "$VAL_C"
```

For audited D1 window count D, use W=79,814+D, updates=W//128, warmup=ceil(.03×updates), samples=128×updates. Keep the existing cosine schedule to zero and root order expert then D1. Do not force 990 updates.

Verify and bind CK2 to the actual final checkpoint. Run the same C64 diagnostic and full R2R seen/unseen evaluation; report the C64 comparison with S1 separately from full-split results.

## 7. F: full-training expert continuation

Initialize from CK2 with fresh AdamW. Use EXPERT_FULL only; no new collection and no D1 replay in F. The full expert set includes group-A examples again. Do not reduce F to only new scenes.

Use VAL_SEEN, the official val-seen diagnostics. Old group C is inside full training and is no longer held-out for F. Audit with CODE_FULL's full-split auditor.

Instantiate a new F_ENTRY from the reference F train_entry.py pattern: bind the local source, new run, new data/audit identities and CK2 initializer; preserve atomic checkpoint publication, run locking, checksum receipts and same-run resume verification. Use new records, not the old authorization or receipts. If using the reference stage supervisor, apply the verified child capture repair so it verifies the actual launched script, not an unrelated parent-directory string.

```bash
"$POL" "$F_ENTRY" --recipe streamvln_stage2 --run-dir "$F" --init "$CK2" \
  --train-root "$EXPERT_FULL" --val-root "$VAL_SEEN" \
  --keep-trainer-states 10000 --keep-final-state
```

Expected: 109,784 windows, 857 updates, 109,696 draws, 88-window tail, 26 warmup updates and existing cosine decay to zero. Bind CKF to F/checkpoints/step_000857 after verification.

Original S1/S2 keep the newest recovery state and clean their final recovery state on successful completion. F keeps all recovery states, including the final one. Do not apply the S1/S2 no-state-at-completion rule to F. Reserve disk space for all retained F checkpoints/states before launch.

## 8. Full navigation evaluation

Use the prepared EVAL_R2R and EVAL_RXR copies. EVAL_RXR explicitly accepts only RxR val-unseen; it is not a substitute R2R entry point. Install pinned metric dependencies in SIM and copy EVAL_VENDOR to each evaluation output's vendor/vlnce directory, preserving SOURCE.json/LICENSE and the source checksum check.

For every policy/split use a separate output and bind ROLEVLN_QUEUE_OUTPUT, ROLEVLN_QUEUE_CODE_ROOT and ROLEVLN_NATIVE_CODE_ROOT to the intended output/evaluator/new native source. POL must also be bound in the copied evaluator common.py. For RxR generate a local split audit and 2B protocol identity using the reference audit/verification implementation, then set ROLEVLN_RXR_AUDIT and ROLEVLN_RXR_PROTOCOL. Do not reuse the 4B canonical config hash or disable identity guards.

Use the evaluators' existing CLI with explicit --dataset, --split, --name, --run-dir, --weights and --workers 6. Keep default --episodes 0 for complete splits, seed 0, memory-mode empty, no force-slot and STOP bias 0. Optional memory being empty does not disable the native eight-view navigation history.

Required coverage:
- S2 and F: R2R val-seen 778 and val-unseen 1,839 episodes each.
- After F's R2R work: RxR val-unseen for F, then S2, then S1; 3,669 English-guide episodes across 11 held-out scenes per policy.

Verify exact episode identities/coverage, no duplicates, scenes, role/language, checkpoint and worker hashes, configuration, source/metric hashes and aggregation. Never reuse another model's cached episode results. Report SR, SPL, nDTW, sDTW and other measured metrics. Unseen results are report-only; no tuning, selection or extra rounds. RxR's 15-degree/700-step protocol is an internal matched comparison, not the official 30-degree protocol.

## 9. Optional E/G/J

Only when RUN_EXTRA_STRATEGIES=true, reuse F and execute E→G→J with verification and full R2R seen/unseen plus RxR unseen after each final:
- E: CK1 → one pass EXPERT_FULL + D1.
- G: CK2 → the same mixture; never initialize G from E.
- J: freeze CKF → one fresh full-training collection DJ → audit → CKF weights with a fresh optimizer → one pass EXPERT_FULL + DJ. Do not include D1 in J or initialize from G.

Use stage-two schedules/settings and fresh optimizers. E/G share order, seed and budget: (109,784+D)//128 updates. J's collection uses CODE_FULL and selects 10,819 R2R + 19,408 RxR eligible episodes; its training budget is derived from the audited fresh round. Do not copy 4B E/G's 1,224 updates. Report unequal compute budgets and do not claim between-seed statistical significance from one seed.

## 10. Supervision, recovery and records

Launch long jobs detached with persistent logs/PID files and a single owner/supervisor. Check for existing live processes and verified completion before starting. Use process identity including start time, not PID existence alone. Protect other jobs and run one admitted GPU stage at a time.

Check every 30–60 minutes: progress, logs, process identity, GPU memory/utilization and disk space. Investigate training progress stale >20 minutes or collection stale >30 minutes, accounting for checkpoint writes. Measure 2B speed/memory; do not enforce historical 4B expectations or promise half its runtime.

Recovery: non-finite loss/gradient stops for diagnosis; OOM requires inspecting contention without changing the recipe; transient failures restore the newest complete valid state from that same run. Restore master weights, optimizer, scheduler, RNG and samples_seen. Use --resume, omitting --init from the recovery command. New stages use --init and a fresh optimizer. Reject partial or mismatched states. Before any recovery checkpoint exists, preserve the interrupted run under a timestamped name before a bounded restart. Collection resumes completed atomic episodes with unchanged identity. Stop after two failed recoveries or a repeated unexplained error. Terminate only owned, verified processes through their PID records/clean shutdown, never pkill -f.

Do not manually delete artifacts to obtain space; preserve the existing normal handling of temporary/rejected frames and S1/S2 recovery-state cleanup. Do not change batch, context, image size, precision, filtering, coverage or checkpoints to make a run fit.

Maintain new two-space JSON records and RESULTS.md with model revision/parameter count; source/environment/data identities; exact commands; all stage initializers; UTC start/end, wall/active hours and interruptions; updates/draws/tails; validation loss, action accuracy, each class including STOP, and all turn offsets; throughput/peak memory; DAgger selected/finished/kept/aborted counts, action sources and takeovers; final checkpoint paths/sizes/SHA256; load tests/C64 diagnostics; full navigation coverage/metrics and matching 4B comparisons. Verify checkpoint aliases with readlink -f.

Completion requires all enabled stages, verified final checkpoints, prescribed evaluations and records—not just process exit or a checkpoint directory. If the session cannot supervise the whole sequence, leave durable stage/PID/ETA/next-dependency instructions and accurately state which supervision remains active.

Action history is visible. About 73% offline accuracy, or 74.3% at turn offset 0, is achievable by the reference history-only baseline. Accuracy near those values does not establish visual use; training/validation numbers are not navigation results.

Start with inspection, package verification, environment/data preparation and isolated 2B qualification. Then execute this sequence without changing its scientific recipe.
