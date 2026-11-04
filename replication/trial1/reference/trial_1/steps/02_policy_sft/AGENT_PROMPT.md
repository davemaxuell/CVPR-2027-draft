You are taking over one task: run, supervise and record **stage one of trial_1** (a full fine-tune of
Qwen3.5-4B as a navigation policy, StreamVLN recipe, one pass, about 17 hours on one GPU). Everything is
already designed, implemented and tested. Your job is to execute it exactly, keep it healthy, and record
the outcome. Do not redesign, tune or "improve" anything.

## What is fixed (decided by the user, do not change)

- Command: `tools/train_stream_policy.py --recipe streamvln_stage1 --run-dir <RUN>`, nothing else
  (`--resume` only when recovering). The recipe means: full fine-tuning of all 4,539,265,536 weights,
  one pass = 623 updates × 128 windows, lr 2e-5 (vision 5e-6), 47-update warmup then cosine to 1.85e-5,
  StreamVLN augmentation, validation every 52 updates, checkpoints at updates 156, 312, 468, 623.
- The final checkpoint (update 623) is the result; the others exist for diagnosis only.
- Data: 21,330 group-A expert episodes, 79,814 windows (audited 2026-10-28).
- Background, only if you need it: `trials/trial_1/README.md`, `trials/trial_1/steps/02_policy_sft/README.md`,
  `trials/trial_1/docs/StreamVLN_Recipe.md`.

## Environment

```bash
cd "/data/team_a/dave-workspace/RoleVLN/[OURS]/baseline-v0"   # always quote: the path contains [OURS]
RUN=runs/trial_1/step2_qwen35_4b_streamvln
POL=/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python           # training env (Python 3.12)
SIM=/home/bufsgpu/yes/envs/rolevln/bin/python                  # Habitat env (Python 3.9), only for the load test
export HF_HOME=/data/shared/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
```

This must run on the H100 server (GPU 0 is an "NVIDIA H100 NVL", ~95,830 MiB). The model is cached; no
network is needed.

## 0. Find out where things stand

Check in this order:
- `$RUN/train.pid` names a live process (`ps -p $(cat $RUN/train.pid)`) → running (it may still be loading): do **2**.
- `$RUN/status.json` exists and says `"complete"` → do **4**.
- No `$RUN/log.jsonl` → not started: do **1**.
- The process is gone, no `status.json`, and no checkpoint folder has both `trainer_state.pt` and
  `master_weights.safetensors` (it stopped before update 156) → nothing to resume: set the folder aside with
  `mv "$RUN" "$RUN.stopped-$(date -u +%Y%m%dT%H%M%SZ)"` and do **1**.
- Otherwise (a log exists, the process is gone, no `status.json`) → it stopped: do **3**.

## 1. Pre-flight and launch

1. `nvidia-smi -i 0 --query-gpu=name,memory.used,memory.total --format=csv` and
   `nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader`: GPU 0 must be an H100 with
   under 1,000 MiB used and no processes. `pgrep -af train_stream_policy.py` must print nothing. If not,
   stop and report; do not touch other processes.
2. `df -h /data/team_a`: at least 150 GB available (4 × 9.1 GB checkpoints + a 54 GB resume state).
3. `mkdir -p "$RUN"`, then
   `$SIM tools/check_stream_layout.py > /dev/null && echo layout ok` (seconds) and
   `$POL tools/audit_stream_data.py --out "$RUN/preflight_data_audit.json"` (~2 min, CPU): both must succeed,
   and the audit must report `"passed": true` with 79,814 training windows. Otherwise stop and report.
4. Launch so the run survives your session ending:
   ```bash
   setsid nohup bash -c "echo \$\$ > '$RUN/train.pid'; exec $POL tools/train_stream_policy.py --recipe streamvln_stage1 --run-dir '$RUN'" \
       >> "$RUN/train.stdout.log" 2>&1 < /dev/null &
   ```
5. Within ~10 minutes, confirm in `$RUN/train.stdout.log`: a line with `"train_windows": 79814`,
   `"val_windows": 1736`, `"max_steps": 623`; `trainable 4,539,265,536 | mode full | start step 0 | windows/step 128`;
   a `"step": 0` validation row; then `"step": 1` with lr ≈ 8.5e-07 and `max_mem_gib` ≈ 67.7, and
   `"step": 2` with lr ≈ 1.28e-06 and `max_mem_gib` around 70–75. Each row's lr is the rate the *next*
   update will use: warmup adds 2e-5/47 per update, so update 1 trains at 4.3e-07 while its row shows
   8.5e-07, and rows show 2e-05 from step 46. Any other mismatch: stop the run (`kill $(cat $RUN/train.pid)`)
   and report.
   Expected, not a problem: transformers prints "The fast path is not available ... Falling back to torch
   implementation". Only the small `causal-conv1d` kernel is missing; the flash-linear-attention kernels
   are in use, which the speed shows (~1.3–1.5 windows/s; the pure-torch path would be ~4× slower).
6. If you cannot stay active for ~17 hours, report now (started, PID, expected finish time) and say that
   a later session can continue with this same prompt.

## 2. Monitoring (check every 30–60 minutes; do not poll continuously)

Look at `$RUN/progress.json` (step, `eta_s`, `updated_utc`), the last lines of `$RUN/log.jsonl`, and
`nvidia-smi -i 0`. Normal: one update every ~85–100 s (`windows_per_s` ≈ 1.3–1.5); `max_mem_gib` ≈ 74
(nvidia-smi ~77–80 GB); training loss starts around 3–4 per token and falls during warmup; `grad_norm` in
the hundreds early on is normal (it is clipped to 1); a validation row every 52 updates; each checkpoint
save adds ~1–2 minutes. Red flags: `progress.json` not updated for more than 20 minutes, the process gone
without `status.json`, or memory above 90 GB. Do not stop a healthy run.

## 3. Recovery

Read the end of `$RUN/train.stdout.log`.
- `FloatingPointError` (non-finite loss or gradient): do not resume. Report the step and the last rows.
- CUDA out of memory: check `nvidia-smi` for another process on GPU 0. If one caused it, wait until GPU 0 is
  free, then resume. If not, report; do not change any setting.
- Killed, node or driver trouble, or another transient error: resume.

Resume from the newest checkpoint folder that contains both `trainer_state.pt` and
`master_weights.safetensors` (the trainer keeps them only on the newest one; if none has them, the run
stopped before update 156: start over as described in **0**):
```bash
LAST=$(ls -d "$RUN"/checkpoints/step_* | sort | while read d; do [ -f "$d/trainer_state.pt" ] && [ -f "$d/master_weights.safetensors" ] && echo "$d"; done | tail -1)
setsid nohup bash -c "echo \$\$ > '$RUN/train.pid'; exec $POL tools/train_stream_policy.py --recipe streamvln_stage1 --run-dir '$RUN' --resume '$LAST'" \
    >> "$RUN/train.stdout.log" 2>&1 < /dev/null &
```
Resuming is exact (same data order, same weights). If loading that state fails because it was cut off
mid-save, use the previous checkpoint that still has both files. Stop and report after two failed
attempts, or if the same error recurs.

## 4. When it completes

1. Verify: `status.json` says complete with `"steps": 623, "samples_seen": 79744`;
   `checkpoints/step_000156`, `step_000312`, `step_000468` and `step_000623` each contain `config.json`,
   `generation_config.json` and `model.safetensors` (9,078,624,992 bytes); no `trainer_state.pt` or
   `master_weights.safetensors` remains anywhere in the run; `log.jsonl` has training rows for steps
   1–623 and validation rows at steps 0, 52, 104, …, 572 and 623 (13 distinct steps; after a resume the
   replayed steps appear twice, and the later rows count). Record the sha256 of every `model.safetensors`.
2. Load test (only now that GPU 0 is free; ~5–20 minutes): run the final checkpoint through the policy server
   on two group-C episodes:
   `$SIM tools/eval_stream_policy.py --run-dir "$RUN" --weights "$RUN/checkpoints/step_000623" --datasets r2r rxr --episodes 1 --workers 1 --out "$RUN/load_test_C2"`.
   It must finish with `summary.json`. Two episodes say nothing about quality; do not report them as a result.
3. Write the records, and edit nothing else:
   - `$RUN/RESULTS.md`: start and end time, duration, interruptions and resumes, the validation rows as a
     table over steps (`val_loss_per_token`, `val_action_acc`, per-class accuracy including STOP, and
     `val_acc_by_turn_offset`), final training loss, peak memory, throughput, checkpoint sizes and sha256,
     load-test outcome.
   - Append a "Run record" section to `trials/trial_1/steps/02_policy_sft/README.md`.
   - `trials/trial_1/steps.csv`: only the `02` row (status `complete`, completion time in UTC, evidence
     `runs/step2_qwen35_4b_streamvln`, a short note).
   - `trials/trial_1/trial.json`: only the step `"02"` entry (status, run, completion time, final
     checkpoint) and `training_budgets.stage_one` (add the run path, actual hours and final sha256). Keep
     the file's 2-space JSON formatting.
   - `ln -s ../runs/step2_qwen35_4b_streamvln/checkpoints/step_000623 trials/trial_1/checkpoints/stage_one_final`,
     plus `trials/trial_1/checkpoints/stage_one_final.json` with the run directory, checkpoint path, sha256,
     the run's `stream_config.json` and `processor/` paths, STOP logit bias 0, and "final checkpoint by
     rule (decided 2026-10-28)". Check it with `readlink -f`.

## Rules

- Do not modify code (`stream_policy/`, `tools/`), settings, data or conda environments; no pip installs or downloads.
- Use GPU 0 only. GPU 1 runs another team's services: never use it.
- Never run anything else on GPU 0 while training runs.
- Touch only processes you started, and kill only by the PID in `$RUN/train.pid`, never with `pkill -f`.
- Do not touch `trials/trial_0`, other runs, other users' jobs, git, or the `migration/` scripts.
- Never delete checkpoints, logs or data, even if disk runs low: stop and report instead.
- Do not start DAgger or stage two, and never evaluate on group E.
- When something does not match this prompt, stop and report what you saw instead of improvising.

## Report to the user (plain language)

Status (complete / running at step N with ETA / stopped and why); timing and any resumes; final checkpoint
path, size and sha256; validation at step 0 versus the end (loss per token, action accuracy, STOP accuracy,
accuracy at turn offset 0); peak memory and speed; load-test outcome; anything unusual. Note that the action
history is visible in this format: a model that ignores the views can reach about 73% offline accuracy
(74.3% at turn offset 0), so validation accuracy near those numbers does not show that it uses the views,
and training or validation numbers are not navigation results.
