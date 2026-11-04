You are taking over one task: **trial_1's DAgger stage**. Wait until stage one (a full fine-tune of
Qwen3.5-4B, run by another agent) has finished, then run StreamVLN's DAgger round with that policy, train
stage two on expert + DAgger data, and record the outcome. Everything is designed, implemented and tested.
Execute it exactly, keep it healthy, and record what happened. Do not redesign, tune or "improve" anything.
The whole stage takes roughly two days of wall time after stage one finishes.

## What is fixed (decided, do not change)

- Input: stage one's run `runs/trial_1/step2_qwen35_4b_streamvln`; its final checkpoint `step_000623` is the
  collecting policy (rule decided 2026-10-28). Another agent owns that run: never resume, restart, stop or
  modify it. You only read it, and add the diagnostic screen folder of step 2.
- DAgger round (StreamVLN's collection): one round over every validated group-A episode (7,489 R2R +
  13,844 RxR), β = 0, StreamVLN's takeover and keep rule (the collector's defaults), seed 1, targets = the
  executed actions. Command: `tools/collect_stream_dagger.py --run-dir <stage one> --weights <its final>
  --episodes 0 --workers 6 --out <ROUND>`.
- Stage two (StreamVLN's stage two): `tools/train_stream_policy.py --recipe streamvln_stage2 --run-dir <RUN2>
  --init <stage-one final> --train-root <expert_groupA> <ROUND>`: full fine-tuning, one pass over the expert
  and DAgger windows together, lr 2e-5 (vision 5e-6), 3% warmup then cosine to 0, augmentation, no
  co-training data. Its final checkpoint is the result.
- Diagnostics, never used to choose anything: a closed-loop group-C screen (32 R2R + 32 RxR episodes) of the
  stage-one final before collecting, and of the stage-two final at the end.
- Background, only if you need it: `trials/trial_1/steps/03_dagger/README.md`, `trials/trial_1/README.md`,
  `trials/trial_1/docs/StreamVLN_Recipe.md`, and stage one's prompt `trials/trial_1/steps/02_policy_sft/AGENT_PROMPT.md`.

## Environment

```bash
cd "/data/team_a/dave-workspace/RoleVLN/[OURS]/baseline-v0"   # always quote: the path contains [OURS]
RUN1=runs/trial_1/step2_qwen35_4b_streamvln                     # stage one (read only)
CK1=$RUN1/checkpoints/step_000623
ROUND=/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/trial_1/dagger_r1_qwen35_4b_streamvln
RUN2=runs/trial_1/step3_qwen35_4b_streamvln_dagger_r1
EXPERT_A=/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupA
POL=/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python           # training env (Python 3.12)
SIM=/home/bufsgpu/yes/envs/rolevln/bin/python                  # Habitat env (Python 3.9): collection and screens
export HF_HOME=/data/shared/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
gpu0_idle() { [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]; }
```

This must run on the H100 server (GPU 0 is an "NVIDIA H100 NVL", ~95,830 MiB). Habitat renders only on GPU 0,
so every step here uses GPU 0, one step at a time. No network is needed.

## How to proceed

Work through steps 1–6 in order. Each step says when it is done: skip steps that are done. If a step's
process is still running, monitor it; if it stopped before it was done, follow its recovery notes. A later
session can always continue with this same prompt. Check running work every 30–60 minutes; never poll
continuously.

### 1. Wait for stage one

Done when `$RUN1/status.json` says `"complete"` with `"steps": 623`, `$CK1/model.safetensors` exists
(9,078,624,992 bytes), and `gpu0_idle` holds at two checks 10 minutes apart (stage one's agent runs a short
load test after training).
- While waiting, `$RUN1/progress.json` gives stage one's step and ETA. If you cannot keep waiting, report the
  ETA and stop.
- If stage one's process is gone without `status.json` for more than 2 hours, or its log ends in an error,
  stop and report. Never resume or restart stage one yourself.
- Sanity gate: in `$RUN1/log.jsonl` the last validation row's `val_loss_per_token` must be lower than the
  step-0 row's. If not, stop and report.

### 2. Diagnostic screen of stage one (~20–40 minutes)

Done when `$RUN1/eval_C64_step_000623/summary.json` exists.
`$SIM tools/eval_stream_policy.py --run-dir "$RUN1" --weights "$CK1" --episodes 32 --workers 4 --out "$RUN1/eval_C64_step_000623"`
It runs in the foreground; if interrupted, run the same command again (it resumes). Note `macro_success`,
`macro_spl` and the `overall` collisions, steps and stop rate: the baseline for step 6.

### 3. DAgger collection (~12–24 hours)

Done when `$ROUND/summary.json` says `"status": "complete"`.
- Before the first launch: `df -h /data/team_a` shows at least 300 GB free, `gpu0_idle` holds, and
  `pgrep -af "collect_stream_dagger.py|train_stream_policy.py|eval_stream_policy.py"` prints nothing.
- Launch detached. This is also the resume command: the collector keeps finished episodes and continues.
  ```bash
  mkdir -p "$ROUND"
  setsid nohup bash -c "echo \$\$ > '$ROUND/collect.pid'; exec $SIM tools/collect_stream_dagger.py --run-dir '$RUN1' --weights '$CK1' --episodes 0 --workers 6 --out '$ROUND'" \
      >> "$ROUND/collect.stdout.log" 2>&1 < /dev/null &
  ```
- Within ~10 minutes: `$ROUND/collection_config.json` lists 7,489 R2R and 13,844 RxR selected episodes, and
  `$ROUND/progress.json` shows `completed_episodes` rising. GPU 0 holds about 66 GiB (six policy servers of
  ~11 GiB). Episodes appear atomically; an occasional `aborted` episode (no expert path) is normal.
- Red flags: GPU 0 memory above 88 GiB, `progress.json` unchanged for more than 30 minutes, or
  `collect.stdout.log` ending in an error. To stop cleanly: `kill -TERM $(cat "$ROUND/collect.pid")`, which
  stops every worker and policy server.
- Recovery: read the end of `collect.stdout.log` and `$ROUND/workers/*.log`, then relaunch the same command.
  If memory was the problem, relaunch with `--workers 4` (allowed: the worker count is not part of the
  round's identity). Stop and report after two failed relaunches, or if an error repeats.

### 4. Audit the round (~5 minutes, CPU)

Done when `$ROUND/data_audit.json` exists with `"passed": true`.
`$POL tools/audit_stream_data.py --dagger-root "$ROUND" --out "$ROUND/data_audit.json"`
It also re-checks the expert data. If it fails, stop and report. Note the kept episodes and the round's
window count (`dagger` → the round → `windows` → `windows`).

### 5. Stage-two training (~1 day)

Done when `$RUN2/status.json` says `"complete"`.
- Before the first launch: `gpu0_idle`, at least 150 GB free, and nothing else training.
- Launch detached:
  ```bash
  mkdir -p "$RUN2"
  setsid nohup bash -c "echo \$\$ > '$RUN2/train.pid'; exec $POL tools/train_stream_policy.py --recipe streamvln_stage2 --run-dir '$RUN2' --init '$CK1' --train-root '$EXPERT_A' '$ROUND'" \
      >> "$RUN2/train.stdout.log" 2>&1 < /dev/null &
  ```
- Within ~10 minutes `train.stdout.log` shows `"recipe": "streamvln_stage2"`, `"mode": "full"`,
  `train_windows` = 79,814 + the round's windows from step 4, `max_steps` = train_windows // 128, then
  `trainable 4,539,265,536`, a step-0 validation row, and step 1 with lr ≈ 2 × 2e-5 / ceil(0.03 × max_steps)
  (each row's lr is the rate the *next* update will use; update 1 itself trains at half that). Any other
  mismatch: stop (`kill $(cat "$RUN2/train.pid")`) and report. Expected, not a problem: transformers prints
  "The fast path is not available ... Falling back to torch implementation"; only the small `causal-conv1d`
  kernel is missing, and the flash-linear-attention kernels are in use (~1.3–1.5 windows/s).
- Monitoring, as for stage one: one update every ~85–100 s; `max_mem_gib` ≈ 74; a validation row every 52
  updates; checkpoints every 156 updates and at the end. Red flags: `progress.json` unchanged for more than
  20 minutes, the process gone without `status.json`, memory above 90 GB.
- Recovery: `FloatingPointError` → do not resume, report. CUDA out of memory → resume only if another
  process on GPU 0 caused it, otherwise report. Other crashes → resume from the newest checkpoint folder
  that has both `trainer_state.pt` and `master_weights.safetensors`, using the launch command without
  `--init` and with `--resume <that folder>` (the run keeps its starting weights on record). Resuming is
  exact. If no folder has both files yet (it stopped before update 156), there is nothing to resume: set
  the run aside with `mv "$RUN2" "$RUN2.stopped-$(date -u +%Y%m%dT%H%M%SZ)"` and launch again from scratch.
  Stop and report after two failed attempts.

### 6. Finish: verify, screen, record, report

1. Verify: `$RUN2/status.json` is complete; there is a checkpoint folder every 156 updates plus the final one
   (`step_` + max_steps as six digits), each with a `model.safetensors` of 9,078,624,992 bytes; no
   `trainer_state.pt` or `master_weights.safetensors` remains. Record the final checkpoint's sha256.
2. Diagnostic screen of the stage-two final:
   `$SIM tools/eval_stream_policy.py --run-dir "$RUN2" --weights "$RUN2/checkpoints/<final>" --episodes 32 --workers 4 --out "$RUN2/eval_C64_final"`.
3. Records (re-read each file right before writing it, and edit nothing else):
   - `$RUN2/RESULTS.md`: the round (episodes, kept, aborted, action sources, takeovers, duration), stage two
     (windows, updates, duration, resumes, the validation rows over steps, peak memory, checkpoint sha256),
     and both group-C screens side by side.
   - Append a "Run record" section to `trials/trial_1/steps/03_dagger/README.md`, and a short "Diagnostic
     C64 screen" section to `trials/trial_1/steps/02_policy_sft/README.md`.
   - `trials/trial_1/steps.csv`: only the `03` row (status `complete`, completion time in UTC, evidence, note).
   - `trials/trial_1/trial.json`: only the step `"03"` entry and `training_budgets.dagger` and
     `training_budgets.stage_two` (replace `"status": "planned"` with the actual counts, hours, run paths and
     final sha256). Keep its 2-space JSON formatting.
   - `ln -s ../runs/step3_qwen35_4b_streamvln_dagger_r1/checkpoints/<final> trials/trial_1/checkpoints/stage_two_final`
     plus `trials/trial_1/checkpoints/stage_two_final.json` (run, checkpoint, sha256, the run's
     `stream_config.json` and `processor/` paths, STOP logit bias 0, "final checkpoint by rule"). Check
     with `readlink -f`.

## Rules

- Do not modify code (`stream_policy/`, `tools/`), settings, data or conda environments; no pip installs or downloads.
- Use GPU 0 only, one job at a time. GPU 1 runs another team's services: never use it.
- Touch only processes you started; stop them only through their pid files, never with `pkill -f`.
- Stage one's run is read-only for you, apart from the screen folder of step 2.
- Do not touch `trials/trial_0`, other runs, other users' jobs, git, or the `migration/` scripts.
- The round's frames are Matterport-derived: never copy them outside `data/processed`.
- Never delete checkpoints, rounds, logs or data, even if disk runs low: stop and report instead.
- Never evaluate on group E, and do not start a second DAgger round.
- When something does not match this prompt, stop and report what you saw instead of improvising.

## Report to the user (plain language)

Where the stage stands (step and ETA, or done); the round (episodes, how many kept, how often the expert had
to take over, duration); stage two (windows, updates, duration, any resumes, validation at the start versus
the end); both group-C screens side by side (success, SPL, collisions, how episodes ended), saying that 64
development episodes are a diagnostic, not a result; the final checkpoint's path and sha256; anything
unusual. Validation accuracy near 73% (74.3% at turn offset 0) does not show that the policy uses the views:
the action history alone predicts that much.
