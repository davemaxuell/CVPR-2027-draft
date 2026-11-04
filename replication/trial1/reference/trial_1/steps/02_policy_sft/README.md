# Step 2 — stage one: expert supervision (ready; launch on the user's go-ahead)

StreamVLN's stage one on the group-A expert renders: one sample per 32-action window, every action and
turn end supervised, StreamVLN augmentation. Decided 28 October 2026: **full fine-tuning, one pass, action
history visible** (StreamVLN as is). `--recipe streamvln_stage1` applies it with StreamVLN's other
stage-one settings:

| Setting | Value |
| --- | --- |
| Trainable | every weight (4,539,265,536), fp32 master weights under bf16 autocast |
| Optimiser | fused AdamW, lr 2e-5, vision tower 5e-6, no weight decay, clip 1.0 |
| Schedule | 47-update linear warmup, then cosine to 1.85e-5 (HF Trainer shape) |
| Batch | micro-batch 1 × 128 accumulation; 623 updates = 79,744 of 79,814 windows, none twice |
| Data | 21,330 group-A expert episodes ([audit](../01_experiment/data_audit.json)), StreamVLN augmentation; validation every 52 updates on 256 group-C windows |
| Checkpoints | updates 156, 312, 468, 623 (9.1 GB each); the newest keeps a 54 GB resume state until the end |
| Cost | ≈ 17 h on one H100, ≈ 74 GiB peak |

```bash
cd "[OURS]/baseline-v0"; export HF_HOME=/data/shared/hf_cache
CUDA_VISIBLE_DEVICES=0 /home/bufsgpu/yes/envs/rolevln-qwen35/bin/python tools/train_stream_policy.py \
    --recipe streamvln_stage1 --run-dir runs/trial_1/step2_<name>
# continue after an interruption (exact): the same command + --resume runs/trial_1/step2_<name>/checkpoints/<newest>
```

To hand the run to another agent, give it [AGENT_PROMPT.md](AGENT_PROMPT.md): it launches, monitors,
resumes after a crash and records the result, and a later session can continue from where it stopped.

GPU 0 must be otherwise idle while it runs (Habitat evaluation can put its policy servers on GPU 1 with
`--policy-gpu 1`). `progress.json` in the run directory gives the step and ETA.

Checkpoint (decided 28 October): **the final one, update 623**, as StreamVLN. Updates 156, 312 and 468
are kept for diagnosis only and never replace it. Record here the run and its final checkpoint.

Reading the validation log: with the action history visible, a model that ignores the views can reach
73% offline accuracy, and 74.3% at turn offset 0 (the first action after a new view), so accuracy near
those numbers says nothing about grounding. Suggested gates before DAgger, from the trial_0 diagnosis:
offline STOP accuracy above zero, a view/history swap probe, and C collision and spin rates.

## Launch log

- 2026-10-28 07:01–07:05 UTC: the first launch (agent) was stopped after update 1 because AGENT_PROMPT.md
  expected the logged learning rate to be 4.3e-07. The prompt was wrong by one update: each log row shows
  the rate the next update will use, so the correct row-1 value is 8.5e-07, which is what the run logged.
  Everything else matched (79,814 windows, 623 updates, 4,539,265,536 trainable weights, 67.7 GiB at the
  first optimizer step, 1.49 windows/s). Prompt fixed; the folder is kept as
  `runs/trial_1/step2_qwen35_4b_streamvln.stopped-20261028T070502Z` (no checkpoint). Its step-0 validation
  of the pretrained model: loss 3.532 per token, action accuracy 50.8%, STOP 18.6%, turn offset 0 29.8%.

<!-- dagger-supervisor-stage-one-c64 -->
## Diagnostic C64 screen

Final update 623: macro success 53.12%, macro SPL 0.44819, mean collisions 44.828, mean steps 129.625, STOP rate 90.62%. Endings: `{"STOP": 58, "step_limit": 6}`.

Evidence: `runs/step2_qwen35_4b_streamvln/eval_C64_step_000623/summary.json`. These 32 R2R + 32 RxR development episodes are diagnostic, not a result; the checkpoint remains the final one by rule.

## Organization reconciliation (2026-11-03)

This stage completed at 2026-10-28T22:51:42.535098+00:00 with 623 updates; see the preserved `runs/step2_qwen35_4b_streamvln/status.json`. Any earlier planned/not-run wording describes preparation history. No training is repeated. Trial 1 uses only group-A R2R-CE/RxR-CE English-guide expert data.
