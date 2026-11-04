# Trial 1: R2R-CE / RxR-CE navigation

The completed original model and its weights remain in Trial 1. Checkpoints **623** and **990** are preserved, with their original training data, logs and evaluations.

| Experiment | Initialization and method | Training coverage | Status |
|---|---|---|---|
| Original stage one, 623 | Pretrained Qwen3.5-4B; one epoch full SFT | Group-A: 7,489 R2R + 13,841 usable RxR English-guide expert episodes | Complete |
| Original DAgger round | Frozen 623; one collection round | Same group-A scope; 11,660 retained trajectories | Complete |
| Original stage two, 990 | 623; one epoch full SFT on expert + original DAgger | 126,721 windows; 990 updates | Complete |
| Full-data continuation from 990 | 990; fresh optimizer, one additional epoch of expert SFT | All official training scenes; 10,819 R2R + 19,402 usable RxR English-guide episodes | Launched; live status linked below |
| Stopped full-data restart from pretrained | Fresh stage-one attempt | Same full training sources | Stopped at 44/857; held, no recovery checkpoint |

The continuation is an **additional fine-tuning stage**. It uses the established native Qwen3.5 encoding and StreamVLN-style full-fine-tuning settings: effective batch 128, LM LR 2e-5, vision LR 5e-6, weight decay 0, 3% warmup and cosine decay, BF16, gradient checkpointing, 32-step windows, eight history views, four actions per response and token budget 16,384. It reuses no DAgger trajectories in this pass and schedules no additional collection.

Full training sources contain 10,819 R2R and 19,996 RxR English-guide episodes. All were processed. The established validity checks exclude 594 RxR episodes (450 failed/truncated, 138 action drift, six too short), leaving 30,221 usable episodes and 109,784 windows. The unchanged one-pass full-batch sampler uses 857 updates / 109,696 windows, with an 88-window tail. English-guide scope is en-US/en-IN; official validation and unseen scenes remain excluded from training. The LookStep comparison confirms the source splits and language scope; processed-example inclusion differs and is documented in the audit.

[Continuation live status](continuation_from_990_20261103/STATUS.md) · [Authorization and recipe](continuation_from_990_20261103/authorization.json) · [Data audit](continuation_from_990_20261103/evidence/expert_data_audit.json) · [Checkpoint catalog](checkpoints/ACTIVE_CHECKPOINTS.json)

Automatic order: continuation SFT from 990 → checkpoint verification → fresh full R2R val-seen (778) → fresh full R2R val-unseen (1,839) → comparison with original 990. The final epoch checkpoint is used by rule; val-unseen is report-only. New results use their own run and evaluation paths.

The preserved original 990 baseline achieved R2R seen SR 47.94% / SPL 39.97%, and unseen SR 44.43% / SPL 35.51%. Full RxR evaluation has not been completed; the old RxR screen was diagnostic only.

EnvDrop, ScaleVLN, LLaVA-Video, ScanQA, SQA3D and MMC4 belong to Trial 2, whose training remains held. MMC4 downloading continues independently. The custom Saved places experiment remains separate. All authorized GPU work uses GPU 0 only.

## Authorized strategy comparison

The user selected a four-approach comparison on 2026-11-03: keep current F, then run E, G and J sequentially. E starts from 623 on full experts + original DAgger; G starts from 990 on that same mixture; J freezes F for one new collection round and trains on full experts + those fresh corrections. F is not repeated. [Protocol and comparisons](strategy_comparison_20261103/README.md) · [Live suite status](strategy_comparison_20261103/STATUS.md).
