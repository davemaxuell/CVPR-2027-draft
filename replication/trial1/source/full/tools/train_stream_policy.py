"""Train the streaming policy on StreamVLN dialogue windows (env rolevln-qwen35).

Stage one is expert supervision on group-A renders; stage two adds DAgger rounds (--train-root
expert and DAgger roots, optionally --source-weights) and starts from a stage-one checkpoint (--init).
The loss is masked multi-turn cross-entropy over every supervised token (each action and each
assistant turn end), normalised by the supervised tokens of the optimizer step.

trial_1 decisions (2026-10-28): full fine-tuning, one pass, as StreamVLN. `--recipe streamvln_stage1`
sets StreamVLN's stage one (scripts/streamvln_train_slurm.sh @ e48f6ff7): every weight, lr 2e-5 with
vision 5e-6, no weight decay, clip 1, effective batch 128, one pass, warmup 7.5% then a cosine to a
1.85e-5 floor, StreamVLN augmentation. `streamvln_stage2` is its stage two: cosine to 0 after 3%
warmup, from --init. Explicit flags override recipe values. One pass is floor(windows / batch)
updates, as with StreamVLN's drop-last loader, so no window is seen twice.

The training stream (order, augmentation, conjunctions) depends on --seed alone, and the newest
checkpoint keeps the optimizer state (+ fp32 master weights in llm/full modes), so --resume continues
exactly. A completed run deletes that resume state unless --keep-final-state.

Run directory:
    stream_config.json, train_args.json, data_summary.json, processor/, log.jsonl, progress.json
    checkpoints/step_{N}/  weights (adapter or bf16 model) [+ trainer_state.pt, master_weights.safetensors]

    CUDA_VISIBLE_DEVICES=0 python tools/train_stream_policy.py --recipe streamvln_stage1 --run-dir runs/trial_1/step2_<name>
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # variable-length windows

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stream_policy.config import ACTION_NAMES, StreamConfig, simulator_settings  # noqa: E402
from stream_policy.data import MixtureSampler, Sources, TrajectoryWindows  # noqa: E402
from stream_policy.qwen35 import MODES, Qwen35Inputs, Qwen35Policy  # noqa: E402

DATA = "/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0"
EXPERT_A, EXPERT_C = f"{DATA}/expert_groupA", f"{DATA}/expert_groupC_600"
STREAMVLN = dict(mode="full", epochs=1.0, batch_size=1, grad_accum=128, lr=2e-5, vision_lr=5e-6, weight_decay=0.0,
                 clip=1.0, augment=True, log_every=1, eval_every=52, eval_windows=256, eval_initial=True, save_every=156)
RECIPES = {
    "streamvln_stage1": dict(STREAMVLN, schedule="cosine_floor", min_lr=1.85e-5, warmup_frac=0.075),
    "streamvln_stage2": dict(STREAMVLN, schedule="cosine", warmup_frac=0.03),
}


def schedule(kind, steps, warmup, floor):
    """The Hugging Face Trainer's linear warmup, then a cosine over the remaining updates (StreamVLN's
    cosine_with_min_lr / cosine); the first update gets lr/warmup instead of HF's lr = 0."""
    def factor(s):
        if s < warmup:
            return (s + 1) / warmup
        if kind == "constant":
            return 1.0
        cosine = 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
        return floor + (1 - floor) * cosine if kind == "cosine_floor" else cosine
    return factor


@torch.no_grad()
def evaluate(policy, loader, cfg):
    """Supervised-token loss, action accuracy overall, per class and per offset inside a turn."""
    policy.model.eval()
    loss, tokens = 0.0, 0
    confusion = np.zeros((4, 4), dtype=np.int64)
    offset = np.zeros((cfg.actions_per_turn, 2), dtype=np.int64)
    for batch in loader:
        batch = {k: v.to(policy.device) for k, v in batch.items()}
        logits = policy.supervised_logits(batch)
        loss += float(torch.nn.functional.cross_entropy(logits, batch["label_tok"], reduction="sum"))
        tokens += int(batch["label_tok"].numel())
        action = batch["label_action"] >= 0
        truth = batch["label_action"][action].tolist()
        guess = logits[action][:, policy.action_ids].argmax(1).tolist()
        steps = batch["label_step"][action].tolist()
        for y, p, step in zip(truth, guess, steps):
            confusion[y, p] += 1
            o = step % cfg.window_steps % cfg.actions_per_turn
            offset[o] += (y == p, 1)
    policy.model.train()
    per_class = {w: float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None for i, w in enumerate(ACTION_NAMES)}
    return {"val_loss_per_token": loss / max(tokens, 1), "val_action_acc": float(np.trace(confusion) / max(confusion.sum(), 1)),
            "val_per_class_acc": per_class, "val_acc_by_turn_offset": [float(a / n) if n else None for a, n in offset],
            "val_confusion": confusion.tolist(), "val_actions": int(confusion.sum())}


RESUME_FILES = ("trainer_state.pt", "master_weights.safetensors")


def save_checkpoint(path, policy, optimizer, scheduler, step, samples_seen, keep_states, run_dir):
    path.mkdir(parents=True, exist_ok=True)
    policy.save(path)
    if policy.mode in ("llm", "full"):
        policy.save_master(path / "master_weights.safetensors")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step,
                "samples_seen": samples_seen, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, path / "trainer_state.pt")
    states = sorted((run_dir / "checkpoints").glob("step_*/trainer_state.pt"))
    for old in states[:max(0, len(states) - keep_states)]:
        for name in RESUME_FILES:  # weights stay; only the newest resume state is kept
            (old.parent / name).unlink(missing_ok=True)


def parser(recipe):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=sorted(RECIPES), help="preset settings; explicit flags override them")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=[m for m in MODES if m != "frozen"])
    ap.add_argument("--train-root", nargs="+", default=[EXPERT_A])
    ap.add_argument("--source-weights", nargs="*", type=float, default=None, help="one per root; default: shuffled concatenation")
    ap.add_argument("--val-root", nargs="*", default=[EXPERT_C])
    ap.add_argument("--datasets", nargs="+", default=["r2r", "rxr"])
    ap.add_argument("--label-source", default="executed", choices=["executed", "expert"])
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False, help="StreamVLN data_augmentation transforms")
    ap.add_argument("--init", default="", help="start from these weights (adapter or full model directory)")
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer updates (or use --epochs)")
    ap.add_argument("--epochs", type=float, default=0.0, help="passes over the training windows")
    ap.add_argument("--batch-size", type=int, default=1, help="micro-batch of windows")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--vision-lr", type=float, default=5e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--schedule", default="cosine_floor", choices=["cosine_floor", "cosine", "constant"])
    ap.add_argument("--min-lr", type=float, default=1.85e-5, help="floor of cosine_floor, for --lr")
    ap.add_argument("--warmup-frac", type=float, default=0.075)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--eval-initial", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--keep-trainer-states", type=int, default=1, help="newest checkpoints that keep their resume state")
    ap.add_argument("--keep-final-state", action="store_true", help="keep the resume state after the last update")
    ap.add_argument("--max-episodes", type=int, default=0, help="per dataset and root, for debugging")
    ap.add_argument("--max-deviation", type=float, default=0.5)
    ap.add_argument("--resume", default="", help="checkpoint directory of this run to continue from")
    if recipe:
        ap.set_defaults(**RECIPES[recipe])
    return ap


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--recipe", choices=sorted(RECIPES))
    ap = parser(pre.parse_known_args()[0].recipe)
    args = ap.parse_args()
    if args.mode is None:
        ap.error("choose --mode or a --recipe")
    if args.resume and args.init:
        ap.error("--resume continues this run; --init starts a new one")
    if args.max_steps:
        args.epochs = 0.0  # an explicit update count replaces the recipe's pass
    if bool(args.max_steps) == bool(args.epochs):
        ap.error("give exactly one of --max-steps and --epochs")
    if args.source_weights is not None and len(args.source_weights) != len(args.train_root):
        ap.error("--source-weights needs one weight per --train-root")

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # StreamVLN --tf32 True
    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    sim = simulator_settings()
    cfg = StreamConfig(forward_step_m=sim["forward_step_m"], turn_angle_deg=sim["turn_angle_deg"])
    if (run / "stream_config.json").exists() and StreamConfig.load(run / "stream_config.json") != cfg:
        raise ValueError("this run directory holds a different stream config")
    cfg.save(run / "stream_config.json")
    inputs = Qwen35Inputs(cfg)
    inputs.processor.save_pretrained(run / "processor")

    sources = [TrajectoryWindows([root], cfg, inputs, args.datasets, args.label_source, args.max_deviation,
                                 args.max_episodes, args.augment) for root in args.train_root]
    if any(len(s) == 0 for s in sources):
        raise ValueError("every --train-root must contribute windows")
    train = Sources(sources, args.seed)
    val = TrajectoryWindows(args.val_root, cfg, inputs, args.datasets, "executed", args.max_deviation,
                            args.max_episodes, fixed_conjunctions=True) if args.val_root else None
    samples_per_step = args.batch_size * args.grad_accum
    max_steps = args.max_steps or max(1, int(args.epochs * len(train)) // samples_per_step)
    vars(args)["resolved_max_steps"] = max_steps
    if args.resume:
        previous = json.loads((run / "train_args.json").read_text())
        args.init = previous.get("init", "")  # a resumed run keeps its starting weights on record; --resume supersedes them
        changed = {k for k in previous if k not in ("resume", "workers", "val_workers") and previous[k] != vars(args).get(k)}
        if changed:
            raise ValueError(f"resume must keep the run's settings; changed: {sorted(changed)}")
    (run / "train_args.json").write_text(json.dumps(vars(args), indent=1) + "\n")
    summary = {"train_windows": len(train), "sources": {root: s.summary() for root, s in zip(args.train_root, sources)},
               "val": val.summary() if val else None, "max_steps": max_steps, "samples_per_step": samples_per_step,
               "passes": max_steps * samples_per_step / len(train)}
    (run / "data_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"recipe": args.recipe, "mode": args.mode, "train_windows": len(train),
                      "val_windows": len(val) if val else 0, "max_steps": max_steps, "passes": summary["passes"]}), flush=True)

    weights = args.resume or args.init or None
    policy = Qwen35Policy(cfg, inputs.action_ids, mode=args.mode, weights=weights, lora_rank=args.lora_rank,
                          lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          gradient_checkpointing=not args.no_grad_ckpt)
    optimizer = torch.optim.AdamW(policy.parameter_groups(args.lr, args.vision_lr, args.weight_decay), fused=True)
    warmup = max(1, math.ceil(args.warmup_frac * max_steps))  # HF: ceil(warmup_ratio * steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(args.schedule, max_steps, warmup, args.min_lr / args.lr))
    step, samples_seen = 0, 0
    if args.resume:
        resume = Path(args.resume)
        if (resume / "master_weights.safetensors").exists():
            policy.load_master(resume / "master_weights.safetensors")
        elif args.mode in ("llm", "full"):
            raise FileNotFoundError(f"{resume} has no master weights; resume from the newest checkpoint")
        state = torch.load(resume / "trainer_state.pt", weights_only=False)
        optimizer.load_state_dict(state["optimizer"]), scheduler.load_state_dict(state["scheduler"])
        step, samples_seen = state["step"], state["samples_seen"]
        torch.set_rng_state(state["torch_rng"]), torch.cuda.set_rng_state(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"]), random.setstate(state["python_rng"])

    sampler = MixtureSampler([len(s) for s in sources], args.seed, args.source_weights, start=samples_seen)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                        collate_fn=inputs.collate, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers else None)
    val_loader = None
    if val:
        picks = np.linspace(0, len(val) - 1, min(len(val), args.eval_windows)).astype(int).tolist()
        val_loader = DataLoader(Subset(val, picks), batch_size=args.batch_size, num_workers=args.val_workers,
                                collate_fn=inputs.collate, persistent_workers=args.val_workers > 0)
    params = policy.trainable_parameters()
    print(f"trainable {sum(p.numel() for p in params):,} | mode {args.mode} | start step {step} | "
          f"windows/step {samples_per_step}", flush=True)
    log = open(run / "log.jsonl", "a")

    def record(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n"), log.flush()

    if val_loader and args.eval_initial and step == 0:
        record({"step": 0, **evaluate(policy, val_loader, cfg)})
    batches = iter(loader)
    policy.model.train()
    began = run_began = time.time()
    first_step, window_samples, window_tokens = step, 0, 0
    while step < max_steps:
        micro = [next(batches) for _ in range(args.grad_accum)]
        total = sum(int(b["label_tok"].numel()) for b in micro)
        step_loss, correct, actions = 0.0, 0, 0
        for b in micro:
            b = {k: v.to(policy.device, non_blocking=True) for k, v in b.items()}
            loss, _, action_logits = policy.loss_sum(b)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step + 1}")
            (loss / total).backward()
            step_loss += float(loss.detach())
            mask = b["label_action"] >= 0
            correct += int((action_logits[mask].argmax(1) == b["label_action"][mask]).sum())
            actions += int(mask.sum())
            window_tokens += int(b["attention_mask"].sum())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at optimizer step {step + 1}")
        optimizer.step(), scheduler.step(), optimizer.zero_grad(set_to_none=True)
        step += 1
        samples_seen += samples_per_step
        window_samples += samples_per_step
        if step % args.log_every == 0 or step == 1:
            dt, elapsed = time.time() - began, time.time() - run_began
            record({"step": step, "loss_per_token": step_loss / total, "train_action_acc": correct / max(actions, 1),
                    "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm, "samples_seen": samples_seen,
                    "epoch": samples_seen / len(train), "windows_per_s": window_samples / dt, "tokens_per_s": window_tokens / dt,
                    "max_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30})
            eta = elapsed / (step - first_step) * (max_steps - step)
            (run / "progress.json").write_text(json.dumps({"step": step, "max_steps": max_steps, "samples_seen": samples_seen,
                                                           "elapsed_s_this_run": elapsed, "eta_s": eta,
                                                           "updated_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
            began, window_samples, window_tokens = time.time(), 0, 0
        if val_loader and (step % args.eval_every == 0 or step == max_steps):
            record({"step": step, **evaluate(policy, val_loader, cfg)})
        if step % args.save_every == 0 or step == max_steps:
            save_checkpoint(run / "checkpoints" / f"step_{step:06d}", policy, optimizer, scheduler, step,
                            samples_seen, args.keep_trainer_states, run)
    log.close()
    if not args.keep_final_state:
        for name in RESUME_FILES:  # stage two starts from the weights with a fresh optimizer, as in StreamVLN
            (run / "checkpoints" / f"step_{max_steps:06d}" / name).unlink(missing_ok=True)
    (run / "status.json").write_text(json.dumps({"status": "complete", "steps": max_steps, "samples_seen": samples_seen,
                                                 "completed_utc": datetime.now(timezone.utc).isoformat()}) + "\n")


if __name__ == "__main__":
    main()
