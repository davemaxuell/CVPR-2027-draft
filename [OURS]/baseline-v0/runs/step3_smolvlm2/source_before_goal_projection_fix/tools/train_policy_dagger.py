"""Step 3: one DAgger refinement round, initialized from the selected Step 2 adapter.

Masked autoregressive cross-entropy over the target action tokens only, normalised by the number
of target tokens in each optimizer step (spec section 2). Only decoder LoRA adapters train.

Run directory layout:
    policy_config.json, train_args.json, data_summary.json, processor/, log.jsonl
    checkpoints/step_{N}/  adapter weights + trainer_state.pt (optimizer, scheduler, sampler, RNG)

    CUDA_VISIBLE_DEVICES=0 python tools/train_policy_dagger.py --run-dir runs/dagger/train --dagger-root ... --init-adapter ... --max-steps 2048
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from policy.common import ACTION_WORDS, PolicyConfig  # noqa: E402
from policy.data import ExpertSteps  # noqa: E402
from policy.dagger_data import BalancedSourceSampler, DaggerSteps  # noqa: E402
from tools.train_policy_sft import evaluate, save_checkpoint as save_sft_checkpoint  # noqa: E402
from policy.smolvlm2 import SmolVLM2Inputs, SmolVLM2Policy, default_config  # noqa: E402
from sim.config import load_manifest, scene_group  # noqa: E402

EXPERT_A = "/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupA"


def save_checkpoint(path, *args):
    from pathlib import Path
    import shutil
    final = Path(path)
    temporary = final.with_name("." + final.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    save_sft_checkpoint(str(temporary), *args)
    temporary.rename(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-root", nargs="+", default=[EXPERT_A])
    ap.add_argument("--dagger-root", nargs="+", required=True)
    ap.add_argument("--init-adapter", required=True, help="selected Step 2 adapter; fresh optimizer and schedule")
    ap.add_argument("--init-run", default="runs/step2_smolvlm2/train", help="parent policy config")
    ap.add_argument("--val-root", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="+", default=["r2r", "rxr"])
    ap.add_argument("--max-steps", type=int, required=True, help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=16, help="micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--past-frames", type=int, default=7)
    ap.add_argument("--frame-sampling", default="uniform", choices=["uniform", "recent"])
    ap.add_argument("--recent-actions", type=int, default=16)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=125)
    ap.add_argument("--eval-initial", action="store_true", help="record offline validation before the first update")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-episodes", type=int, default=0, help="per dataset, for debugging")
    ap.add_argument("--max-deviation", type=float, default=0.5, help="exclude experts drifting this far (m) from the stored path")
    ap.add_argument("--resume", default="", help="checkpoint directory to continue from")
    args = ap.parse_args()

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    sim = load_manifest()["simulator"]
    cfg = default_config(past_frames=args.past_frames, frame_sampling=args.frame_sampling,
                         recent_actions=args.recent_actions, forward_step_m=sim["forward_step_m"],
                         turn_angle_deg=sim["turn_angle_deg"])
    parent_config = PolicyConfig.load(os.path.join(args.init_run, "policy_config.json"))
    if cfg != parent_config:
        raise ValueError("DAgger must preserve the parent policy prompt/window/action configuration")
    cfg.save(os.path.join(args.run_dir, "policy_config.json"))
    with open(os.path.join(args.run_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    inputs = SmolVLM2Inputs(cfg)
    inputs.processor.save_pretrained(os.path.join(args.run_dir, "processor"))
    expert = ExpertSteps(args.train_root, cfg, args.datasets, args.max_episodes, args.max_deviation)
    dagger = DaggerSteps(args.dagger_root, cfg, set(scene_group("A")))
    if not len(expert) or not len(dagger):
        raise ValueError("Expert and DAgger sources must both contain usable steps")
    train = ConcatDataset([expert, dagger])
    val = ExpertSteps(args.val_root, cfg, args.datasets, args.max_episodes, args.max_deviation) if args.val_root else None
    if len(train) == 0 or (val is not None and len(val) == 0):
        raise ValueError("Training and supplied validation roots must contain usable expert steps")
    summary = {"train_steps": len(train), "train_episodes": len(expert.episodes) + len(dagger.episodes),
               "expert_steps": len(expert), "expert_episodes": len(expert.episodes),
               "dagger_steps": len(dagger), "dagger_episodes": len(dagger.episodes),
               "sampling": "50% original expert / 50% DAgger, uniform within each source with replacement",
               "expert_action_counts": dict(zip(ACTION_WORDS, expert.action_counts().tolist())),
               "dagger_action_counts": dict(zip(ACTION_WORDS, dagger.action_counts().tolist())),
               "invalid_dagger_labels_excluded": dagger.invalid_labels,
               "train_excluded_episodes": expert.excluded,
               "val_steps": len(val) if val else 0, "val_episodes": len(val.episodes) if val else 0,
               "val_excluded_episodes": val.excluded if val else {}}
    with open(os.path.join(args.run_dir, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.run_dir, "dagger_metadata_sha256.json"), "w") as f:
        json.dump(dagger.metadata_hashes, f, indent=1)
    print({k: v for k, v in summary.items() if "excluded" not in k}, flush=True)

    policy = SmolVLM2Policy(cfg, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from pathlib import Path
    import hashlib
    parent_adapter = Path(args.init_adapter)
    parent_lora = json.loads((parent_adapter / "adapter_config.json").read_text())
    for key, expected in (("r", args.lora_rank), ("lora_alpha", args.lora_alpha), ("lora_dropout", args.lora_dropout)):
        if parent_lora[key] != expected:
            raise ValueError(f"Parent LoRA {key} does not match the refinement configuration")
    loaded = set_peft_model_state_dict(policy.model, load_file(str(parent_adapter / "adapter_model.safetensors")))
    if loaded.unexpected_keys or any("lora_" in key for key in loaded.missing_keys):
        raise ValueError(f"Adapter state mismatch: {loaded}")
    for parameter in policy.trainable_parameters():
        parameter.data = parameter.data.float()
    with open(os.path.join(args.run_dir, "initialization.json"), "w") as f:
        json.dump({"adapter": str(parent_adapter.resolve()), "sha256": hashlib.sha256((parent_adapter / "adapter_model.safetensors").read_bytes()).hexdigest(), "optimizer": "fresh AdamW", "schedule": "fresh warmup/cosine", "loaded_trainable_parameters": sum(p.numel() for p in policy.trainable_parameters())}, f, indent=2)
    if args.grad_ckpt:
        policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.model.enable_input_require_grads()
    params = policy.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_frac * args.max_steps))
    schedule = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.max_steps)))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    step, samples_seen = 0, 0
    if args.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(policy.model, load_file(os.path.join(args.resume, "adapter_model.safetensors")))
        for p in params:
            p.data = p.data.float()
        state = torch.load(os.path.join(args.resume, "trainer_state.pt"), weights_only=False)
        optimizer.load_state_dict(state["optimizer"]), scheduler.load_state_dict(state["scheduler"])
        step, samples_seen = state["step"], state["samples_seen"]
        torch.set_rng_state(state["torch_rng"]), torch.cuda.set_rng_state(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"]), random.setstate(state["python_rng"])

    collate = lambda steps: inputs(steps, with_targets=True)  # noqa: E731
    loader = DataLoader(train, batch_size=args.batch_size, sampler=BalancedSourceSampler(len(expert), len(dagger), args.batch_size * args.grad_accum, args.seed, samples_seen),
                        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers else None)
    val_loader = None
    if val:
        picks = np.linspace(0, len(val) - 1, min(len(val), args.eval_batches * args.batch_size)).astype(int).tolist()
        val_loader = DataLoader(torch.utils.data.Subset(val, picks), batch_size=args.batch_size,
                                num_workers=args.val_workers, collate_fn=collate,
                                persistent_workers=args.val_workers > 0)

    n_params = sum(p.numel() for p in params)
    print(f"trainable {n_params:,} params | start step {step} | samples/step {args.batch_size * args.grad_accum}", flush=True)
    log = open(os.path.join(args.run_dir, "log.jsonl"), "a")
    if val_loader and args.eval_initial and step == 0:
        row = {"step": 0, **evaluate(policy, val_loader, args.eval_batches)}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n"), log.flush()
    batches = iter(loader)
    policy.model.train()
    t0, window_samples, window_tokens = time.time(), 0, 0
    while step < args.max_steps:
        micro = [next(batches) for _ in range(args.grad_accum)]
        total_targets = sum(int((b["labels"] != -100).sum()) for b in micro)
        step_loss = 0.0
        for b in micro:
            b = {k: v.to(policy.device, non_blocking=True) for k, v in b.items()}
            loss, _ = policy.loss_sum(b)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step + 1}")
            (loss / total_targets).backward()
            step_loss += float(loss.detach())
            window_tokens += int(b["attention_mask"].sum())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at optimizer step {step + 1}")
        optimizer.step(), scheduler.step(), optimizer.zero_grad(set_to_none=True)
        step += 1
        samples_seen += args.batch_size * args.grad_accum
        window_samples += args.batch_size * args.grad_accum

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            row = {"step": step, "loss_per_token": step_loss / total_targets, "lr": scheduler.get_last_lr()[0],
                   "grad_norm": grad_norm, "samples_seen": samples_seen, "source_samples_each": samples_seen // 2,
                   "samples_per_s": window_samples / dt, "tokens_per_s": window_tokens / dt,
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n"), log.flush()
            t0, window_samples, window_tokens = time.time(), 0, 0
        if val_loader and (step % args.eval_every == 0 or step == args.max_steps):
            row = {"step": step, **evaluate(policy, val_loader, args.eval_batches)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n"), log.flush()
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(os.path.join(args.run_dir, "checkpoints", f"step_{step:06d}"),
                            policy, optimizer, scheduler, step, samples_seen)
    log.close()


if __name__ == "__main__":
    main()
