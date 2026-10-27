"""One bounded STOP-focused refinement from Step-3/1024, with audited data views."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.common import PolicyConfig
from policy.smolvlm2 import SmolVLM2Inputs, SmolVLM2Policy
from policy.stop_data import StopSampler, StopSteps, validation_picks
from tools.train_policy_dagger import save_checkpoint
from tools.train_policy_sft import evaluate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-views", nargs="+", required=True)
    ap.add_argument("--val-view", required=True)
    ap.add_argument("--init-run", default="runs/step3_smolvlm2/train")
    ap.add_argument("--init-adapter", default="runs/step3_smolvlm2/train/checkpoints/step_001024")
    ap.add_argument("--max-steps", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--val-workers", type=int, default=16)
    ap.add_argument("--eval-initial", action="store_true")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    seed, batch_size, grad_accum, lr = 3, 16, 4, 2e-5
    assert 0 < args.max_steps <= 2048
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = PolicyConfig.load(Path(args.init_run) / "policy_config.json")
    parent = Path(args.init_adapter).resolve()
    adapter_config = json.loads((parent / "adapter_config.json").read_text())
    assert adapter_config["r"] == 16 and adapter_config["lora_alpha"] == 32 and adapter_config["lora_dropout"] == .05
    sources = ["tools/train_policy_stop.py", "policy/stop_data.py", "policy/smolvlm2.py", "policy/common.py",
               "tools/train_policy_sft.py", "tools/train_policy_dagger.py"]
    signature = {**{k: v for k, v in vars(args).items() if k != "resume"}, "seed": seed, "batch_size": batch_size,
                 "grad_accum": grad_accum, "learning_rate": lr, "warmup_fraction": .03, "clip": 1.0,
                 "optimizer": "fresh AdamW; weight_decay=0", "parent_adapter_sha256": sha(parent / "adapter_model.safetensors"),
                 "source_sha256": {p: sha(ROOT / p) for p in sources},
                 "sampling_categories": StopSampler.CATEGORIES, "validation": "fixed balanced diagnostic; not natural accuracy"}
    # JSON round trip normalizes category tuples for resume equality.
    signature = json.loads(json.dumps(signature))
    if (out / "train_args.json").exists():
        assert json.loads((out / "train_args.json").read_text()) == signature, "Changed training protocol"
        if not args.resume and (out / "log.jsonl").exists():
            raise ValueError("Existing training log requires explicit checkpoint resume")
    else:
        (out / "train_args.json").write_text(json.dumps(signature, indent=2) + "\n")
    cfg.save(out / "policy_config.json")
    train = StopSteps(args.train_views, cfg, group="A")
    val = StopSteps([args.val_view], cfg, group="C")
    picks = validation_picks(val)
    summary = {"train": train.summary(), "validation": val.summary(), "validation_picks": picks,
               "validation_targets": np.bincount(val.targets[picks], minlength=4).tolist(),
               "effective_batch": 64, "STOP_per_batch": 6, "R2R_per_batch": 32, "RxR_per_batch": 32}
    data_path = out / "data_summary.json"
    if data_path.exists():
        assert json.loads(data_path.read_text()) == summary, "Training data changed"
    else:
        data_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"training_pools": summary["train"]["pool_sizes"], "validation_targets": summary["validation_targets"]}), flush=True)
    inputs = SmolVLM2Inputs(cfg)
    inputs.processor.save_pretrained(out / "processor")
    policy = SmolVLM2Policy(cfg, lora_rank=16, lora_alpha=32, lora_dropout=.05)
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    adapter_path = Path(args.resume) if args.resume else parent
    loaded = set_peft_model_state_dict(policy.model, load_file(str(adapter_path / "adapter_model.safetensors")))
    assert not loaded.unexpected_keys and not any("lora_" in key for key in loaded.missing_keys)
    params = list(policy.trainable_parameters())
    assert params and all("lora_" in name for name, value in policy.model.named_parameters() if value.requires_grad)
    for parameter in params:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    warmup = max(1, int(.03 * args.max_steps))
    schedule = lambda s: min(1., (s + 1) / warmup) * .5 * (1 + math.cos(math.pi * min(1., s / args.max_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step, samples_seen = 0, 0
    if args.resume:
        state = torch.load(adapter_path / "trainer_state.pt", weights_only=False)
        optimizer.load_state_dict(state["optimizer"]), scheduler.load_state_dict(state["scheduler"])
        step, samples_seen = state["step"], state["samples_seen"]
        assert samples_seen == step * 64
        torch.set_rng_state(state["torch_rng"]), torch.cuda.set_rng_state(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"]), random.setstate(state["python_rng"])
    collate = lambda steps: inputs(steps, with_targets=True)
    loader = DataLoader(train, batch_size=batch_size, sampler=StopSampler(train.pools, seed, samples_seen),
                        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers else None, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(Subset(val, picks), batch_size=batch_size, num_workers=args.val_workers,
                            collate_fn=collate, persistent_workers=args.val_workers > 0,
                            generator=torch.Generator().manual_seed(seed + 1))
    n_params = sum(p.numel() for p in params)
    (out / "initialization.json").write_text(json.dumps({"adapter": str(parent), "sha256": signature["parent_adapter_sha256"],
        "optimizer": "fresh AdamW", "loaded_trainable_parameters": n_params}, indent=2) + "\n")
    print(f"trainable {n_params:,} params | start step {step} | samples/step 64", flush=True)
    with open(out / "log.jsonl", "a") as log:
        def record(row):
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if args.eval_initial and step == 0:
            record({"step": 0, **evaluate(policy, val_loader, len(val_loader))})
        batches = iter(loader)
        policy.model.train()
        t0, window_samples = time.time(), 0
        while step < args.max_steps:
            micro = [next(batches) for _ in range(grad_accum)]
            target_count = sum(int((batch["labels"] != -100).sum()) for batch in micro)
            step_loss = 0.
            target_actions = torch.cat([b["targets"] for b in micro])
            assert int((target_actions == 0).sum()) == 6
            for batch in micro:
                batch = {k: v.to(policy.device, non_blocking=True) for k, v in batch.items()}
                loss, _ = policy.loss_sum(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at {step + 1}")
                (loss / target_count).backward()
                step_loss += float(loss.detach())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at {step + 1}")
            optimizer.step(), scheduler.step(), optimizer.zero_grad(set_to_none=True)
            step += 1
            samples_seen += 64
            window_samples += 64
            if step == 1 or step % 32 == 0 or step == args.max_steps:
                elapsed = time.time() - t0
                record({"step": step, "loss_per_token": step_loss / target_count, "lr": scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm, "samples_seen": samples_seen, "samples_per_s": window_samples / elapsed,
                        "batch_action_counts": torch.bincount(target_actions, minlength=4).tolist(),
                        "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
                t0, window_samples = time.time(), 0
            if step % 512 == 0 or step == args.max_steps:
                record({"step": step, **evaluate(policy, val_loader, len(val_loader))})
                save_checkpoint(str(out / "checkpoints" / f"step_{step:06d}"), policy, optimizer, scheduler, step, samples_seen)
    final = out / "checkpoints" / f"step_{step:06d}" / "adapter_model.safetensors"
    assert sha(final) != signature["parent_adapter_sha256"]
    (out / "complete.json").write_text(json.dumps({"status": "complete", "steps": step, "samples_seen": samples_seen,
         "final_adapter_sha256": sha(final), "finite_losses_gradients": True, "STOP_in_every_batch": 6}, indent=2) + "\n")


if __name__ == "__main__":
    main()
