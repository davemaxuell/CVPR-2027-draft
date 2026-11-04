"""GPU checks of the Qwen3.5 streaming backbone on real group-A expert windows (env rolevln-qwen35).

1. Tokens: one token per action, 192 tokens per 512x384 view, labels on the token before each action.
2. Cached decoding equals full re-encoding: the logits of each decoded action match a fresh forward
   over the same teacher-forced prefix.
3. Training equals inference: an action's logits inside a whole training window match the
   prefix-only forward the policy server runs before that action.
4. Each train mode in a fresh process: one optimizer update on the longest window, peak memory, time.
5. Kernels, in a fresh process: bf16 training gradients with flash-linear-attention and with
   transformers' torch fallback (chunk rule and gated norm), each against an fp32 torch reference on
   one window. fla must be at least as close to fp32 as the fallback on every probed weight.

    CUDA_VISIBLE_DEVICES=0 python tools/check_stream_backbone.py --out logs/check_stream_backbone.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from stream_policy.config import StreamConfig  # noqa: E402
from stream_policy.data import TrajectoryWindows  # noqa: E402
from stream_policy.layout import fixed_conjunction, layout, window_start  # noqa: E402
from stream_policy.qwen35 import Qwen35Inputs, Qwen35Policy  # noqa: E402

EXPERT_A = "/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/expert_groupA"


def windows_with_memory(data, count):
    """Indices of windows with memory views (start > 0), longest episodes first, plus window 0 of one."""
    picks = sorted(range(len(data.index)), key=lambda i: -len(data.episodes[data.index[i][0]]["executed"]))
    later = [i for i in picks if data.index[i][1] > 0][:count]
    return [next(i for i in picks if data.index[i][1] == 0)] + later


def check_tokens(inputs, data, indices):
    rows = []
    for i in indices:
        s = data[i]
        e, start, stop = data.index[i]
        ep = data.episodes[e]
        ids = s["ids"]
        pads = sum(t == inputs.image_pad for t in ids)
        actions = [(p, t, st) for p, t, st in zip(s["label_pos"], s["label_tok"], s["label_step"]) if st >= 0]
        ok = (pads == sum(inputs.tokens_per_image(k) for _, k in s["images"])
              and [st for _, _, st in actions] == list(range(start, stop))
              and all(ids[p + 1] == inputs.action_ids[ep["executed"][st]] and t == inputs.action_ids[ep["targets"][st]]
                      for p, t, st in actions)
              and len(s["label_pos"]) - len(actions) == len(range(start, stop, data.cfg.actions_per_turn)))
        rows.append({"window": list(s["key"]), "tokens": len(ids), "images": len(s["images"]),
                     "memory_views": sum(k == "memory" for _, k in s["images"]), "supervised": len(s["label_pos"]), "ok": ok})
    return rows


def prefix_batch(inputs, cfg, ep, t, executed, device):
    start = window_start(t, cfg)
    pieces = layout(cfg, ep["instruction"], start, t, fixed_conjunction(cfg, start), open_turn=True)
    sample = inputs.encode(pieces, executed)
    sample["pixels"] = [inputs.resize(load(ep, f), k) for f, k in sample["images"]]
    return {k: v.to(device) for k, v in inputs.collate([sample]).items()}


def load(ep, frame):
    from PIL import Image
    with Image.open(Path(ep["dir"]) / "rgb" / f"{frame:04d}.jpg") as image:
        return image.convert("RGB")


@torch.no_grad()
def full_scores(policy, batch):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = policy.model(**policy.model_inputs(batch), use_cache=False, logits_to_keep=1).logits
    return logits[0, -1, policy.action_ids].float()


def check_decoding(policy, inputs, cfg, data, index):
    """Cached decode at a turn start vs fresh forwards over the teacher-forced prefixes."""
    e, start, _ = data.index[index]
    ep = data.episodes[e]
    t = start + cfg.window_steps // 2
    executed = list(ep["executed"])
    actions, scores = policy.decode(prefix_batch(inputs, cfg, ep, t, executed, policy.device), cfg.actions_per_turn)
    rows = []
    for k, (a, s) in enumerate(zip(actions, scores)):
        forced = executed[:t] + actions[:k] + [0] * (len(executed) - t - k)
        fresh = full_scores(policy, prefix_batch(inputs, cfg, ep, t + k, forced, policy.device))
        rows.append({"step": t + k, "action": a, "fresh_argmax": int(fresh.argmax()),
                     "max_abs_logit_diff": float((fresh - torch.tensor(s, device=fresh.device)).abs().max())})
    return {"window": [ep["dataset"], ep["episode_id"], start], "rows": rows,
            "passed": all(r["action"] == r["fresh_argmax"] and r["max_abs_logit_diff"] < 0.5 for r in rows)}


def check_train_vs_inference(policy, inputs, cfg, data, index):
    """Logits of every action in one training window vs the server's prefix forward before it."""
    e, start, stop = data.index[index]
    ep = data.episodes[e]
    conj = fixed_conjunction(cfg, start)
    sample = inputs.encode(layout(cfg, ep["instruction"], start, stop, conj), ep["executed"], ep["targets"])
    sample["pixels"] = [inputs.resize(load(ep, f), k) for f, k in sample["images"]]
    batch = {k: v.to(policy.device) for k, v in inputs.collate([sample]).items()}
    with torch.no_grad():
        window_logits = policy.supervised_logits(batch)[:, policy.action_ids]
    rows = []
    for j, step in enumerate(sample["label_step"]):
        if step < 0 or step % 5:
            continue
        fresh = full_scores(policy, prefix_batch(inputs, cfg, ep, step, ep["executed"], policy.device))
        rows.append({"step": step, "max_abs_logit_diff": float((fresh - window_logits[j]).abs().max()),
                     "same_argmax": int(fresh.argmax()) == int(window_logits[j].argmax())})
    return {"window": [ep["dataset"], ep["episode_id"], start], "rows": rows,
            "passed": all(r["same_argmax"] and r["max_abs_logit_diff"] < 0.5 for r in rows)}


def profile_mode(mode, args):
    """One optimizer update in `mode` on the longest window: peak memory and seconds per update."""
    cfg = StreamConfig()
    inputs = Qwen35Inputs(cfg)
    data = TrajectoryWindows([EXPERT_A], cfg, inputs, max_episodes=args.max_episodes)
    index = windows_with_memory(data, 1)[-1]
    batch = {k: v.to("cuda") for k, v in inputs.collate([data[index]] * args.batch).items()}
    torch.cuda.reset_peak_memory_stats()
    began = time.time()
    policy = Qwen35Policy(cfg, inputs.action_ids, mode=mode, device="cuda", gradient_checkpointing=True)
    load_s = time.time() - began
    policy.model.train()
    optimizer = torch.optim.AdamW(policy.parameter_groups(2e-5, 5e-6), fused=True)
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        began = time.time()
        loss, n, _ = policy.loss_sum(batch)
        (loss / n).backward()
        optimizer.step(), optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - began)
    return {"mode": mode, "micro_batch": args.batch, "tokens_per_window": int(batch["lengths"].max()),
            "trainable_parameters": sum(p.numel() for p in policy.trainable_parameters()),
            "loss_per_token": float(loss) / n, "load_seconds": load_s,
            "seconds_per_update": times[1:], "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30}


KERNEL_PROBES = ("layers.0.linear_attn.in_proj_qkv", "layers.2.linear_attn.out_proj", "layers.3.self_attn.q_proj",
                 "layers.30.mlp.down_proj", "visual.blocks.0.attn.qkv.weight", "visual.blocks.12.mlp.linear_fc1.weight",
                 "visual.merger.linear_fc2.weight")


def use_torch_kernels(model):
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    for m in [m for m in model.modules() if isinstance(m, qwen.Qwen3_5GatedDeltaNet)]:
        m.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
        norm = qwen.Qwen3_5RMSNormGated(m.head_v_dim, eps=m.norm.eps).to(m.norm.weight.device, m.norm.weight.dtype)
        norm.weight.data.copy_(m.norm.weight.data)
        m.norm = norm


def kernel_gradients(policy, batch, autocast):
    """Mean supervised-token loss, seconds, and the probed gradients (CPU, float64) of one backward pass."""
    keep, inverse = torch.unique(batch["label_pos"], sorted=True, return_inverse=True)
    policy.model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    began = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits = policy.model(**policy.model_inputs(batch), logits_to_keep=keep, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(logits[batch["label_rows"], inverse].float(), batch["label_tok"])
    loss.backward()
    torch.cuda.synchronize()
    grads = {n: p.grad.detach().double().flatten().cpu() for n, p in policy.model.named_parameters()
             if p.grad is not None and any(k in n for k in KERNEL_PROBES)}
    return float(loss), time.time() - began, grads


def compare_kernels(args):
    """bf16 gradients with fla and with the torch fallback, each against an fp32 torch reference."""
    cfg = StreamConfig()
    inputs = Qwen35Inputs(cfg)
    data = TrajectoryWindows([EXPERT_A], cfg, inputs, max_episodes=args.max_episodes, fixed_conjunctions=True)
    batch = {k: v.to("cuda") for k, v in inputs.collate([data[windows_with_memory(data, 1)[-1]]]).items()}
    rows = {}
    for name, dtype, torch_kernels in (("fp32_torch_reference", torch.float32, True), ("bf16_fla", torch.bfloat16, False),
                                       ("bf16_torch_fallback", torch.bfloat16, True)):
        policy = Qwen35Policy(cfg, inputs.action_ids, mode="frozen", device="cuda", gradient_checkpointing=True)
        if dtype == torch.float32:
            policy.model.float()
        policy.model.requires_grad_(True)
        policy.model.train()
        if torch_kernels:
            use_torch_kernels(policy.model)
        step = dict(batch, pixel_values=batch["pixel_values"].to(dtype))
        if not torch_kernels:
            kernel_gradients(policy, step, True)  # compile the Triton kernels first
        rows[name] = kernel_gradients(policy, step, dtype != torch.float32)
        del policy
        torch.cuda.empty_cache()
    reference = rows["fp32_torch_reference"][2]
    cosine = lambda a, b: float(a @ b / (a.norm() * b.norm()))  # noqa: E731
    vs = {k: {n.split("model.")[-1]: cosine(g, reference[n]) for n, g in v[2].items()} for k, v in rows.items() if k != "fp32_torch_reference"}
    return {"tokens": int(batch["lengths"].max()), "loss_per_token": {k: v[0] for k, v in rows.items()},
            "seconds_forward_backward": {k: v[1] for k, v in rows.items()}, "gradient_cosine_vs_fp32": vs,
            "fla_at_least_as_close": all(vs["bf16_fla"][n] >= vs["bf16_torch_fallback"][n] - 0.02 for n in vs["bf16_fla"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--modes", default="lora,llm,full", help="train modes to profile, each in its own process")
    ap.add_argument("--profile-only", default="", help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=40, help="per dataset, read from the group-A renders")
    ap.add_argument("--skip-kernels", action="store_true", help="skip the fla vs torch-fallback comparison")
    args = ap.parse_args()
    if args.profile_only:
        result = compare_kernels(args) if args.profile_only == "kernels" else profile_mode(args.profile_only, args)
        print("PROFILE " + json.dumps(result), flush=True)
        return
    cfg = StreamConfig()
    inputs = Qwen35Inputs(cfg)
    data = TrajectoryWindows([EXPERT_A], cfg, inputs, max_episodes=args.max_episodes)
    indices = windows_with_memory(data, 3)
    import fla
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    report = {"config": cfg.__dict__, "action_token_ids": dict(zip(cfg.action_tokens, inputs.action_ids)),
              "tokens_per_view": inputs.tokens_per_image("view"), "tokens_per_memory_view": inputs.tokens_per_image("memory"),
              "fla": fla.__version__, "fla_chunk_kernel": qwen.chunk_gated_delta_rule is not None,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
              "data": {"windows": len(data), "episodes": len(data.episodes)}, "tokens": check_tokens(inputs, data, indices)}
    policy = Qwen35Policy(cfg, inputs.action_ids, mode="frozen", device="cuda")
    report["cached_decoding"] = [check_decoding(policy, inputs, cfg, data, i) for i in indices[:2]]
    report["train_vs_inference"] = [check_train_vs_inference(policy, inputs, cfg, data, i) for i in indices[:2]]
    del policy
    torch.cuda.empty_cache()
    def fresh(what):
        run = subprocess.run([sys.executable, __file__, "--profile-only", what, "--batch", str(args.batch),
                              "--max-episodes", str(args.max_episodes)], capture_output=True, text=True, cwd=ROOT)
        line = [x for x in run.stdout.splitlines() if x.startswith("PROFILE ")]
        return json.loads(line[-1][8:]) if line else {"mode": what, "error": run.stderr[-3000:]}

    report["train_modes"] = [fresh(mode) for mode in args.modes.split(",") if mode]
    report["kernels"] = fresh("kernels") if not args.skip_kernels else None
    kernels_ok = args.skip_kernels or report["kernels"].get("fla_at_least_as_close", False)
    report["passed"] = (all(r["ok"] for r in report["tokens"]) and all(r["passed"] for r in report["cached_decoding"])
                        and all(r["passed"] for r in report["train_vs_inference"])
                        and all("error" not in r for r in report["train_modes"]) and kernels_ok)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
