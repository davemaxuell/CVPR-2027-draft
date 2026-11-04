"""Policy server for simulator tools: `python -m stream_policy.server --weights DIR` (env rolevln-qwen35).

Requests arrive on stdin and replies leave on stdout (protocol.py); everything else is sent to stderr.
One episode at a time:
    {"op": "hello"}                                   -> identity: config, weights, hashes, decoding
    {"op": "reset", "instruction": str}               -> {}
    {"op": "act", "t", "executed", "shape"} + RGB     -> {"action", "scores", "remaining", "replanned"}
    {"op": "observe", "t", "executed", "shape"} + RGB -> {}   (a step the policy did not choose)
`executed` is the action actually taken before frame t (null at t = 0). The server keeps the actual
frames and actions and answers with the argmax of the next-token distribution after
layout(..., stop=t, open_turn=True). It decodes the rest of a turn in one pass and reuses that plan
while the executed actions follow it; after any override it re-plans from the new prefix, so its
context always equals the training layout of the actual trajectory.
"""
import argparse
import hashlib
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_hashes(weights):
    if not weights:
        return {}
    return {p.name: sha256(p) for p in sorted(Path(weights).glob("*.safetensors"))}


class StreamingPolicy:
    def __init__(self, args):
        import numpy as np
        import torch
        from PIL import Image

        from .config import StreamConfig
        from .layout import fixed_conjunction, layout, turn_end, window_start
        from .qwen35 import Qwen35Inputs, Qwen35Policy
        self.np, self.torch, self.Image = np, torch, Image
        self.layout, self.fixed_conjunction, self.turn_end, self.window_start = layout, fixed_conjunction, turn_end, window_start
        run = Path(args.run_dir) if args.run_dir else None
        self.cfg = StreamConfig.load(run / "stream_config.json") if run else StreamConfig()
        processor = run / "processor" if run and (run / "processor").exists() else None
        self.inputs = Qwen35Inputs(self.cfg, processor_dir=str(processor) if processor else None)
        torch.manual_seed(0)
        self.policy = Qwen35Policy(self.cfg, self.inputs.action_ids, mode="frozen", weights=args.weights, device="cuda:0")
        self.stop_bias = args.stop_bias
        self.identity = {"backbone": self.cfg.backbone, "repo": self.cfg.repo, "revision": self.cfg.revision,
                         "run_dir": str(run.resolve()) if run else None,
                         "weights": str(Path(args.weights).resolve()) if args.weights else None,
                         "weight_sha256": weight_hashes(args.weights), "stream_config": self.cfg.__dict__,
                         "stop_logit_bias": self.stop_bias, "decoding": "argmax over the four action tokens, batch of one",
                         "source_sha256": {f"stream_policy/{n}": sha256(ROOT / "stream_policy" / n)
                                           for n in ("config.py", "layout.py", "qwen35.py", "server.py")},
                         "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
        self.reset("")

    def reset(self, instruction):
        self.instruction, self.frames, self.executed = instruction, [], []
        self.plan, self.pixels = None, {}

    def observe(self, header, payload):
        t, h, w = header["t"], *header["shape"][:2]
        if t != len(self.frames) or (t > 0) != (header["executed"] is not None):
            raise ValueError(f"step {t} out of order (have {len(self.frames)} frames)")
        if t > 0:
            self.executed.append(int(header["executed"]))
        self.frames.append(self.Image.fromarray(self.np.frombuffer(payload, dtype=self.np.uint8).reshape(h, w, 3)))
        return t

    def act(self, header, payload):
        t = self.observe(header, payload)
        plan, replanned = self.plan, False
        if not (plan and plan["start"] <= t < plan["start"] + len(plan["actions"])
                and self.executed[plan["start"]:t] == plan["actions"][:t - plan["start"]]):
            plan, replanned = self.decode(t), True
        i = t - plan["start"]
        return {"action": plan["actions"][i], "scores": plan["scores"][i], "remaining": len(plan["actions"]) - i - 1,
                "replanned": replanned, "prompt_tokens": plan["tokens"] if replanned else None}

    def decode(self, t):
        start = self.window_start(t, self.cfg)
        pieces = self.layout(self.cfg, self.instruction, start, t, self.fixed_conjunction(self.cfg, start), open_turn=True)
        sample = self.inputs.encode(pieces, self.executed)
        sample["pixels"] = [self.inputs.resize(self.frames[frame], kind) for frame, kind in sample["images"]]
        batch = {k: v.to(self.policy.device) for k, v in self.inputs.collate([sample]).items()}
        actions, scores = self.policy.decode(batch, self.turn_end(t, self.cfg) - t, self.stop_bias)
        self.plan = {"start": t, "actions": actions, "scores": scores, "tokens": len(sample["ids"])}
        return self.plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="", help="training run with stream_config.json and processor/")
    ap.add_argument("--weights", default="", help="checkpoint directory (adapter or full); empty = base model")
    ap.add_argument("--stop-bias", type=float, default=0.0)
    args = ap.parse_args()
    out = os.fdopen(os.dup(1), "wb", buffering=0)  # protocol channel; ordinary prints go to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.path.insert(0, str(ROOT))
    from stream_policy.protocol import receive, send
    stdin = sys.stdin.buffer
    try:
        policy = StreamingPolicy(args)
    except Exception:
        send(out, {"error": traceback.format_exc()})
        raise
    while True:
        header, payload = receive(stdin)
        if header is None or header.get("op") == "close":
            break
        try:
            began = time.time()
            op = header["op"]
            if op == "hello":
                reply = policy.identity
            elif op == "reset":
                policy.reset(header["instruction"])
                reply = {}
            elif op == "observe":
                policy.observe(header, payload)
                reply = {}
            elif op == "act":
                reply = policy.act(header, payload)
                reply["seconds"] = time.time() - began
            else:
                raise ValueError(f"unknown op {op!r}")
            send(out, reply)
        except Exception:
            send(out, {"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
