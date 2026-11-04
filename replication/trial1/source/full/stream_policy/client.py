"""Simulator-side handle on a policy server process (standard library only; env `rolevln`).

    with PolicyClient(run_dir=..., weights=..., gpu=0) as policy:
        policy.reset(instruction)
        action, reply = policy.act(t, previous_action, rgb)   # rgb: H x W x 3 uint8 array
        policy.observe(t, previous_action, rgb)               # a step chosen by someone else

The server runs in the policy environment's Python: $ROLEVLN_QWEN_PYTHON, otherwise the
`rolevln-qwen35` conda env next to the current one.
"""
import os
import subprocess
import sys
from pathlib import Path

from .protocol import receive, send

ROOT = Path(__file__).resolve().parents[1]
POLICY_ENV = "rolevln-qwen35"


def policy_python():
    explicit = os.environ.get("ROLEVLN_QWEN_PYTHON")
    if explicit:
        return explicit
    prefix = Path(sys.prefix)
    envs = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    return str(envs / POLICY_ENV / "bin" / "python")


class PolicyClient:
    def __init__(self, run_dir="", weights="", gpu=0, stop_bias=0.0, log=None, python=None):
        command = [python or policy_python(), "-u", "-m", "stream_policy.server", "--run-dir", str(run_dir),
                   "--weights", str(weights), "--stop-bias", repr(float(stop_bias))]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        env.pop("PYTHONPATH", None)
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                        cwd=str(ROOT), env=env)
        self.identity = self.call({"op": "hello"})

    def call(self, header, payload=b""):
        send(self.process.stdin, header, payload)
        reply, _ = receive(self.process.stdout)
        if reply is None:
            raise RuntimeError(f"policy server exited (code {self.process.poll()})")
        if "error" in reply:
            raise RuntimeError("policy server error:\n" + reply["error"])
        return reply

    def reset(self, instruction):
        self.call({"op": "reset", "instruction": instruction})

    def _frame(self, op, t, executed, rgb):
        header = {"op": op, "t": int(t), "executed": None if executed is None else int(executed), "shape": list(rgb.shape)}
        return self.call(header, rgb.tobytes())

    def act(self, t, executed, rgb):
        reply = self._frame("act", t, executed, rgb)
        return int(reply["action"]), reply

    def observe(self, t, executed, rgb):
        self._frame("observe", t, executed, rgb)

    def close(self):
        if self.process.poll() is None:
            try:
                send(self.process.stdin, {"op": "close"})
                self.process.stdin.close()
                self.process.wait(timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
