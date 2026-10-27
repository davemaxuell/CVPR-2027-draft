"""Run the isolated pretrained baseline; keep the original training run unchanged."""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def status(value):
    tmp = ROOT / "status.json.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(ROOT / "status.json")


if __name__ == "__main__":
    lock = open(ROOT / "run.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HOME="/data/shared/hf_cache",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-u", str(ROOT / "source/tools/eval_policy.py"),
               "--run-dir", str(ROOT / "config"), "--episodes", "32", "--seed", "0",
               "--workers", "4", "--out", str(ROOT / "evaluation")]
    state = {"status": "running", "started_utc": now(), "pid": os.getpid(), "command": command}
    status(state)
    with open(ROOT / "eval.stdout.log", "a") as log:
        child = subprocess.Popen(command, cwd=ROOT / "source", env=env, stdout=log, stderr=subprocess.STDOUT)
        state["child_pid"] = child.pid
        status(state)
        code = child.wait()
    state.update(status="complete" if code == 0 else "failed", exit_code=code, completed_utc=now())
    status(state)
    print(json.dumps(state), flush=True)
    sys.exit(code)
