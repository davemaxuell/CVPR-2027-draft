"""Evaluate finished checkpoints while SFT continues, with a safe handoff deadline.

Writes the same outputs consumed by run_step2.py. Stops its evaluator by update
3968, well before the main job can launch evaluations at update 4096. Incomplete
episodes are discarded; the main job resumes from atomic completed episode files.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/step2_smolvlm2"


def training_status():
    status = json.loads((OUT / "status.json").read_text())
    rows = []
    for line in (OUT / "train/log.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    step = max(r["step"] for r in rows)
    active = status["status"] == "running" and status["stage"] == "training"
    return active and step < 3968, step


def record(**value):
    value["pid"] = os.getpid()
    tmp = OUT / "overlap_status.json.tmp"
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(OUT / "overlap_status.json")
    print(json.dumps(value), flush=True)


def main():
    lock = open(OUT / "overlap.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, HF_HOME="/data/shared/hf_cache", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for step in [1024, 2048, 3072]:
        checkpoint = OUT / "train/checkpoints" / f"step_{step:06d}"
        destination = OUT / "evaluations" / checkpoint.name
        record(status="waiting", checkpoint=step)
        while True:
            allowed, current = training_status()
            if not allowed:
                record(status="handed_back", training_step=current)
                return
            if current > step and (checkpoint / "trainer_state.pt").exists():
                break
            time.sleep(5)
        if (destination / "summary.json").exists():
            continue
        command = [sys.executable, "-u", "tools/eval_policy.py", "--run-dir", str(OUT / "train"),
                   "--checkpoint", str(checkpoint), "--episodes", "32", "--seed", "0", "--out", str(destination)]
        with open(OUT / f"eval_{step:06d}.stdout.log", "a") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            record(status="evaluating", checkpoint=step, child_pid=child.pid)
            try:
                while child.poll() is None:
                    allowed, current = training_status()
                    if not allowed:
                        child.terminate()
                        try:
                            child.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        record(status="handed_back", training_step=current, checkpoint=step)
                        return
                    time.sleep(5)
                if child.returncode:
                    record(status="failed", checkpoint=step, returncode=child.returncode)
                    return  # Main job will report/retry the incomplete evaluation.
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
    record(status="complete")


if __name__ == "__main__":
    main()
