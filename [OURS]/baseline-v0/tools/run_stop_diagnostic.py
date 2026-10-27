"""Durable first stage: score two carrier policies and verify native prefixes."""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/step3b_smolvlm2_stop"


def main():
    os.chdir(ROOT)
    lock = open(OUT / "diagnostic.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HOME="/data/shared/hf_cache", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    assert json.loads((OUT / "check_diagnostics.json").read_text())["status"] == "passed"
    state = {"status": "running", "pid": os.getpid()}
    def update(**values):
        state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        temporary = OUT / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(OUT / "status.json")
        print(json.dumps(state), flush=True)
    try:
        for step in (1024, 2048):
            path = OUT / "diagnostic" / f"step_{step:06d}"
            if (path / "summary.json").exists():
                continue
            command = [sys.executable, "-u", "tools/eval_policy_scores.py", "--run-dir", "runs/step3_smolvlm2/train",
                       "--checkpoint", f"runs/step3_smolvlm2/train/checkpoints/step_{step:06d}",
                       "--selection", str(OUT / "development.json"), "--episodes", "32", "--seed", "0",
                       "--movement-only", "--verify-against", str(ROOT / f"runs/step3_smolvlm2/evaluations/step_{step:06d}"),
                       "--out", str(path)]
            update(stage=f"diagnostic_{step}", command=command)
            with open(OUT / f"diagnostic_{step}.log", "a") as logfile:
                child = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT)
                update(child_pid=child.pid)
                code = child.wait()
            if code:
                raise RuntimeError(f"Score evaluation failed: {step}, exit {code}")
            state.pop("child_pid", None)
        update(status="diagnostic_complete", stage="awaiting_calibration_analysis")
    except BaseException as error:
        update(status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
