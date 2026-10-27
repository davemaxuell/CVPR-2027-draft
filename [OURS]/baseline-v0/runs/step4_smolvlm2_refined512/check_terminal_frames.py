"""Fully decode every selected-policy terminal frame and verify its saved pixels."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

RUN = Path(__file__).resolve().parent
VISITS = Path("/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/wm_visits_groupA_smolvlm2_refined512_v1")


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(VISITS / "summary.json")
    reference = read(RUN / "reference/reference.json")
    assert summary["status"] == "complete" and summary["episodes"] == 256
    assert summary["policy_identity_sha256"] == reference["policy_identity_sha256"]
    rows = []
    for path in sorted(VISITS.glob("*/*/meta.json")):
        if path.parent.name.startswith("."):
            continue
        meta = read(path)
        assert meta["num_frames"] == meta["num_actions"] + 1
        assert len(meta["raw_rgb_sha256"]) == meta["num_frames"]
        relative = f"rgb/{meta['num_actions']:04d}.png"
        target = path.parent / relative
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        assert digest == meta["file_sha256"][relative]
        with Image.open(target) as image:
            assert image.format == "PNG" and image.mode == "RGB" and image.size == (640, 480)
            pixels = np.asarray(image)
            assert pixels.dtype == np.uint8 and pixels.shape == (480, 640, 3)
            pixel_digest = hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()
        assert pixel_digest == meta["raw_rgb_sha256"][-1]
        rows.append({"dataset": meta["dataset"], "episode_id": meta["episode_id"],
                     "num_actions": meta["num_actions"], "termination": meta["termination"],
                     "file_sha256": digest, "raw_rgb_sha256": pixel_digest})
    assert len(rows) == 256
    report = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "policy_identity_sha256": reference["policy_identity_sha256"],
              "fully_decoded_terminal_frames": len(rows),
              "all_terminal_frames_match_live_simulator_pixel_hash": True,
              "episodes": rows}
    (RUN / "check_terminal_frames.json").write_text(json.dumps(report, indent=2) + "\n")
    packages = {}
    for name in ("torch", "torchvision", "transformers", "tokenizers", "peft", "numpy", "Pillow",
                 "safetensors", "huggingface-hub", "habitat-lab", "habitat-sim", "omegaconf"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    environment = {"recorded_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                   "executable": sys.executable, "platform": platform.platform(), "packages": packages,
                   "physical_gpu_0": gpu, "collection_cuda_visible_devices": "0",
                   "collection_offline_models": True}
    (RUN / "runtime_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}))
    print(json.dumps(environment))


if __name__ == "__main__":
    main()
