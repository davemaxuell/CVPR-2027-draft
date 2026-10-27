"""Download the first-trial checkpoints (SmolVLM2-500M policy, DINOv2-S/14 encoder) into HF_HOME
and record their resolved revisions for the experiment manifest.

    HF_HOME=/data/shared/hf_cache python -I env/download_models.py manifest/model_revisions.json
"""
import json, sys
from huggingface_hub import snapshot_download

REPOS = {
    "policy_smolvlm2_500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "dino_encoder": "facebook/dinov2-small",
}
out = {}
for key, repo in REPOS.items():
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
    out[key] = {"repo": repo, "revision": path.rstrip("/").split("/")[-1], "path": path}
    print(key, out[key], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)
