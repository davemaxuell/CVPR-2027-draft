#!/usr/bin/env bash
# Build the `rolevln-qwen35` conda env for the trial_1 streaming policy (Qwen3.5-4B).
#
# Why a second env: Qwen3.5 (model_type qwen3_5) needs transformers >= 5, which needs Python >= 3.10,
# while habitat-sim 0.2.4 is built only for Python 3.9. Habitat therefore stays in `rolevln` and
# drives this env's policy server through stream_policy/client.py; training needs only this env.
#
# Pins: flash-linear-attention (fla-core) runs the 24 Gated DeltaNet layers ~4x faster than the torch
# fallback, but on Hopper it refuses its backward with Triton 3.4-3.7.0 (wrong gradients, fla #640),
# so torch 2.13 + Triton 3.7.1. The cu126 build runs on CUDA 12.x drivers (550 / CUDA 12.4 on the
# H100 server) through minor-version compatibility; torch 2.13's PyPI default is CUDA 13 and does not.
# On another GPU or driver, pick the matching torch CUDA build (Blackwell needs cu128 or newer); the
# Triton >= 3.7.1 requirement is Hopper-specific. No causal-conv1d: its CUDA kernel must be compiled,
# and the torch conv fallback is a small depthwise convolution.
set -eo pipefail  # no -u: conda activate scripts reference unset vars

ENV_NAME=${ENV_NAME:-rolevln-qwen35}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_BASE:-$(conda info --base)}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  CONDA_ADD_PIP_AS_PYTHON_DEPENDENCY=0 conda create -y -n "$ENV_NAME" --no-default-packages python=3.12
fi
conda activate "$ENV_NAME"
PY="$(command -v python)"

if command -v uv >/dev/null; then
  uv pip install --python "$PY" --index-strategy unsafe-best-match -r "$HERE/qwen35_requirements.txt"
else
  python -m ensurepip --upgrade && python -m pip install -r "$HERE/qwen35_requirements.txt"
fi

# A shared uv cache can hand out hardlinked binaries without their executable bit (seen on the H100
# server: Triton's ptxas). Replace any such file with a private executable copy; the cache is untouched.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
find "$SITE/triton/backends/nvidia/bin" "$SITE/torch/bin" -type f ! -perm -u+x 2>/dev/null | while read -r f; do
  cp "$f" "$f.copy" && mv -f "$f.copy" "$f" && chmod 755 "$f"
done

python - <<'EOF'
import torch, triton, transformers, peft, fla
from transformers.utils.import_utils import is_flash_linear_attention_available
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("triton", triton.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| fla", fla.__version__)
print("fla kernels visible to transformers:", is_flash_linear_attention_available())
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401  (model_type qwen3_5)
EOF
echo "Weights: Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (~8.8 GB) under \$HF_HOME."
