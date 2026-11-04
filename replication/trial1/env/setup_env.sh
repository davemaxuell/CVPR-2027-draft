#!/usr/bin/env bash
# Build the `rolevln` conda env for the vanilla baseline.
# Simulator stack follows StreamVLN (Python 3.9, habitat-sim 0.2.4 headless+bullet,
# habitat-lab v0.2.4). The ML stack is newer than StreamVLN's torch 2.1.2 because
# SmolVLM2 / LLaVA-OneVision / Qwen3-VL need transformers 4.57.
set -eo pipefail  # no -u: conda activate scripts reference unset vars

ENV_NAME=${ENV_NAME:-rolevln}
source /home/bufsgpu/yes/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.9
fi
conda activate "$ENV_NAME"
# Outside the project tree: pip misreads "[OURS]" in an editable path as extras.
HABITAT_LAB_DIR="$CONDA_PREFIX/src/habitat-lab"

if ! python -c 'import habitat_sim' 2>/dev/null; then
  conda install -y habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
fi

if [ ! -d "$HABITAT_LAB_DIR" ]; then
  git clone --depth 1 --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git "$HABITAT_LAB_DIR"
fi
pip install -e "$HABITAT_LAB_DIR/habitat-lab"

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.1" "peft==0.17.1" "accelerate==1.10.1" \
  num2words einops timm safetensors pyyaml huggingface_hub

python - <<'EOF'
import habitat_sim, habitat, torch, transformers, peft
print("habitat_sim", habitat_sim.__version__, "| habitat", habitat.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| peft", peft.__version__)
EOF
