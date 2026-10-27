# Usage: source env.sh. SCALEVLN_ENV may select another compatible environment.
source /home/bufsgpu/yes/etc/profile.d/conda.sh
conda activate "${SCALEVLN_ENV:-scalevln}" || return 1
# An inherited virtualenv may precede conda in PATH even after activation.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export SCALEVLN_PYTHON="${CONDA_PREFIX}/bin/python"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Matterport3DSimulator/build:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
# Everything needed by the released agents is cached locally.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
