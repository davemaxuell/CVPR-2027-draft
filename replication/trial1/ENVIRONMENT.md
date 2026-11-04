# Environments

Two Python environments are required. Keep Habitat and Qwen3.5 separate.

| Environment | Recorded reference |
|---|---|
| Policy/training | Python 3.12, torch 2.13.0+cu126, torchvision 0.28.0+cu126, Triton 3.7.1, transformers 5.5.4, peft 0.18.1, fla-core 0.6.0 |
| Simulator | Python 3.9, habitat-sim 0.2.4 headless + withbullet, habitat-lab v0.2.4; complete package records under `env/recorded/rolevln/` |

`env/setup_env_qwen35.sh`, `env/qwen35_requirements.txt`, and `env/setup_env.sh` are source recipes, not commands to run unreviewed against shared environments. The Habitat recipe contains the old conda path. Adjust a working copy for the receiving machine. Use the recorded environment files to detect missing or drifting packages; recorded conda URLs/prefixes are provenance, not a universal cross-platform lockfile.

Use a CUDA build supported by the target GPU/driver. Preserve the numerical/training recipe and document any necessary environment compatibility differences. Do not upgrade an existing shared environment in place. The target machine must run full fine-tuning at the prescribed batch/context; do not silently substitute LoRA, quantization or a smaller image size to fit.

The full evaluation adapters also require the packages in `env/metrics_requirements.txt` in the simulator environment. The reference server had these in a local dependency directory; compiled binaries are intentionally not exported. Install the pinned packages in the receiving environment and verify official metric execution. Upstream VLN-CE metric source is included with its license under `evaluation/vendor/vlnce/`.

Acquire `Qwen/Qwen3.5-2B` from a pinned immutable Hugging Face revision, with its own processor/tokenizer and complete checkpoint. A model cache is not included. Once present, use `HF_HUB_OFFLINE=1` for training and evaluation.

Before a training launch verify native action tokenization, image-token layout, training/inference agreement, finite full-model gradients, exact checkpoint resume and a Habitat policy-server load test. The optional causal-conv1d warning is not by itself a failure; measure the actual kernel path and speed.

Only the receiving machine can validate its CUDA/EGL setup. This export's validation is CPU-only and does not establish successful 2B training.
