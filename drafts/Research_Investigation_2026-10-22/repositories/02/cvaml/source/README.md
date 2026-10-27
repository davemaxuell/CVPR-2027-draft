# Code for "Calibrated Value-Aware Model Learning with Probabilistic Environment Models"

Authors: Claas Voelcker, Anastasiia Pedan, Arash Ahmadian, Romina Abachi, Igor Gilitschenski, Amir-massoud Farahmand

## Installation

Please ensure that you have a cuda12 capable GPU installed. Other GPUs can work, but we do not provide installation help.
All dependencies are best installed via pip using the provided `pyproject.toml` and we strongly recommend using [uv](https://docs.astral.sh/uv/).
With this tool you can simply execute `uv run mad_td/main.py` and all requirements will be installed in a virtual environment.

The codebase is built heavily on [mad-td](https://github.com/adaptive-agents-lab/MAD-TD), so in case of issues, it's always good to double check in that repo as well.

## Paper experiments

We provide raw results in the corresponding folder.

## Citation

If you use our paper or results, please cite us as 

```
@InProceedings{voelcker2025cvaml,
  title={Calibrated Value-Aware Model Learning with Probabilistic Environment Models},
  author={Voelcker, Claas and Pedan, Anastasiia and Ahmadian, Arash and Abachi, Romina and Gilitschenski, Igor and Farahmand, Amir-massoud},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2025}
}
```
