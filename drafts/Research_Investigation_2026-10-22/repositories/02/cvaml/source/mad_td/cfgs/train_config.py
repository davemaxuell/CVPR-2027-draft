from typing import List
from dataclasses import dataclass


@dataclass
class TrainHyperparams:
    seed: int
    num_seeds: int
    tqdm_interval: int
    init_steps: int
    env_steps: int
    update_steps: int
    total_steps: int
    action_samples: int
    batch_size: int
    obs_norm: bool
    log_freq: int
    save_freq: int
    eval_sample_ratio: float
    eval_freq: int
    eval_episodes: int
    save_path: str
    log_adversarial: bool
    log_act_noise: bool
    log_physics: bool
    loss_functions: List[str]
    eval_batches: int = 100
    tdmpc2_rb: bool = False

    precision: str = "float32"
    param_precision: str = "float32"
