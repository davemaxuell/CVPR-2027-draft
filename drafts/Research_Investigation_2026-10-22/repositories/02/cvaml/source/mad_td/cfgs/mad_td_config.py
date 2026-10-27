from dataclasses import dataclass, replace
from typing import Callable, Dict

from flax.training.train_state import TrainState
from flax import struct


@dataclass
class MadTdLossFunctions:
    loss_functions: Dict[str, Callable]

    def __hash__(self):
        return hash(self.loss_functions.keys())

    def __eq__(self, __value: object) -> bool:
        return hash(self) == hash(__value)

    def __repr__(self) -> str:
        return str(self.loss_functions.keys())


@dataclass
class MadTdHyperparams:
    batch_size: int

    actor_learning_rate: float
    critic_learning_rate: float
    encoder_learning_rate: float
    model_learning_rate: float
    gradient_clip: float
    gamma: float
    tau: float
    update_encoder_with_critic_loss: bool = False
    skip_model_update: bool = False
    use_model_states_for_actor_update: bool = True
    proportion_real: float = 0.5

    use_daml: bool = True
    use_aux: bool = True
    use_model_loss: bool = True

    num_critic_samples: int = 1

    # MPC parameters
    use_mpc: bool = False
    num_policy_mpc_samples: int = 1
    num_mpc_samples: int = 1
    mpc_iterations: int = 1
    mpc_top_k: int = 1
    mpc_temperature: float = 1.0
    use_cheap_mpc: bool = False

    use_target_encoder: bool = True
    use_model_gradient_for_actor: bool = True

    mpc_rollout_length: int = 1
    mpc_lower_std_bound: float = 0.05
    mpc_upper_std_bound: float = 0.3
    mpc_add_final_noise: bool = False

    exploration_noise_sigma: float = 0.3
    expl_multiplier: float = 0.0
    expl_bonus: str = "none"

    mpc_v: str = "mean"

    model_loss_scaler: float = 1.0
    model_loss: str = "likelihood"

    bayes_risk_lin: bool = False
    bayes_risk_stop: int = 100_000
    bayes_risk_lambda: float = 0.3

    uncertainty_method: str = "entropy"
    stoch_agg: str = "mean"
    use_encoder_uncertainty: str = "const"

    use_encoder_mean_critic: bool = False
    use_rank_by_model_critic: bool = False
    num_model_samples_critic: int = 1
    num_stochastic_samples_mpc: int = 1

    penalize_enc_div_from_uniform: float = 0.0

    lat_to_env_ratio: float = 0.0

    # daml parameters
    model_daml_model_steps: int = 1
    model_daml_real_env_steps: int = 0
    model_daml_target_steps: int = 0

    critic_daml_model_steps: int = 1
    critic_daml_real_env_steps: int = 0
    critic_daml_target_steps: int = 0

    add_first_model_state: bool = False
    num_stochastic_samples: int = 32

    scale_loss: bool = False

    update_model: bool = True

    def __hash__(self):
        return hash(
            (
                self.actor_learning_rate,
                self.critic_learning_rate,
                self.encoder_learning_rate,
                self.model_learning_rate,
                self.gradient_clip,
                self.gamma,
            )
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@struct.dataclass
class MadTdModels:
    critic: TrainState
    critic_target: TrainState
    actor: TrainState
    encoder: TrainState
    encoder_target: TrainState
    latent_model: TrainState
