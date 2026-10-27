from typing import Callable, List, Optional, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.typing as tp

from mad_td.utils.jax import mish, multi_softmax
from mad_td.nn.model import MLPEncoder, LatentModel
from mad_td.nn.common import MLP, EpinetMLP


class EpinetModel(nn.Module):
    feature_dim: Sequence[int]
    output_dim: int

    epinet_feature_dim: Sequence[int]
    epinet_noise_dim: int

    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states, actions, get_logits=False):
        x = jnp.concatenate([states, actions], axis=-1)

        true_reward = MLP(
            output_dim=1,
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=None,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )
        reward = true_reward(x)

        _forward = EpinetMLP(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
            epinet_feature_dim=self.epinet_feature_dim,
            epinet_noise_dim=self.epinet_noise_dim,
        )
        forward = _forward(x, get_logits=get_logits)

        return forward, reward


class EpinetEncoder(nn.Module):
    hidden_dim: Sequence[int]
    feature_dim: int

    epinet_feature_dim: Sequence[int]
    epinet_noise_dim: int

    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states, get_logits=False):
        return EpinetMLP(
            feature_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
            epinet_feature_dim=self.epinet_feature_dim,
            epinet_noise_dim=self.epinet_noise_dim,
        )(states, get_logits=get_logits)
