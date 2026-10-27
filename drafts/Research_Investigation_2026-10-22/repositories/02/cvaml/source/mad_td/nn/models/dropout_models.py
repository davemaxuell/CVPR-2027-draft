from typing import Callable, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn

from mad_td.utils.jax import mish, multi_softmax
from mad_td.nn.model import MLPEncoder, LatentModel
from mad_td.nn.common import (
    MLP,
    DropoutMLP,
)


class DropoutModel(LatentModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True
    init_weight: float = 1.0
    rate: float = 0.1

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

        _forward = DropoutMLP(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
            rate=self.rate,
        )
        forward = _forward(x, get_logits=get_logits)

        return forward, reward


class DropoutEncoder(MLPEncoder):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = multi_softmax
    normalize_input: bool = True
    normalize_output: bool = True
    normalize_hidden: bool = True
    rate: float = 0.1

    @nn.compact
    def __call__(self, states, get_logits=False):
        return DropoutMLP(
            feature_dim=self.hidden_dim,
            output_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
            rate=self.rate,
        )(states, get_logits=get_logits)
