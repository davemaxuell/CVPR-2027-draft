from typing import Callable, Sequence

from abc import ABC, abstractmethod
import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from distrax import Tanh, Block, Deterministic, MultivariateNormalDiag

from mad_td.utils.distributions import TanhMultivariateNormalDiag

from mad_td.nn.common import (
    MLP,
    Layer,
    EncoderDecoderResidualNetwork,
    DecoderResidualNetwork,
    ResidualBlock,
)
from mad_td.utils.jax import mish

# _half_log2pi = 0.5 * math.log(2 * math.pi)
#
#
# class DecoderResidualActor(nn.Module):
#     feature_dim: Sequence[int]
#     output_dim: int
#     num_blocks: int
#     activation: Callable[[jax.Array], jax.Array] = mish
#
#     @nn.compact
#     def __call__(self, x):
#         x = DecoderResidualNetwork(
#             output_dim=self.output_dim,
#             num_blocks=self.num_blocks,
#             feature_dim=self.feature_dim,
#             activation=self.activation,
#             output_activation=jnp.tanh,
#         )(x)
#         return x
#
#
# class EncoderDecoderResidualActor(nn.Module):
#     feature_dim: Sequence[int]
#     output_dim: int
#     num_blocks: int
#     activation: Callable[[jax.Array], jax.Array] = mish
#
#     @nn.compact
#     def __call__(self, x):
#         x = EncoderDecoderResidualNetwork(
#             output_dim=self.output_dim,
#             num_blocks=self.num_blocks,
#             feature_dim=self.feature_dim,
#             activation=self.activation,
#             output_activation=jnp.tanh,
#         )(x)
#         return x


class Actor(nn.Module, ABC):
    @abstractmethod
    def _get_distribution(self, output):
        pass

    @abstractmethod
    def _get_output_size(self, output_size):
        pass


class DeterministicTanhActor(Actor, ABC):
    def _get_distribution(self, output):
        return Deterministic(output)

    def _get_output_size(self, output_size):
        return output_size


class TanhGaussianActor(Actor, ABC):
    def _get_distribution(self, output):
        mu, sigma = jnp.split(output, 2, axis=-1)
        sigma = jax.nn.softplus(sigma)
        return TanhMultivariateNormalDiag(mu, sigma)

    def _get_output_size(self, output_size):
        return output_size * 2


class ActorModel(Actor, ABC):
    feature_dim: Sequence[int]
    output_dim: int
    activation_hidden: Callable[[jax.Array], jax.Array]
    activation_input: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_hidden: bool
    init_temp: float = 1.0
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states):
        x = MLP(
            output_dim=self._get_output_size(self.output_dim),
            feature_dim=self.feature_dim,
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=jnp.tanh,
            normalize_input=self.normalize_input,
            normalize_output=False,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )(states)

        temp = jnp.exp(
            self.param(
                "log_temp",
                lambda rng, shape: jnp.ones((1,)) * jnp.log(self.init_temp),
                None,
            )
        )

        return self._get_distribution(x), temp


class SimbaActorModel(Actor, ABC):
    num_blocks: int
    feature_dim: int
    output_dim: int
    init_temp: float = 1.0

    @nn.compact
    def __call__(self, states):
        for i in range(self.num_blocks):
            states = ResidualBlock(self.feature_dim)(states)
        states = nn.LayerNorm()(states)
        states = nn.tanh(nn.Dense(self._get_output_size(self.output_dim))(states))
        temp = jnp.exp(
            self.param(
                "log_temp",
                lambda rng, shape: jnp.ones((1,)) * jnp.log(self.init_temp),
                None,
            )
        )
        return self._get_distribution(states), temp


class GaussianTDMPC2Actor(TanhGaussianActor, ActorModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Callable[[jax.Array], jax.Array] = mish
    normalize_input: bool = True
    normalize_hidden: bool = True


class DeterministicTDMPC2Actor(DeterministicTanhActor, ActorModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Callable[[jax.Array], jax.Array] = mish
    normalize_input: bool = True
    normalize_hidden: bool = True


class GaussianSimbaActor(TanhGaussianActor, SimbaActorModel):
    pass


class DeterministicSimbaActor(DeterministicTanhActor, SimbaActorModel):
    pass
