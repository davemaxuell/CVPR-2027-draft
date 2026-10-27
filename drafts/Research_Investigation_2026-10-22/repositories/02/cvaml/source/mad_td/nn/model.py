from abc import ABC, abstractmethod

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from distrax import Deterministic, MultivariateNormalDiag, Dirichlet

from mad_td.nn.common import MLP, ResidualBlock
from mad_td.utils.jax import mish, multi_softmax
from mad_td.utils.distributions import (
    NaturalMultivariateNormalDiag,
    ContinuousDeterministic,
    MultiSoftmaxDeterministic,
    MultiSoftmaxDirichlet,
    SplitMultivariateNormalDiag,
)


class ProbNN(nn.Module, ABC):
    @abstractmethod
    def get_output_size(self, output_size):
        pass

    @abstractmethod
    def get_distribution(self, output):
        pass


class MLPEncoder(ProbNN):
    hidden_dim: Sequence[int]
    feature_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
    init_weight: float = 1.0

    @nn.compact
    def __call__(self, states, get_logits=False, get_params=False, **kwargs):
        out = MLP(
            feature_dim=self.hidden_dim,
            output_dim=self.get_output_size(self.feature_dim),
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )(states, get_logits=get_logits, get_params=get_params)

        return self.get_distribution(out)


class SimbaEncoder(ProbNN):

    num_blocks: int
    feature_dim: int

    @nn.compact
    def __call__(self, states, get_logits=False, get_params=False, **kwargs):
        x = nn.Dense(self.feature_dim)(states)
        for i in range(self.num_blocks):
            x = ResidualBlock(
                feature_dim=self.feature_dim,
                activation=mish,
            )(x)
        return self.get_distribution(nn.LayerNorm()(x))


class SimbaLatentModel(ProbNN):
    num_blocks: int
    feature_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, states, actions, get_logits=False):
        x = jnp.concatenate([states, actions], axis=-1)
        x = nn.Dense(self.feature_dim)(x)
        for i in range(self.num_blocks):
            x = ResidualBlock(
                feature_dim=self.feature_dim,
                activation=mish,
            )(x)
        out = nn.Dense(self.get_output_size(self.output_dim) + 1)(x)
        out, reward = out[..., :-1], out[..., -1:]
        return self.get_distribution(out), reward


class SimbaSplitLatentModel(ProbNN):
    num_blocks: int
    feature_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, states, actions, get_logits=False):
        x = jnp.concatenate([states, actions], axis=-1)
        x = nn.Dense(self.feature_dim)(x)
        for i in range(self.num_blocks):
            x = ResidualBlock(
                feature_dim=self.feature_dim,
                activation=mish,
            )(x)
        out = nn.Dense(self.get_output_size(self.output_dim) + 1)(x)
        features = jnp.concatenate([x, states], axis=-1)
        features = nn.Dense(self.feature_dim)(features)
        sigma = ResidualBlock(
            feature_dim=self.feature_dim,
            activation=mish,
        )(features)
        sigma = nn.Dense(self.get_output_size(self.output_dim))(sigma)
        return self.get_distribution((out[..., :-1], sigma)), out[..., -1:]


class LatentModel(ProbNN):
    feature_dim: Sequence[int]
    output_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
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

        _forward = MLP(
            feature_dim=self.feature_dim,
            output_dim=self.get_output_size(self.output_dim),
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )
        forward = _forward(x, get_logits=get_logits)

        return self.get_distribution(forward), reward


class SplitLatentModel(ProbNN):
    feature_dim: Sequence[int]
    output_dim: int
    activation_input: Optional[Callable[[jax.Array], jax.Array]]
    activation_output: Optional[Callable[[jax.Array], jax.Array]]
    activation_hidden: Callable[[jax.Array], jax.Array]
    normalize_input: bool
    normalize_output: bool
    normalize_hidden: bool
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

        _forward = MLP(
            feature_dim=self.feature_dim,
            output_dim=self.get_output_size(self.output_dim),
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )
        forward = _forward(x, get_logits=get_logits)
        _std = MLP(
            feature_dim=self.feature_dim[:-1],
            output_dim=self.get_output_size(self.output_dim),
            activation_hidden=self.activation_hidden,
            activation_input=self.activation_input,
            activation_output=self.activation_output,
            normalize_input=self.normalize_input,
            normalize_output=self.normalize_output,
            normalize_hidden=self.normalize_hidden,
            init_weight=self.init_weight,
        )
        features = jax.lax.stop_gradient(jnp.concatenate([x, forward], -1))
        std = _std(features, get_logits=get_logits)

        return self.get_distribution((forward, std)), reward


class SoftmaxDeterministicNN(ProbNN):
    def get_output_size(self, output_size):
        return output_size

    def get_distribution(self, output):
        return MultiSoftmaxDeterministic(output)


class GaussianNN(ProbNN):
    def get_output_size(self, output_size):
        return 2 * output_size

    def get_distribution(self, output):
        mu, sigma = jnp.split(output, 2, axis=-1)
        sigma = jax.nn.softplus(sigma) + 1.0e-10
        return MultivariateNormalDiag(mu, sigma)


class SplitGaussianNN(ProbNN):
    def get_output_size(self, output_size):
        return output_size

    def get_distribution(self, output):
        mu, sigma = output
        sigma = jax.nn.softplus(sigma) + 1.0e-4
        return SplitMultivariateNormalDiag(mu, sigma)


class NaturalGaussianNN(ProbNN):
    def get_output_size(self, output_size):
        return 2 * output_size

    def get_distribution(self, output):
        mu, sigma = jnp.split(output, 2, axis=-1)
        sigma = jax.nn.softplus(sigma)
        return NaturalMultivariateNormalDiag((mu, sigma))


class ContinuousDeterministicNN(ProbNN):
    def get_output_size(self, output_size):
        return output_size

    def get_distribution(self, output):
        return ContinuousDeterministic(output)


class IdentityEncoder(ContinuousDeterministicNN):
    feature_dim: int

    @nn.compact
    def __call__(self, states, get_logits=False, get_params=False, **kwargs):
        return self.get_distribution(states)


class DirichletNN(ProbNN):
    def get_output_size(self, output_size):
        return output_size

    def get_distribution(self, output):
        return MultiSoftmaxDirichlet(output)


class TDMPC2Encoder(MLPEncoder):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True


class TDMPC2LatentModel(LatentModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = True
    normalize_output: bool = False
    normalize_hidden: bool = True


class TDMPC2SplitLatentModel(SplitLatentModel):
    activation_hidden: Callable[[jax.Array], jax.Array] = mish
    activation_input: Optional[Callable[[jax.Array], jax.Array]] = mish
    activation_output: Optional[Callable[[jax.Array], jax.Array]] = None
    normalize_input: bool = True
    normalize_output: bool = True
    normalize_hidden: bool = True


class TDMPC2SoftmaxDeterministicEncoder(TDMPC2Encoder, SoftmaxDeterministicNN):
    pass


class TDMPC2SoftmaxDeterministicLatentModel(TDMPC2LatentModel, SoftmaxDeterministicNN):
    pass


class TDMPC2ContinuousDeterministicEncoder(TDMPC2Encoder, ContinuousDeterministicNN):
    pass


class TDMPC2ContinuousDeterministicLatentModel(
    TDMPC2LatentModel, ContinuousDeterministicNN
):
    pass


class TDMPC2GaussianEncoder(TDMPC2Encoder, GaussianNN):
    pass


class TDMPC2GaussianLatentModel(TDMPC2LatentModel, GaussianNN):
    pass


class TDMPC2SplitGaussianLatentModel(TDMPC2SplitLatentModel, SplitGaussianNN):
    pass


class TDMPC2NaturalGaussianEncoder(TDMPC2Encoder, NaturalGaussianNN):
    pass


class TDMPC2NaturalGaussianLatentModel(TDMPC2LatentModel, NaturalGaussianNN):
    pass


class TDMPC2DirichletEncoder(TDMPC2Encoder, DirichletNN):
    pass


class TDMPC2DirichletLatentModel(TDMPC2LatentModel, DirichletNN):
    pass


class SimbaContinuousDeterministicEncoder(SimbaEncoder, ContinuousDeterministicNN):
    pass


class SimbaGaussianEncoder(SimbaEncoder, GaussianNN):
    pass


class SimbaContinuousDeterministicLatentModel(
    SimbaLatentModel, ContinuousDeterministicNN
):
    pass


class SimbaGaussianLatentModel(SimbaLatentModel, GaussianNN):
    pass


class SimbaSplitGaussianLatentModel(SimbaSplitLatentModel, SplitGaussianNN):
    pass
