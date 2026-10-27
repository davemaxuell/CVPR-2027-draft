# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MultivariateNormalDiag distribution."""

from typing import Optional, Sequence, Tuple
import chex

import jax.numpy as jnp

from distrax import MultivariateNormalDiag, Dirichlet
from distrax import Deterministic, Distribution
from distrax import Transformed, Chain, Block, Tanh

from mad_td.utils.jax import multi_softmax


class NaturalMultivariateNormalDiag(MultivariateNormalDiag):
    """Multivariate normal distribution on `R^k` with diagonal covariance."""

    def __init__(self, eta: Tuple[chex.Array, chex.Array]):
        """Initializes a MultivariateNormalDiag distribution.

        Args:
            eta: Tuple of natural parameters. The second natural parameter is assumed to be positive for ease of use, deviating from the literature definition
        """
        eta_1 = eta[0]
        eta_2 = eta[1]

        var = 1 / (2 * eta_2)
        mu = eta_1 * var
        sigma = jnp.sqrt(var)

        super().__init__(mu, sigma)


class SplitMultivariateNormalDiag(MultivariateNormalDiag):
    pass


class ContinuousDeterministic(Deterministic):
    def __getitem__(self, index):
        return ContinuousDeterministic(self.loc[index])


class MultiSoftmaxDeterministic(Deterministic):
    def sample(self, *, seed, softmax=True, sample_shape=None):
        if not sample_shape is None:
            s = super().sample(seed=seed, sample_shape=sample_shape)
        else:
            s = super().sample(seed=seed)

        return multi_softmax(s, get_logits=not softmax)


class TanhMultivariateNormalDiag(Transformed):
    def __init__(self, loc, scale_diag):
        dist = MultivariateNormalDiag(loc, scale_diag)
        bijector = Chain([Block(Tanh(), 1)])

        super().__init__(distribution=dist, bijector=bijector)

    def log_prob(self, value):
        return super().log_prob(value)[..., None]

    def _sample_n_and_log_prob(self, key, n):
        s, l = super()._sample_n_and_log_prob(key, n)
        return s, l[..., None]


class MultiSoftmaxDirichlet(Dirichlet):
    def __init__(self, concentration, dim=8):
        conc = concentration.reshape([concentration.shape[0], -1, dim])
        self.orig_shape = concentration.shape

        super().__init__(conc)

    def sample(self, *, seed, sample_shape=...):
        sample = super().sample(seed=seed, sample_shape=sample_shape)
        return sample.reshape(self.orig_shape)

    def log_prob(self, value):
        if value.shape == self.orig_shape:
            value = value.reshape(self.concentration.shape)

        return super().log_prob(value)


def dist_stack(dists: Sequence[Distribution]):
    pass


class StackedDistribution(Distribution):
    def __init__(self, dists: Sequence[Distribution]):
        pass


class StackedGaussian(MultivariateNormalDiag, StackedDistribution):
    def __init__(self, dists):
        locs = jnp.stack([d.loc for d in dists], 0)
        scales = jnp.stack([d.scale for d in dists], 0)

        return MultivariateNormalDiag(loc=locs, scale_diag=scales)
