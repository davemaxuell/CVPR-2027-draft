import jax.numpy as jnp
import flax.linen as nn

from mad_td.nn.common import (
    DecoderResidualNetwork,
    EncoderResidualNetwork,
    ResidualBlock,
    EMANorm,
    Layer,
)


class GaussianRSNormEncoder(GaussianEncoder):
    num_blocks: int = 1

    @nn.compact
    def __call__(self, states, get_logits=False):
        x = EMANorm()(states)

        out = EncoderResidualNetwork(
            num_blocks=self.num_blocks - 1,
            feature_dim=self.feature_dim,
            activation=self.activation_hidden,
        )(x)

        mu = ResidualBlock(
            feature_dim=self.feature_dim, activation=self.activation_hidden
        )(out)

        sigma = ResidualBlock(
            feature_dim=self.feature_dim, activation=self.activation_hidden
        )(out)

        # mu = self.activation_output(mu)
        sigma = nn.softplus(sigma) + 1.0e-10

        return mu, sigma


class GaussianRSNormModel(GaussianLatentModel):
    num_blocks: int = 1

    @nn.compact
    def __call__(self, states, actions, get_logits=False):
        x = jnp.concatenate([states, actions], axis=-1)

        true_reward = DecoderResidualNetwork(
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            activation=self.activation_hidden,
            output_dim=1,
            output_activation=None,
        )
        reward = true_reward(x)

        out = EncoderResidualNetwork(
            num_blocks=self.num_blocks - 1,
            feature_dim=self.feature_dim,
            activation=self.activation_hidden,
        )(x)

        mu = ResidualBlock(
            feature_dim=self.feature_dim, activation=self.activation_hidden
        )(out)

        sigma = ResidualBlock(
            feature_dim=self.feature_dim, activation=self.activation_hidden
        )(out)

        # mu = self.activation_output(mu)
        sigma = nn.softplus(sigma) + 1.0e-10

        return (mu, sigma, reward)


class GaussianRSNormModelv2(GaussianRSNormModel):
    num_blocks: int = 1

    @nn.compact
    def __call__(self, states, actions, get_logits=False):
        x = jnp.concatenate([states, actions], axis=-1)

        true_reward = DecoderResidualNetwork(
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            activation=self.activation_hidden,
            output_dim=1,
            output_activation=None,
        )
        reward = true_reward(x)

        out = EncoderResidualNetwork(
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            activation=self.activation_hidden,
        )(x)

        mu = Layer(self.feature_dim, activation=None)(out)

        sigma = Layer(self.feature_dim, activation=None)(out)

        # mu = self.activation_output(mu)
        sigma = nn.softplus(sigma) + 1.0e-10

        return (mu, sigma, reward)


class GaussianRSNormEncoderv2(GaussianRSNormEncoder):
    @nn.compact
    def __call__(self, states, get_logits=False):
        x = EMANorm()(states)

        out = EncoderResidualNetwork(
            num_blocks=self.num_blocks,
            feature_dim=self.feature_dim,
            activation=self.activation_hidden,
        )(x)

        mu = Layer(self.feature_dim, activation=None)(out)

        sigma = Layer(self.feature_dim, activation=None)(out)

        # mu = self.activation_output(mu)
        sigma = nn.softplus(sigma) + 1.0e-10

        return mu, sigma
