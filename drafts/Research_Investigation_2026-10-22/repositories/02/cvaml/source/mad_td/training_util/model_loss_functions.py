import jax
import jax.numpy as jnp
from flax.training import train_state
from distrax import Distribution

from mad_td.cfgs.mad_td_config import MadTdHyperparams
from mad_td.training_util.rl_targets import (
    bellman_residual_loss,
    get_daml_target_value_estimate,
)
from mad_td.utils.distributions import (
    MultivariateNormalDiag,
    ContinuousDeterministic,
)


def _dist_mse(weight, dist_enc, dist_model):
    y = dist_model.sample(seed=None)
    return jnp.mean(jnp.square(dist_enc[None] - y) * weight[None]), {}


def _dist_split_nnl(weight, dist_enc, dist_model):
    target = dist_enc.sample(seed=None)
    mean = dist_model.loc
    scale = dist_model.scale_diag

    mse = jnp.square(target - mean)
    scale_loss = jnp.log(scale) + 1.0 / jnp.square(scale) * jax.lax.stop_gradient(mse)
    nll_loss, nll_info = _dist_normal_nn(weight, dist_enc, dist_model)

    return jnp.mean(weight * (mse + scale_loss)), {
        "gaussian/nll_loss": nll_loss,
        "split_gaussian/mse": jnp.mean(mse),
        "split_gaussian/scale_loss": jnp.mean(scale_loss),
        **nll_info,
    }


def _dist_ce(weight, dist_enc, dist_model):
    x = dist_enc.sample(seed=None)
    y = dist_model.sample(seed=None, softmax=False)
    return -jnp.mean(weight * x * y), {}


def _dist_normal_nn(weight, dist_enc, dist_model: MultivariateNormalDiag):
    model_entropy = dist_model.entropy()
    return -jnp.mean(
        weight[..., 0] * (dist_model.log_prob(dist_enc[None]) / dist_enc.shape[-1])
    ), {
        "gaussian/entropy/model": jnp.mean(model_entropy),
        "gaussian/entropy_std/model": jnp.std(model_entropy),
        "gaussian/min_sigma/model": jnp.min(dist_model.scale_diag),
        "gaussian/max_sigma/model": jnp.max(dist_model.scale_diag),
        "gaussian/mean_mu/model": jnp.mean(dist_model.loc),
    }


def _dist_normal_kl(weight, dist_enc, dist_model):
    model_entropy = dist_model.entropy()
    encoder_entropy = dist_enc.entropy()
    return jnp.mean(
        weight * (dist_model.kl_divergence(dist_enc) / dist_model.loc.shape[-1])
    ), {
        "gaussian/entropy/model": jnp.mean(model_entropy),
        "gaussian/entropy_std/model": jnp.std(model_entropy),
        "gaussian/min_sigma/model": jnp.min(dist_model.scale_diag),
        "gaussian/max_sigma/model": jnp.max(dist_model.scale_diag),
        "gaussian/mean_mu/model": jnp.mean(dist_model.loc),
        "gaussian/entropy/encoder": jnp.mean(encoder_entropy),
        "gaussian/entropy_std/encoder": jnp.std(encoder_entropy),
        "gaussian/min_sigma/encoder": jnp.min(dist_enc.scale_diag),
        "gaussian/max_sigma/encoder": jnp.max(dist_enc.scale_diag),
        "gaussian/mean_mu/encoder": jnp.mean(dist_enc.loc),
        "gaussian/l2": jnp.square(dist_model.loc - dist_enc.loc),
    }


def _dist_dirichlet_kl(weight, dist_enc, dist_model):
    model_entropy = dist_model.entropy()
    encoder_entropy = dist_enc.entropy()
    return jnp.mean(
        weight * (dist_model.kl_divergence(dist_enc) / dist_model.loc.shape[-1])
    ), {
        "dirichlet/entropy/model": jnp.mean(model_entropy),
        "dirichlet/entropy_std/model": jnp.std(model_entropy),
        "dirichlet/min_alpha/model": jnp.min(dist_model.concentration),
        "dirichlet/max_alpha/model": jnp.max(dist_model.concentration),
        "dirichlet/alpha_0/model": jnp.mean(dist_model.concentration.sum(-1)),
        "dirichlet/entropy/encoder": jnp.mean(encoder_entropy),
        "dirichlet/entropy_std/encoder": jnp.std(encoder_entropy),
        "dirichlet/min_alpha/encoder": jnp.min(dist_enc.concentration),
        "dirichlet/max_alpha/encoder": jnp.max(dist_enc.concentration),
        "dirichlet/mean_mu/encoder": jnp.mean(dist_enc.concentration.sum(-1)),
    }


def _dist_dirichlet_nn(weight, dist_enc, dist_model):
    x = dist_enc.sample(seed=None)
    model_entropy = dist_model.entropy()
    return -jnp.mean(weight * (dist_model.log_prob(x) / x.shape[-1])), {
        "dirichlet/entropy/model": jnp.mean(model_entropy),
        "dirichlet/entropy_std/model": jnp.std(model_entropy),
        "dirichlet/min_alpha/model": jnp.min(dist_model.concentration),
        "dirichlet/max_alpha/model": jnp.max(dist_model.concentration),
        "dirichlet/alpha_0/model": jnp.mean(dist_model.concentration.sum(-1)),
    }


_LATENT_LOSS_FUNCTION_REGISTRY = {
    ContinuousDeterministic: _dist_mse,
    MultivariateNormalDiag: _dist_normal_nn,
}


def latent_loss(weight, dist_enc, dist_model):
    loss_fn = _LATENT_LOSS_FUNCTION_REGISTRY[dist_model.__class__]
    return loss_fn(weight, dist_enc, dist_model)


def daml_mse(key, real_state, pred_state, critic_target, actor, mask):
    num_samples = pred_state.shape[0]
    real_action_key, pred_action_key = jax.random.split(key)

    # actions
    real_action = jax.lax.stop_gradient(
        actor.apply_fn(actor.params, real_state)[0].sample(seed=real_action_key)
    )
    pred_action = jax.lax.stop_gradient(
        actor.apply_fn(actor.params, pred_state)[0].sample(seed=pred_action_key)
    )[0]

    # critic values
    real_pred_value = critic_target.apply_fn(
        critic_target.params, real_state, pred_action
    )
    pred_pred_value = critic_target.apply_fn(
        critic_target.params,
        pred_state,
        jnp.repeat(pred_action[None], pred_state.shape[0], axis=0),
    )
    real_real_value = critic_target.apply_fn(
        critic_target.params, real_state, real_action
    )
    pred_real_value = critic_target.apply_fn(
        critic_target.params,
        pred_state,
        jnp.repeat(real_action[None], pred_state.shape[0], axis=0),
    )

    real_pred_value = jnp.stack(real_pred_value, 0)
    pred_pred_value = jnp.stack(pred_pred_value, 0)
    real_real_value = jnp.stack(real_real_value, 0)
    pred_real_value = jnp.stack(pred_real_value, 0)

    # compute DAML variance correction terms
    if num_samples > 1:
        pred_var = (1.0 / num_samples) * jnp.var(pred_pred_value, axis=1, ddof=0)
        real_var = (1.0 / num_samples) * jnp.var(pred_real_value, axis=1, ddof=0)
    else:
        pred_var = 0.0
        real_var = 0.0

    loss = jnp.mean(
        jnp.square(jnp.mean(pred_pred_value, axis=1) - real_pred_value) - pred_var
    )
    loss += jnp.mean(
        jnp.square(jnp.mean(pred_real_value, axis=1) - real_real_value) - real_var
    )
    loss = 0.5 * loss

    return loss, {"l2_vaml": loss}


def daml_ce(key, real_state_dist, pred_state_dist, critic_target, actor, mask):
    real_state_key, pred_state_key, real_action_key, pred_action_key = jax.random.split(
        key, 4
    )
    # get state samples
    real_state = real_state_dist.sample(seed=real_state_key)
    pred_state = pred_state_dist.sample(seed=pred_state_key)

    # actions
    real_action = jax.lax.stop_gradient(
        actor.apply_fn(actor.params, real_state)[0].sample(seed=real_action_key)
    )
    pred_action = jax.lax.stop_gradient(
        actor.apply_fn(actor.params, pred_state)[0].sample(seed=pred_action_key)
    )

    # critic values
    real_pred_value = critic_target.apply_fn(
        critic_target.params, real_state, pred_action, get_distribution=True
    )
    pred_pred_value = critic_target.apply_fn(
        critic_target.params, pred_state, pred_action, get_logits=True
    )
    real_real_value = critic_target.apply_fn(
        critic_target.params, real_state, real_action, get_distribution=True
    )
    pred_real_value = critic_target.apply_fn(
        critic_target.params, pred_state, real_action, get_logits=True
    )

    real_pred_value = jnp.stack(real_pred_value, 0)
    pred_pred_value = jnp.stack(pred_pred_value, 0)
    real_real_value = jnp.stack(real_real_value, 0)
    pred_real_value = jnp.stack(pred_real_value, 0)

    loss = jnp.mean(mask * real_pred_value * pred_pred_value)
    loss += jnp.mean(mask * real_real_value * pred_real_value)
    loss = -0.5 * loss

    return loss, {
        "l2_vaml": daml_mse(
            key, real_state_dist, pred_state_dist, critic_target, actor, mask
        )[0]
    }


def target_daml_loss(
    key,
    model_pred_states,
    states,
    actions,
    rewards,
    critic,
    critic_target,
    actor,
    model,
    model_steps,
    real_env_steps,
    model_target_steps,
    num_critic_samples,
    gamma,
):
    num_samples = model_pred_states.shape[0]
    # compute MuZero style target
    target, target_actions = jax.lax.stop_gradient(
        get_daml_target_value_estimate(
            key,
            states,
            actions,
            rewards,
            critic_target,
            actor,
            model,
            model_steps,
            real_env_steps,
            model_target_steps,
            num_critic_samples,
            gamma,
        )
    )
    target = target[:, :model_steps + 1]
    target_actions = target_actions[:, :model_steps + 1]
    target_actions = jnp.repeat(
        target_actions[None], num_samples, axis=0
    )
    model_values_1, model_values_2 = critic.apply_fn(
        critic.params, model_pred_states, target_actions
    )

    model_values_mean = jnp.mean(model_values_1, axis=0)
    if num_samples > 1:
        model_values_var = (1.0 / num_samples) * jnp.var(model_values_1, axis=0, ddof=0)
    else:
        model_values_var = 0.0

    loss = jnp.mean(jnp.square(model_values_mean - target) - model_values_var)

    model_values_mean = jnp.mean(model_values_2, axis=0)
    if num_samples > 1:
        model_values_var = (1.0 / num_samples) * jnp.var(model_values_2, axis=0, ddof=0)
    else:
        model_values_var = 0.0

    loss += jnp.mean(jnp.square(model_values_mean - target) - model_values_var)
    return loss, {
        "l2_daml_loss": loss,
        "mean_target": jnp.mean(target),
        "mean_model_values": jnp.mean(model_values_mean),
        "var_model_values": jnp.mean(model_values_var),
    }


def get_daml_loss(hyperparams: MadTdHyperparams):
    return daml_mse


def get_n_sample_model_rollout(
    key,
    state: jax.Array,
    actions: jax.Array,
    model: train_state.TrainState,
    num_steps: int,
    num_samples: int,
):
    state = jnp.repeat(state[None], num_samples, axis=0)
    actions = jnp.repeat(actions[None], num_samples, axis=0)
    model_rewards_ = []
    model_next_states_ = []
    model_next_states_dists_ = []

    model_next_states_ = []

    for i in range(num_steps):
        latent_state_dist, pred_rewards = model.apply_fn(
            model.params, state, actions[..., i, :]
        )
        state = latent_state_dist.sample(seed=key)
        model_next_states_dists_.append(latent_state_dist)
        model_next_states_.append(state)
        model_rewards_.append(pred_rewards)

    if len(model_rewards_) > 0:
        model_rewards = jnp.stack(model_rewards_, axis=-2)
    else:
        model_rewards = None
    if len(model_next_states_) > 0:
        model_next_states = jnp.stack(model_next_states_, axis=-2)
    else:
        model_next_states = None

    return model_next_states, model_next_states_dists_, model_rewards


def compute_model_loss(
    state: jax.Array,
    actions: jax.Array,
    reward: jax.Array,
    next_states: jax.Array,
    mask: jax.Array,
    encoder: train_state.TrainState,
    encoder_target: train_state.TrainState,
    model: train_state.TrainState,
    critic: train_state.TrainState,
    critic_target: train_state.TrainState,
    actor: train_state.TrainState,
    keys: jax.Array,
    hyperparams: MadTdHyperparams,
):
    # get state encoding
    enc_key, keys = jax.random.split(keys)
    first_latent_state_dist: Distribution = encoder.apply_fn(encoder.params, state)
    first_latent_state = first_latent_state_dist.sample(seed=enc_key)

    (
        model_next_states,
        model_next_states_dists_,
        model_rewards,
    ) = get_n_sample_model_rollout(
        keys,
        first_latent_state,
        actions,
        model,
        hyperparams.model_daml_model_steps,
        hyperparams.num_stochastic_samples,
    )
    real_next_states = encoder_target.apply_fn(
        encoder_target.params, next_states
    ).sample(seed=enc_key)

    # compute losses
    reward_loss = jnp.mean(
        mask[:, : hyperparams.model_daml_model_steps]
        * (reward[None, :, : hyperparams.model_daml_model_steps] - model_rewards) ** 2
    )
    total_loss = reward_loss

    log_dict = {}

    if hyperparams.use_aux:
        model_loss = 0.0
        for i in range(hyperparams.model_daml_model_steps):
            model_loss_, log_dict_ = latent_loss(
                mask[:, i],
                real_next_states[:, i],
                model_next_states_dists_[i],
            )
            model_loss += model_loss_.mean()
        total_loss += model_loss
        log_dict.update(log_dict_)
    else:
        model_loss = jnp.zeros_like(total_loss)

    if hyperparams.use_daml:
        daml_key, keys = jax.random.split(keys)
        if hyperparams.model_daml_real_env_steps > 0:
            real_next_states = jnp.concatenate(
                [first_latent_state[:, None], real_next_states],
                axis=-2,
            )
            model_next_states = jnp.concatenate(
                [jnp.repeat(first_latent_state[None, :, None], model_next_states.shape[0], axis=0), model_next_states],
                axis=-2,
            )
            vaml_loss, log_dict_ = target_daml_loss(
                daml_key,
                model_next_states,
                real_next_states,
                actions,
                reward,
                critic,
                critic_target,
                actor,
                model,
                hyperparams.model_daml_model_steps,
                hyperparams.model_daml_real_env_steps,
                hyperparams.model_daml_target_steps,
                hyperparams.num_critic_samples,
                hyperparams.gamma,
            )
        else:
            vaml_loss, log_dict_ = jax.vmap(
                get_daml_loss(hyperparams), in_axes=(0, 1, 2, None, None, 1)
            )(
                jax.random.split(daml_key, hyperparams.model_daml_model_steps),
                real_next_states,
                model_next_states,
                critic_target,
                actor,
                mask,
            )

        total_loss += vaml_loss.mean()
        log_dict.update(log_dict_)
    else:
        vaml_loss = jnp.zeros_like(model_loss)

    if not hyperparams.use_model_loss:
        total_loss = jnp.zeros_like(total_loss)
        model_loss = jnp.zeros_like(model_loss)
        reward_loss = jnp.zeros_like(reward_loss)
        vaml_loss = jnp.zeros_like(vaml_loss)

    losses = {
        **log_dict,
        "reward_loss": reward_loss,
        "model_loss": model_loss,
        "vaml_loss": vaml_loss.mean(),
    }

    return total_loss, {"total_loss": total_loss, **losses}
