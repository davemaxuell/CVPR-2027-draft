from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from mad_td.rl_types import RLBatch
from mad_td.training_util.model_loss_functions import get_n_sample_model_rollout
from mad_td.training_util.rl_targets import (
    bellman_residual_loss,
    get_daml_target_value_estimate,
    old_bellman_residual_loss,
    soft_target_update,
)
from mad_td.cfgs.mad_td_config import (
    MadTdModels,
    MadTdHyperparams,
)


def get_target_fn(hyperparams: MadTdHyperparams):
    loss_f = bellman_residual_loss
    target_f = get_daml_target_value_estimate
    return loss_f, target_f


def critic_loss(
    batch: RLBatch,
    encoder,
    encoder_target,
    critic,
    critic_target,
    actor,
    latent_model,
    hyperparams,
    key,
):
    loss_fn, target_fn = get_target_fn(hyperparams)

    state = batch.state
    next_states = batch.next_state
    action = batch.action
    reward = batch.reward
    mask = batch.mask

    # embed states
    first_state = encoder.apply_fn(encoder.params, state).sample(seed=key)
    next_states = encoder_target.apply_fn(encoder_target.params, next_states).sample(
        seed=key
    )
    state = jnp.concatenate([first_state[:, None], next_states], axis=1)

    # get n step model rollout
    model_pred, _, _ = get_n_sample_model_rollout(
        key=key,
        state=state[:, 0],
        actions=action,
        model=latent_model,
        num_steps=hyperparams.critic_daml_model_steps,
        num_samples=hyperparams.num_stochastic_samples,
    )
    # prepend initial state
    initial_state = jnp.repeat(
        state[None, :, :1], hyperparams.num_stochastic_samples, axis=0
    )
    if model_pred is not None:
        model_pred = jnp.concatenate([initial_state, model_pred], axis=-2)
    else:
        model_pred = initial_state

    # compute targets from real and from on policy model
    proportion_real = int(hyperparams.batch_size * hyperparams.proportion_real)

    # real samples get single step target
    real_state = state[:proportion_real]
    real_action = action[:proportion_real]
    real_reward = reward[:proportion_real]

    model_state = state[proportion_real:]
    model_action = action[proportion_real:]
    model_reward = reward[proportion_real:]

    # split keys
    real_key, model_key, loss_key = jax.random.split(key, 3)

    mask = jnp.concatenate(arrays=[jnp.ones_like(mask[:, :1]), mask], axis=1)
    real_target, real_actions = get_daml_target_value_estimate(
        real_key,
        real_state,
        real_action,
        real_reward,
        critic_target,
        actor,
        latent_model,
        hyperparams.critic_daml_model_steps,
        hyperparams.critic_daml_real_env_steps,
        hyperparams.critic_daml_target_steps,
        hyperparams.num_critic_samples,
        hyperparams.gamma,
    )
    real_actions = jnp.repeat(
        real_actions[None, :], hyperparams.num_stochastic_samples, axis=0
    )
    real_loss, real_loss_dict = loss_fn(
        model_pred[:, :proportion_real, : hyperparams.critic_daml_model_steps + 1, :],
        real_actions[..., : hyperparams.critic_daml_model_steps + 1, :],
        real_target,
        critic,
        mask[:proportion_real, : hyperparams.critic_daml_model_steps + 1],
        loss_key,
        hyperparams,
    )
    model_target, model_actions = get_daml_target_value_estimate(
        model_key,
        model_state,
        model_action,
        model_reward,
        critic_target,
        actor,
        latent_model,
        0,
        0,
        1,
        hyperparams.num_critic_samples,
        hyperparams.gamma,
    )
    model_loss, model_loss_dict = old_bellman_residual_loss(
        jax.lax.stop_gradient(initial_state[0][proportion_real:]),
        model_actions,
        model_target,
        critic,
        mask[proportion_real:, :1],
        loss_key,
        hyperparams,
    )
    loss = (
        hyperparams.proportion_real * real_loss
        + (1 - hyperparams.proportion_real) * model_loss
    )
    loss_dict = jax.tree_util.tree_map(
        lambda x, y: hyperparams.proportion_real * x
        + (1 - hyperparams.proportion_real) * y,
        real_loss_dict,
        model_loss_dict,
    )
    if hyperparams.proportion_real == 1.0:
        return real_loss, real_loss_dict
    elif hyperparams.proportion_real == 0.0:
        return model_loss, model_loss_dict
    return loss, loss_dict


def update_critic(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
) -> Tuple[MadTdModels, Dict]:

    loss_func = jax.value_and_grad(
        # binned_bellman_crossentropy_loss,
        critic_loss,
        argnums=(1, 3),
        has_aux=True,
        allow_int=True,
    )
    (_, loss_dict), grads = loss_func(
        batch,
        models.encoder,
        models.encoder_target,
        models.critic,
        models.critic_target,
        models.actor,
        models.latent_model,
        hyperparams,
        key,
    )

    # update parameters
    new_critic = models.critic.apply_gradients(
        grads=grads[1].params,
    )
    if hyperparams.update_encoder_with_critic_loss:
        new_encoder = models.encoder.apply_gradients(
            grads=grads[0].params,
        )
        new_encoder_target = soft_target_update(
            new_encoder,
            models.encoder_target,
            hyperparams.tau if hyperparams.use_target_encoder else 0.0,
        )
    else:
        new_encoder = models.encoder
        new_encoder_target = models.encoder_target
    new_models = MadTdModels(
        encoder=new_encoder,
        encoder_target=new_encoder_target,
        latent_model=models.latent_model,
        critic=new_critic,
        critic_target=soft_target_update(
            new_critic, models.critic_target, hyperparams.tau
        ),
        actor=models.actor,
    )

    # compute largest eigenvalues of the Hessian
    loss_dict["reward"] = jnp.mean(batch.reward[:, 0])
    loss_dict["reward_std"] = jnp.std(batch.reward[:, 0])

    return new_models, loss_dict
