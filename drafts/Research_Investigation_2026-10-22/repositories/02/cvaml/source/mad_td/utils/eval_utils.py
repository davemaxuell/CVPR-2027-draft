import functools
from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np

from mad_td.cfgs import mad_td_config
from mad_td.cfgs.mad_td_config import MadTdModels
from mad_td.rl_types import RLBatch, select_idx_batch
from mad_td.utils.jax import multi_softmax
from mad_td.update_functions.critic_updates import get_target_fn


@functools.partial(jax.jit, static_argnames=["hyperparams"])
def get_action(obs, models, hyperparams, key):
    enc_key, act_key, key = jax.random.split(key, 3)

    latent = models.encoder.apply_fn(
        models.encoder.params, obs, rngs={"random": enc_key}
    )
    action = models.actor.apply_fn(models.actor.params, latent)[0].sample(seed=act_key)
    return action


def get_batch_on_policy_tuples(batch: RLBatch, models, env, hyperparams, key):
    assert batch.physics_state is not None  # WHY?
    reset_fn = env.get_reset()
    step_fn = env.get_step(jax.random.PRNGKey(0))
    actions = jax.vmap(get_action, in_axes=(0, 0, None, None))(
        batch.state, models, hyperparams, key
    )
    rewards_ = []
    next_states_ = []
    for i in range(batch.state.shape[1]):
        keys = [jax.random.PRNGKey(0) for _ in range(batch.state.shape[0])]

        reset_dict = [
            {"internal_state": batch.physics_state[j, i]}
            for j in range(batch.physics_state.shape[0])
        ]
        env_state = reset_fn(keys, options=reset_dict)
        env_state, reward, done = step_fn(keys, env_state, actions[:, i])
        rewards_.append(reward)
        next_states_.append(env_state.obs)
    rewards = np.stack(rewards_, axis=1)
    next_states = np.stack(next_states_, axis=1)
    return RLBatch(
        state=batch.state,
        action=actions.reshape(*batch.action.shape),
        reward=rewards.reshape(*batch.reward.shape),
        next_state=next_states.reshape(*batch.next_state.shape),
        mask=batch.mask,
        idxs=batch.idxs,
        physics_state=batch.physics_state,
    )


@functools.partial(jax.jit, static_argnums=(2))
def critic_loss(batch: RLBatch, models, hyperparams, key):
    loss_fn, target_fn = get_target_fn(hyperparams)

    state = batch.state
    action = batch.action[:, 0]
    reward = batch.reward[:, 0]
    next_state = batch.next_state[:, 0]
    mask = batch.mask[:, 0]

    loss_dict = {}

    # compute targets from real and from on policy model
    proportion_real = int(hyperparams.batch_size * hyperparams.proportion_real)

    enc_key, tar_key, lat_key, act_key, key = jax.random.split(key, 5)
    real_latent_state = models.encoder.apply_fn(
        models.encoder.params, state[proportion_real:]
    ).sample(seed=enc_key)
    real_latent_next_state = models.encoder_target.apply_fn(
        models.encoder_target.params,
        next_state[:proportion_real],
        rngs={"random": tar_key},
    )
    latent_state = models.encoder.apply_fn(
        models.encoder.params, state, rngs={"random": lat_key}
    )

    model_actions = models.actor.apply_fn(models.actor.params, real_latent_state)[
        0
    ].sample(seed=act_key)

    model_key, key = jax.random.split(key)
    model_latent_next_state, model_reward = models.latent_model.apply_fn(
        models.latent_model.params,
        real_latent_state,
        model_actions,
        rngs={"random": model_key},
    )
    # model_latent_next_state = jnp.log(model_latent_next_state + 1e-8)

    action = jnp.concatenate([action[:proportion_real], model_actions], axis=0)
    reward = jnp.concatenate([reward[:proportion_real], model_reward], axis=0)
    latent_next_state = jnp.concatenate(
        [real_latent_next_state, model_latent_next_state], axis=0
    )

    target_func = jax.vmap(target_fn, in_axes=(0, 0, None, None, None, 0))
    target_key, loss_key = jax.random.split(key, 2)
    key = jax.random.split(target_key, latent_next_state.shape[0])
    target = target_func(
        latent_next_state,
        reward,
        models.critic_target,
        models.actor,
        hyperparams,
        key,
    )

    (_, loss_dict) = loss_fn(
        latent_state,
        action,
        target,
        models.critic,
        mask,
        target_key,
        hyperparams,
    )
    return loss_dict


@functools.partial(jax.jit, static_argnums=(2))
def model_loss(batch: RLBatch, models, hyperparams, key):
    encoder = models.encoder
    encoder_target = models.encoder_target
    model = models.latent_model
    critic = models.critic
    actor = models.actor

    state = batch.state
    next_states = batch.next_state
    actions = batch.action
    mask = batch.mask
    reward = batch.reward

    if hyperparams.use_stochastic_encoder:
        enc_key, key = jax.random.split(key)
        encoder_alpha, first_latent_state = sample_parametric_dirichlet(
            encoder, enc_key, state, get_params=True
        )
    else:
        first_latent_state: jax.Array = encoder.apply_fn(encoder.params, state)  # type: ignore

    latent_state = first_latent_state

    model_rewards_ = []
    model_next_states_ = []
    model_next_state_alpha_ = []

    for i in range(hyperparams.length_training_rollout):
        key, dir_key = jax.random.split(key)
        latent_dir_params, (latent_state, pred_rewards) = sample_parametric_dirichlet(
            model, dir_key, latent_state, actions[:, i], get_params=True
        )
        model_next_state_alpha_.append(latent_dir_params)
        model_next_states_.append(latent_state)
        model_rewards_.append(pred_rewards)

    model_next_states = jnp.stack(model_next_states_, axis=1)  # [N, T, F]
    model_next_state_alpha = jnp.stack(model_next_state_alpha_, axis=1)  # [N, T, F]
    model_rewards = jnp.stack(model_rewards_, axis=1)

    if hyperparams.use_stochastic_encoder:
        key, enc_key, next_lat_key = jax.random.split(key, 3)

        latent_next_alphas, latent_next_states = sample_parametric_dirichlet(
            encoder_target, enc_key, next_states, get_params=True
        )
        latent_next_states = jax.lax.stop_gradient(latent_next_states)

        enc_next_latent_states = sample_parametric_dirichlet(
            encoder, next_lat_key, next_states
        )

        model_loss = (
            jax.vmap(byol_dir_kl, in_axes=(1, 1, 1))(
                model_next_state_alpha,
                jax.lax.stop_gradient(latent_next_alphas),
                mask,
            ).mean()
            * hyperparams.model_loss_scaler
        )
    else:
        latent_next_states_logits: jax.Array = jax.lax.stop_gradient(
            encoder_target.apply_fn(encoder_target.params, next_states, get_logits=True)
        )  # type: ignore
        latent_next_states = multi_softmax(latent_next_states_logits)
        enc_next_latent_states = encoder.apply_fn(encoder.params, next_states)

        if hyperparams.model_loss == "likelihood":
            model_loss_fn = byol_dir_likelihood
        elif hyperparams.model_loss == "bayesian_risk":
            model_loss_fn = byol_dir_bayesian_risk
        else:
            raise NotImplementedError(
                f"Expected one off 'likelihood', 'bayesian_risk', got {hyperparams.model_loss} instead"
            )

        model_loss = (
            jax.vmap(model_loss_fn, in_axes=(1, 1, 1, None))(
                model_next_state_alpha, latent_next_states_logits, mask, 0.0
            ).mean()
            * hyperparams.model_loss_scaler
        )

    act_key, key = jax.random.split(key)
    pred_action = models.actor.apply_fn(models.actor.params, latent_next_states)[
        0
    ].sample(seed=act_key)

    # temporary_logging
    value = critic.apply_fn(critic.params, latent_next_states, pred_action)
    value = jnp.stack(value, axis=0)

    # compute losses
    reward_loss = jnp.mean(mask * (reward - model_rewards) ** 2)

    losses = {
        "model_reward_loss": reward_loss,
        "model_loss": model_loss,
        "dirichlet/entropy/model": jnp.mean(
            dirichlet_entropy(model_next_state_alpha.reshape([-1, 8]))
        ),
        "dirichlet/alpha_0/model": jnp.mean(
            model_next_state_alpha.reshape(
                [model_next_state_alpha.shape[0], -1, 8]
            ).sum(-1)
        ),
        "dirichlet/alpha_max/model": jnp.max(model_next_state_alpha),
        "dirichlet/alpha_min/model": jnp.min(model_next_state_alpha),
        "uncertainty/max_alpha/model": jnp.mean(
            jnp.max(model_next_state_alpha.reshape([-1, 8]), -1)
            / jnp.sum(model_next_state_alpha.reshape([-1, 8]), -1)
        ),
        "uncertainty/epkl/model": jnp.mean(
            (7.0) / jnp.sum(model_next_state_alpha.reshape([-1, 8]), -1)
        ),
        "model_bias": jnp.mean(
            model.params["params"]["DirichletMLP_0"]["Layer_2"]["Dense_0"]["bias"]
        ),
        "model_q_mean": jnp.mean(value),
        "encoder_entropy": -jnp.mean(
            first_latent_state * jnp.log(first_latent_state + 1e-8)
        )
        * 8,  # factor 8 due to the multi_softmax
    }

    if hyperparams.use_stochastic_encoder:
        losses.update(
            {
                "dirichlet/entropy/encoder": jnp.mean(
                    dirichlet_entropy(encoder_alpha.reshape([-1, 8]))
                ),
                "dirichlet/alpha_0/encoder": jnp.mean(
                    encoder_alpha.reshape([encoder_alpha.shape[0], -1, 8]).sum(-1)
                ),
                "dirichlet/alpha_max/encoder": jnp.max(encoder_alpha),
                "dirichlet/alpha_min/encoder": jnp.min(encoder_alpha),
                "uncertainty/max_alpha/encoder": jnp.mean(
                    jnp.max(encoder_alpha.reshape([-1, 8]), -1)
                    / jnp.sum(encoder_alpha.reshape([-1, 8]), -1)
                ),
                "uncertainty/epkl/encoder": jnp.mean(
                    (7.0) / jnp.sum(encoder_alpha.reshape([-1, 8]), -1)
                ),
            }
        )

    return losses


def eval_on_policy_batches(
    batches: RLBatch,
    models: MadTdModels,
    env,
    algo_cfg: mad_td_config.MadTdHyperparams,
    batch_shape,
):
    algo_cfg = algo_cfg.replace(proportion_real=1.0)

    buffer_loss_dicts = []
    on_policy_loss_dicts = []

    for i in tqdm(range(batches.state.shape[1])):
        buffer_batch = select_idx_batch(batches, i)
        on_policy_batch = get_batch_on_policy_tuples(
            buffer_batch, models, env, algo_cfg, jax.random.PRNGKey(0)
        )

        buffer_loss_dict = jax.vmap(critic_loss, in_axes=(batch_shape, 0, None, None))(
            buffer_batch, models, algo_cfg, jax.random.PRNGKey(0)
        )
        on_policy_loss_dict = jax.vmap(
            critic_loss, in_axes=(batch_shape, 0, None, None)
        )(on_policy_batch, models, algo_cfg, jax.random.PRNGKey(0))

        if algo_cfg.use_stochastic_model and algo_cfg.prior == "dirichlet":
            buffer_loss_model_dict = jax.vmap(
                model_loss, in_axes=(batch_shape, 0, None, None)
            )(buffer_batch, models, algo_cfg, jax.random.PRNGKey(0))
            on_policy_loss_model_dict = jax.vmap(
                model_loss, in_axes=(batch_shape, 0, None, None)
            )(on_policy_batch, models, algo_cfg, jax.random.PRNGKey(0))

            buffer_loss_dict.update(buffer_loss_model_dict)
            on_policy_loss_dict.update(on_policy_loss_model_dict)

        buffer_loss_dicts.append(buffer_loss_dict)
        on_policy_loss_dicts.append(on_policy_loss_dict)

    return [
        {f"eval_buffer/{k}": v for k, v in buffer_loss_dict.items()}
        for buffer_loss_dict in buffer_loss_dicts
    ], [
        {f"eval_on_policy/{k}": v for k, v in on_policy_loss_dict.items()}
        for on_policy_loss_dict in on_policy_loss_dicts
    ]
