from typing import Dict, Tuple, Union
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from mad_td.utils.jax import get_ij_slice
from mad_td.cfgs.mad_td_config import MadTdHyperparams
from mad_td.nn.common import RSNormTrainState


def compute_td_target_from_state(
    state: jax.Array,
    reward: jax.Array,
    critic: TrainState,
    actor: TrainState,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
):
    actor_key, key = jax.random.split(key)
    action_dist, temp = actor.apply_fn(actor.params, state)
    action, log_probs = action_dist.sample_and_log_prob(seed=actor_key)
    value = critic.apply_fn(critic.params, state, action)
    value = jnp.stack(value, axis=0)
    value = value.min(axis=0) - log_probs.mean() * temp
    td_target = reward + hyperparams.gamma * value
    return td_target


def get_daml_target_value_estimate(
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
):
    """
    Compute unified IterVAML / MuZero target with optional model rollout

    Args:
        key: PRNG key
        states: [B, T, S] state tensor
        actions: [B, T, A] action tensor
        rewards: [B, T, 1] reward tensor
        critic: critic model
        actor: actor model
        model: model model
        model_steps: number of model rollout steps
        real_env_steps: number of real environment steps
        model_target_steps: number of additional model steps for target computation
        gamma: discount factor
    """

    # get ij slices
    states_slice = get_ij_slice(states, model_steps + 1, real_env_steps + 1)
    actions_slice = get_ij_slice(actions, model_steps + 1, real_env_steps)
    rewards_slice = get_ij_slice(rewards, model_steps + 1, real_env_steps)

    # expand if model_target_steps > 0
    if model_target_steps > 0:
        model_rewards = []
        model_actions = []

        state = states_slice[..., -1, :]
        state = jnp.repeat(state[None], num_critic_samples, axis=0)
        for _ in range(model_target_steps):
            key, subkey = jax.random.split(key)
            action = actor.apply_fn(actor.params, state)[0].sample(seed=subkey)

            key, subkey = jax.random.split(key)
            next_state_dist, reward = model.apply_fn(model.params, state, action)
            next_state = next_state_dist.sample(seed=subkey)

            model_rewards.append(reward.mean(0))
            model_actions.append(action.mean(0))

            state = next_state

        final_state = state

        model_rewards = jnp.stack(model_rewards, axis=-2)
        model_actions = jnp.stack(model_actions, axis=-2)

        rewards = jnp.concatenate([rewards_slice, model_rewards], axis=-2)
        actions = jnp.concatenate([actions_slice, model_actions], axis=-2)
        final_action = actor.apply_fn(actor.params, final_state)[0].sample(seed=key)
        value = jnp.min(
            jnp.stack(
                critic_target.apply_fn(critic_target.params, final_state, final_action),
                axis=0,
            ),
            axis=0,
        ).mean(0)
        value = (gamma ** (real_env_steps + model_target_steps)) * value
    else:
        final_state = states_slice[..., -1, :]
        rewards = rewards_slice
        actions = actions_slice
        final_action = actor.apply_fn(actor.params, final_state)[0].sample(seed=key)
        value = jnp.min(
            jnp.stack(
                critic_target.apply_fn(critic_target.params, final_state, final_action),
                axis=0,
            ),
            axis=0,
        )
        value = (gamma ** (real_env_steps + model_target_steps)) * value

    # compute discounted reward sum
    rewards = jnp.sum(
        rewards * gamma ** jnp.arange(real_env_steps + model_target_steps)[:, None],
        axis=-2,
    )
    return jax.lax.stop_gradient(rewards + value), jax.lax.stop_gradient(
        actions[..., 0, :]
    )


def old_bellman_residual_loss(
    state: jax.Array,
    action: jax.Array,
    value_target: jax.Array,
    critic: TrainState,
    mask: jax.Array,
    key: jax.Array,
    hyperparams: MadTdHyperparams,
):
    q1, q2 = critic.apply_fn(critic.params, state, action)
    critic_loss = jnp.mean(mask * (q1 - value_target) ** 2) + jnp.mean(
        mask * (q2 - value_target) ** 2
    )
    return jnp.mean(critic_loss), {
        "critic_loss": critic_loss,
        "q1": q1.mean(),
        "q2": q2.mean(),
    }


def bellman_residual_loss(
    state: jax.Array,
    action: jax.Array,
    value_target: jax.Array,
    critic: TrainState,
    mask: jax.Array,
    key: jax.Array,
    hyperparams: MadTdHyperparams,
    mean: bool = True,
) -> Tuple[jax.Array, Dict]:
    num_samples = state.shape[0]

    q1, q2 = critic.apply_fn(critic.params, state, action)

    mean_q1 = jnp.mean(q1, axis=0)
    mean_q2 = jnp.mean(q2, axis=0)
    if num_samples > 1:
        var_q1 = (1.0 / num_samples) * jnp.var(q1, axis=0, ddof=0)
        var_q2 = (1.0 / num_samples) * jnp.var(q2, axis=0, ddof=0)
    else:
        var_q1 = jnp.array([0.0])
        var_q2 = jnp.array([0.0])

    critic_loss = (
        jnp.mean(mask * (mean_q1 - value_target) ** 2)
        + jnp.mean(mask * (mean_q2 - value_target) ** 2)
        - jnp.mean(mask * (var_q1 + var_q2))
    )

    return (
        jnp.mean(critic_loss),
        {  # type: ignore
            "critic_loss": critic_loss,
            "q1": mean_q1.mean(),
            "q2": mean_q2.mean(),
        },
    )


def crossentropy(pred, target, weight):
    return -jnp.mean(weight * (target * pred).mean(-1).reshape(weight.shape))


def soft_target_update(
    source: Union[RSNormTrainState, TrainState],
    target: Union[RSNormTrainState, TrainState],
    tau: float,
) -> TrainState:
    source_params = source.params
    target_params = target.params
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: tau * x + (1 - tau) * y, target_params, source_params
    )

    if getattr(source, "batch_stats", None):
        new_target = RSNormTrainState.create(
            apply_fn=target.apply_fn,
            params=new_target_params,
            batch_stats=source.batch_stats,
            tx=target.tx,
        )
    else:
        new_target = TrainState.create(
            apply_fn=target.apply_fn,
            params=new_target_params,
            tx=target.tx,
        )
    return new_target
