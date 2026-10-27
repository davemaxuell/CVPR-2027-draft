import jax
from flax.training.train_state import TrainState
from mad_td.cfgs.mad_td_config import MadTdHyperparams
import jax.numpy as jnp


def get_policy_action(
    state: jax.Array,
    encoder: TrainState,
    actor: TrainState,
    critic: TrainState,
    model: TrainState,
    exploration_noise_sigma: jax.Array,
    random: bool,
    hyperparams: MadTdHyperparams,
    rand_key: jax.Array,
):
    act_info_dict = {}
    enc_key, rand_key = jax.random.split(rand_key)

    if getattr(encoder, "batch_stats", None):
        feature, _ = encoder.apply_fn(
            {"params": encoder.params, "batch_stats": encoder.batch_stats},
            state,
            mutable="batch_stats",
        )
    else:
        feature = encoder.apply_fn(encoder.params, state).sample(seed=enc_key)

    act_key, rand_key = jax.random.split(rand_key)
    action = actor.apply_fn(actor.params, feature)[0].sample(seed=act_key)
    exploration_noise = (
        jax.random.normal(rand_key, shape=action.shape) * exploration_noise_sigma
    )
    exploration_noise = jax.numpy.clip(exploration_noise, -0.3, 0.3)
    action = action + exploration_noise
    action = jax.numpy.clip(action, -1.0, 1.0)

    if random:
        action = jax.random.uniform(rand_key, shape=action.shape, minval=-1, maxval=1)
    # assert jax.numpy.all(jax.numpy.isfinite(action))
    return action, act_info_dict


def get_mpc_action(
    state: jax.Array,
    encoder: TrainState,
    actor: TrainState,
    critic: TrainState,
    model: TrainState,
    exploration_noise_sigma: jax.Array,
    random: bool,
    hyperparams: MadTdHyperparams,
    rand_key: jax.Array,
):
    enc_key, rand_key = jax.random.split(rand_key)
    z = encoder.apply_fn(encoder.params, state).sample(seed=enc_key)

    if random:
        act_key, rand_key = jax.random.split(rand_key)
        dummy_action = actor.apply_fn(actor.params, z)[0].sample(seed=act_key)
        action = jax.random.uniform(
            rand_key, shape=dummy_action.shape, minval=-1, maxval=1
        )
        top_k_softmax = jnp.ones(hyperparams.mpc_top_k) / hyperparams.mpc_top_k
        return action, {}

    act_key, rand_key = jax.random.split(rand_key)
    policy_action_dist, temp = actor.apply_fn(actor.params, z, rngs={"random": act_key})
    policy_action = policy_action_dist.sample(seed=act_key)

    z = z[jnp.newaxis, ...]
    z = jnp.repeat(z, hyperparams.num_mpc_samples + 1, axis=0)

    mean = policy_action
    std = jnp.ones_like(mean) * exploration_noise_sigma
    softmax_per_round = []
    stds_per_round = []
    reward_per_round, unc_per_round = [], []

    for _ in range(hyperparams.mpc_iterations):
        traj_key, rand_key = jax.random.split(rand_key)

        random_actions = (
            mean
            + jax.random.normal(
                traj_key, shape=[hyperparams.num_mpc_samples, *policy_action.shape]
            )
            * std
        )

        actions = jnp.concatenate([policy_action[jnp.newaxis], random_actions], axis=0)
        actions = jnp.clip(actions, -1, 1)
        if hyperparams.use_cheap_mpc:
            values1, values2 = critic.apply_fn(critic.params, z, actions)
            values = (values1 + values2) / 2
        else:
            rewards = 0.0
            next_state_latent = z
            next_actions = actions
            for i in range(hyperparams.mpc_rollout_length):
                model_key, rand_key = jax.random.split(rand_key)
                next_state_latent_dist, reward = model.apply_fn(
                    model.params, next_state_latent, next_actions
                )

                next_state_latent = next_state_latent_dist.sample(
                    seed=model_key,
                    sample_shape=[
                        hyperparams.num_stochastic_samples_mpc,
                    ],
                )
                next_actions_dist, temp = actor.apply_fn(
                    actor.params, next_state_latent
                )
                next_actions, log_prob = next_actions_dist.sample_and_log_prob(
                    seed=act_key
                )

                rewards += (
                    reward[None]
                    - log_prob.mean(-1, keepdims=True) * temp * hyperparams.gamma
                ) * hyperparams.gamma ** i
                act_key, rand_key = jax.random.split(rand_key)

            q1, q2 = critic.apply_fn(critic.params, next_state_latent, next_actions)

            reward_per_round.append(reward)

            if len(z.shape) < len(q1.shape):
                if hyperparams.stoch_agg == "mean":
                    agg_fn = jnp.mean
                elif hyperparams.stoch_agg == "median":
                    agg_fn = jnp.median
                elif "percentile" in hyperparams.stoch_agg:
                    agg_fn = lambda x, axis: jnp.percentile(
                        x, q=int(hyperparams.stoch_agg.split("_")[-1]), axis=axis
                    )
                elif "ucb" in hyperparams.stoch_agg:
                    agg_fn = lambda x, axis: jnp.mean(x, axis=axis) + float(
                        hyperparams.stoch_agg.split("_")[-1]
                    ) * jnp.std(x, axis=axis)
                else:
                    raise NotImplementedError(
                        f"Expected stoch_agg to be one of 'mean', 'median', 'percentile_*', got {hyperparams.stoch_agg} instead"
                    )

                q1 = agg_fn(q1.reshape([-1, *q1.shape[-2:]]), axis=0)
                q2 = agg_fn(q2.reshape([-1, *q2.shape[-2:]]), axis=0)

                if len(z.shape) < len(rewards.shape):
                    rewards = agg_fn(rewards.reshape([-1, *rewards.shape[-2:]]), axis=0)

            if hyperparams.mpc_v == "max":
                next_q = jnp.max(jnp.stack([q1, q2], axis=0), axis=0)
            elif hyperparams.mpc_v == "min":
                next_q = jnp.min(jnp.stack([q1, q2], axis=0), axis=0)
            elif hyperparams.mpc_v == "mean":
                next_q = (q1 + q2) / 2
            values = (
                rewards + hyperparams.gamma ** hyperparams.mpc_rollout_length * next_q
            )

        elite_values, elite_idx = jax.lax.top_k(values.squeeze(), hyperparams.mpc_top_k)
        elite_weights = jax.nn.softmax(
            elite_values / hyperparams.mpc_temperature, axis=0
        )[:, jnp.newaxis]

        mean = jnp.sum(elite_weights * actions[elite_idx], axis=0, keepdims=True)
        std = jnp.sqrt(
            jnp.sum(
                elite_weights * jnp.square(actions[elite_idx] - mean),
                axis=0,
                keepdims=True,
            )
        )
        std = jnp.clip(
            std,
            hyperparams.mpc_lower_std_bound,
            hyperparams.mpc_upper_std_bound,
        )
        stds_per_round.append(jnp.mean(std))

        top_k_softmax = jax.nn.softmax(
            elite_values / hyperparams.mpc_temperature, axis=0
        )
        top = jax.random.choice(rand_key, elite_idx, p=top_k_softmax)
        softmax_per_round.append(
            -jnp.sum(top_k_softmax * jnp.log(top_k_softmax + 1e-8))
        )
    if hyperparams.mpc_add_final_noise:
        action = actions[top] + std[0] * jax.random.normal(
            rand_key, shape=actions[top].shape
        )
    else:
        action = actions[top]

    return action, {}
