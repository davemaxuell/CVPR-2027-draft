import jax
import jax.numpy as jnp

from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels
from mad_td.rl_types import RLBatch


def compute_noise(
    models: MadTdModels,
    rb_batch: RLBatch,
    key: jax.Array,
    hyperparams: MadTdHyperparams,
):
    log_dict = {}

    encoder = models.encoder
    actor = models.actor
    critic = models.critic
    model = models.latent_model

    N_SAMPLES = 32

    rb_states = rb_batch.state

    enc_key, model_key, rand_key = jax.random.split(key, 3)

    def get_act_val(rb_states, key):
        enc_key, act_key = jax.random.split(key)
        features = encoder.apply_fn(encoder.params, rb_states).sample(seed=enc_key)
        actions = actor.apply_fn(actor.params, features)[0].sample(seed=act_key)
        values = critic.apply_fn(
            critic.params, features, actions
        )  # [N, 1, B, 1], [N, 1, B, 1]
        return actions, values, features

    actions, values, features = jax.vmap(get_act_val, in_axes=(None, 0))(
        rb_states,
        jax.random.split(enc_key, N_SAMPLES),
    )

    mean_value = 0.5 * (values[0] + values[1])
    log_dict.update(
        {
            "policy_actions/encoder/action_std": jnp.mean(jnp.std(actions, 0)),
            "policy_actions/encoder/var_of_mean_action": jnp.mean(
                jnp.std(jnp.mean(actions, 0), 1)
            ),
            "policy_actions/encoder/values_std": jnp.mean(jnp.std(mean_value, 0)),
            "policy_actions/encoder/var_of_mean_value": jnp.mean(
                jnp.std(jnp.mean(mean_value, 0), 1)
            ),
            # "policy_actions/encoder/latents_std": jnp.mean(jnp.std(features, 0)),
        }
    )

    features = features[0]
    actions = actions[0]

    def get_model_act_val(features, actions, key):
        model_key, act_key = jax.random.split(key)
        next_features = model.apply_fn(model.params, features, actions)[0].sample(
            seed=model_key
        )

        next_actions = actor.apply_fn(actor.params, next_features)[0].sample(
            seed=act_key
        )
        next_values = critic.apply_fn(critic.params, next_features, next_actions)
        return next_actions, next_values

    next_actions, next_values = jax.vmap(get_model_act_val, in_axes=(None, None, 0))(
        features, actions, jax.random.split(model_key, N_SAMPLES)
    )

    mean_value = 0.5 * (next_values[0] + next_values[1])

    log_dict.update(
        {
            "policy_actions/model/action_std": jnp.mean(jnp.std(next_actions, 0)),
            "policy_actions/model/var_of_mean_action": jnp.mean(
                jnp.std(jnp.mean(next_actions, 0), 1)
            ),
            "policy_actions/model/values_std": jnp.mean(jnp.std(mean_value, 0)),
            "policy_actions/model/var_of_mean_value": jnp.mean(
                jnp.std(jnp.mean(mean_value, 0), 1)
            ),
            # "policy_actions/model/latents_std": jnp.mean(jnp.std(next_features, 0)),
        }
    )

    return log_dict
