from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels

from mad_td.rl_types import RLBatch


def update_actor(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
) -> Tuple[MadTdModels, Dict]:
    state = batch.state
    batch_action = batch.action[:, 0]

    enc_key, key = jax.random.split(key)
    latent_state = models.encoder.apply_fn(models.encoder.params, state).sample(
        seed=enc_key
    )

    def _j_sac_and_td3(actor_fw, state, random_key):
        action_dist, temp = actor_fw.apply_fn(actor_fw.params, state)

        action, log_probs = action_dist.sample_and_log_prob(seed=random_key)
        entropy_loss = jnp.mean(log_probs * jax.lax.stop_gradient(temp))
        temp_loss = (
            temp * (-jax.lax.stop_gradient(log_probs) + action.shape[-1] * 0.5).mean()
        )

        target = models.critic_target.apply_fn(
            models.critic_target.params, state, action
        )
        target = jnp.stack(target, axis=0)
        target = jnp.min(target, axis=0)

        log_dict = {}

        log_dict.update(
            {
                "mean_action": jnp.mean(action),
                "mean_abs_action": jnp.mean(jnp.abs(action)),
                "action_churn": jnp.mean(jnp.square(action - batch_action)),
                "entropy loss": jnp.mean(entropy_loss),
                "temperature loss": jnp.mean(temp_loss),
                "actor_loss": -jnp.mean(target),
            }
        )

        loss = jnp.mean(-target + temp_loss + entropy_loss)
        return loss, log_dict

    loss_grad = jax.value_and_grad(
        _j_sac_and_td3, argnums=0, has_aux=True, allow_int=True
    )

    actor_key, key = jax.random.split(key)
    (target, loss_dict), grad = loss_grad(
        models.actor,
        latent_state,
        actor_key,
    )

    act_grad = grad.params
    # update parameters

    new_actor = models.actor.apply_gradients(grads=act_grad)
    new_models = MadTdModels(
        encoder=models.encoder,
        encoder_target=models.encoder_target,
        latent_model=models.latent_model,
        critic=models.critic,
        critic_target=models.critic_target,
        actor=new_actor,
    )

    return new_models, loss_dict
