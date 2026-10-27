import jax
import jax.numpy as jnp
from mad_td.update_functions.actor_updates import update_actor as actor_update_fn
from mad_td.update_functions.joint_update_functions import joint_update_function
from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels

from mad_td.rl_types import (
    RLBatch,
    select_idx_batch,
)


def vmaped_utd_update(
    models,
    batch,
    hyperparams: MadTdHyperparams,
    shape,
    num_seeds,
    num_update_steps,
    key,
    step,
):
    return_dicts = []
    for i in range(num_update_steps):
        key, train_step_key = jax.random.split(key)
        train_step_key = jax.random.split(train_step_key, num_seeds)
        models, return_dict = jax.vmap(
            mad_td_update_step,
            in_axes=(
                shape,
                0,
                None,
                0,
                None,
                None,
                None,
                None,
            ),
        )(
            select_idx_batch(batch, i),
            models,
            hyperparams,
            train_step_key,
            True,
            i == 0 or not hyperparams.slow_actor,
            i == 0 or not hyperparams.slow_model,
            step,
        )
        return_dicts.append(return_dict)

    return models, return_dicts[0]


def mad_td_update_step(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
    update_critic: bool = True,
    update_actor: bool = True,
    update_model: bool = True,
    step: int = 0,
):
    critic_key, actor_key, model_key = jax.random.split(key, 3)

    # batch = truncate_batch(batch, hyperparams.length_training_rollout)
    models, join_loss_dict = joint_update_function(
        batch,
        models,
        hyperparams,
        model_key,
    )
    if update_actor:
        models, actor_loss_dict = actor_update_fn(
            batch=batch,
            models=models,
            hyperparams=hyperparams,
            key=actor_key,
        )
    else:
        actor_loss_dict = {}
    # flatten across all batch dimensions in the logging dict
    info_dict = jax.tree_map(
        lambda x: jnp.mean(x),
        {
            **join_loss_dict,
            "actor": actor_loss_dict,
        },
    )

    return models, info_dict
