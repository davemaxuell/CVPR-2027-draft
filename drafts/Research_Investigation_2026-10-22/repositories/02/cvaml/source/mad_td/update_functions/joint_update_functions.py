import jax
import jax.numpy as jnp

from mad_td.cfgs.mad_td_config import MadTdHyperparams, MadTdModels
from mad_td.rl_types import RLBatch
from mad_td.training_util.model_loss_functions import compute_model_loss
from mad_td.training_util.rl_targets import soft_target_update
from mad_td.update_functions.critic_updates import critic_loss
from mad_td.utils.flax_freezing import freeze_train_state


def joint_loss_function(
    models: MadTdModels,
    batch: RLBatch,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
):
    if hyperparams.update_model:
        model_loss_value, model_loss_dict = compute_model_loss(
            batch.state,
            batch.action,
            batch.reward,
            batch.next_state,
            batch.mask,
            models.encoder,
            freeze_train_state(models.encoder_target),
            models.latent_model,
            freeze_train_state(models.critic),
            freeze_train_state(models.critic_target),
            freeze_train_state(models.actor),
            key,
            hyperparams,
        )
    else:
        model_loss_value = 0.0
        model_loss_dict = {}
    critic_enc = (
        models.encoder
        if hyperparams.update_encoder_with_critic_loss
        else freeze_train_state(models.encoder)
    )
    critic_loss_value, critic_loss_dict = critic_loss(
        batch,
        critic_enc,
        freeze_train_state(models.encoder_target),
        models.critic,
        freeze_train_state(models.critic_target),
        freeze_train_state(models.actor),
        freeze_train_state(models.latent_model),
        hyperparams,
        key,
    )
    loss = model_loss_value + critic_loss_value
    return loss, {
        "model": model_loss_dict,
        "critic": critic_loss_dict,
    }


def joint_update_function(
    batch: RLBatch,
    models: MadTdModels,
    hyperparams: MadTdHyperparams,
    key: jax.Array,
):
    

    (_, loss_dict), grads = jax.value_and_grad(
        joint_loss_function,
        has_aux=True,
        allow_int=True,
    )(models, batch, hyperparams, key)

    lm_grad, enc_grad, cr_grad = \
        grads.latent_model.params, \
        grads.encoder.params, \
        grads.critic.params

    new_encoder = models.encoder.apply_gradients(grads=enc_grad)
    new_encoder_target = soft_target_update(
        new_encoder,
        models.encoder_target,
        hyperparams.tau if hyperparams.use_target_encoder else 0.0,
    )
    new_critic = models.critic.apply_gradients(grads=cr_grad)
    new_critic_target = soft_target_update(
        new_critic, models.critic_target, hyperparams.tau
    )
    new_model = models.latent_model.apply_gradients(grads=lm_grad)
    new_models = MadTdModels(
        encoder=new_encoder,
        encoder_target=new_encoder_target,
        latent_model=new_model,
        critic=new_critic,
        critic_target=new_critic_target,
        actor=models.actor,
    )
    return new_models, loss_dict
