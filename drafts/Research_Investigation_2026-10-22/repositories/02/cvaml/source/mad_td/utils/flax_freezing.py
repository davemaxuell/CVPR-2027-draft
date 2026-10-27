from flax.training import train_state
import jax


def freeze_train_state(
    ts: train_state.TrainState,
):
    return train_state.TrainState(
        apply_fn=ts.apply_fn,
        params=jax.lax.stop_gradient(ts.params),
        tx=ts.tx,
        opt_state=ts.opt_state,
        step=ts.step,
    )
