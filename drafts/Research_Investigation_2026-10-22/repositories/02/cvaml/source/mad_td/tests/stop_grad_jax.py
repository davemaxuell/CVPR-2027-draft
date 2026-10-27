from typing import Callable
import flax

import flax.struct
import jax
import jax.numpy as jnp


class MinimalNN(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        return flax.linen.Dense(2)(x)


def loss(ts_1, ts_2, x):
    return ts_1["fn"](ts_1["params"], ts_2["fn"](ts_2["params"], x))


def setup():
    ts_1 = MinimalNN()
    ts_2 = MinimalNN()
    params_1 = ts_1.init(jax.random.PRNGKey(0), jnp.ones((2, 2)))
    params_2 = ts_2.init(jax.random.PRNGKey(0), jnp.ones((2, 2)))
    ts_1 = {"params": params_1, "fn": ts_1.apply}
    ts_2 = {"params": params_2, "fn": ts_2.apply}

    return ts_1, ts_2


def compute_grad(ts_1, ts_2, x):
    grad_1 = jax.grad(loss, argnums=0, allow_int=True)(
        ts_1, jax.lax.stop_gradient(ts_2), x
    )
    grad_2 = jax.grad(loss, argnums=1, allow_int=True)(
        jax.lax.stop_gradient(ts_1), ts_2, x
    )
    return grad_1, grad_2


def test():
    ts_1, ts_2 = setup()
    x = jnp.ones((2, 2))
    grad_1, grad_2 = compute_grad(ts_1, ts_2, x)
    print(grad_1, grad_2)


if __name__ == "__main__":
    test()
