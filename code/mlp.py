from typing import Dict, List, Sequence

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from plotting import plot_vector_field


def mlp_init(layer_sizes: Sequence[int], key: Array) -> List[Dict[str, Array]]:
    subkeys = jax.random.split(key, num=(len(layer_sizes) - 1) * 2)
    params = []
    for i in range(len(layer_sizes) - 1):
        layer = {
            "weights": jax.random.normal(
                subkeys[i], (layer_sizes[i], layer_sizes[i + 1])
            ),
            "bias": jax.random.normal(
                subkeys[i + len(layer_sizes) - 1], (layer_sizes[i + 1],)
            ),
        }
        params.append(layer)
    return params


def mlp_forward(x: Float[Array, "batch dim"], params: List[Dict[str, Array]]):
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def node_forward(
    x0: Float[Array, "batch dim"],
    t: Float[Array, "time"],
    dt0: Float,
    params: List[Dict[str, Array]],
) -> Float[PyTree, "time batch dim"]:
    T0 = t[0]
    T1 = t[-1]
    args = params

    term = diffrax.ODETerm(lambda t, x, args: mlp_forward(x, args))
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t)
    stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    # adjoint = diffrax.BacksolveAdjoint(solver=solver)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=T0,
        t1=T1,
        dt0=dt0,
        y0=x0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
    )
    return solution.ys


if __name__ == "__main__":
    seed = 42069
    key = jax.random.PRNGKey(seed)

    key, subkey = jax.random.split(key)
    params = mlp_init((2, 10, 10, 2), subkey)

    batch_size = 10
    dim = 2

    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (batch_size, dim))
    t = jnp.arange(0.0, 4.0, 1.0)
    dt0 = 0.01
    x = node_forward(x0, t, dt0, params)
    print(x.shape)

    plot_vector_field(lambda x: mlp_forward(x, params))
