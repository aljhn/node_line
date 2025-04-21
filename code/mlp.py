from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array


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


def mlp_forward(x: Array, params: List[Dict[str, Array]]):
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x
