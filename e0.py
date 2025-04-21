import random

import jax
import jax.numpy as jnp
import numpy as np
import optax

from node_line.mlp import mlp_forward, mlp_init, node_forward
from node_line.plotting import plot_vector_field
from node_line.vector_fields import MassSpringDamper, SinglePendulum


def loss(x_true, x_pred):
    assert x_true.shape == x_pred.shape
    return optax.l2_loss(x_pred, x_true).mean()


def step(params, t, x):
    x0 = x[0, :, :]
    dt0 = t[1] - t[0]
    x_pred = node_forward(x0, t, dt0, params)
    return loss(x, x_pred)


@jax.value_and_grad
def train_step(params, t, x):
    loss_value = step(params, t, x)
    return loss_value


def main():
    seed = 42069
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    dim = 2
    # m = 1.0
    # d = 1.0
    # k = 1.0
    # system = MassSpringDamper(m, d, k)
    l = 1.0
    g = 9.81
    system = SinglePendulum(l, g)
    system.plot()
    exit()
    # if dim == 2:
    #     system.plot()

    T0 = 0.0
    T1 = 1.0
    dt = 0.01
    t = jnp.arange(T0, T1, dt)

    batch_size = 20

    key, subkey = jax.random.split(key)
    x0 = jax.random.uniform(subkey, shape=(batch_size, dim), minval=-10.0, maxval=10.0)
    x = system.generate(x0, t, dt)

    layer_sizes = (dim, 50, 50, dim)
    key, subkey = jax.random.split(key)
    params = mlp_init(layer_sizes, subkey)

    initial_learning_rate = 1e-3
    learning_rate = optax.schedules.constant_schedule(initial_learning_rate)
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train(params, opt_state, t, x):
        loss_value, loss_grads = train_step(params, t, x)
        updates, opt_state = optimizer.update(loss_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss_value, opt_state

    epochs = 200
    train_losses = np.zeros(epochs)

    for epoch in range(1, epochs + 1):
        try:
            key, subkey = jax.random.split(key)
            x0 = jax.random.uniform(subkey, shape=(batch_size, dim), minval=-10.0, maxval=10.0)
            x = system.generate(x0, t, dt)

            params, train_loss, opt_state = train(params, opt_state, t, x)
            train_losses[epoch - 1] = train_loss.item()
            print(f"Epoch: {epoch:4d}/{epochs}, Loss: {train_loss.item():.4f}")

        except KeyboardInterrupt:
            break

    plot_vector_field(lambda x: mlp_forward(x, params), "temp_filename")


if __name__ == "__main__":
    main()
