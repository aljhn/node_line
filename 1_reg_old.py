import random
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax


def mass_spring_damper(t, x, args):
    m, d, k = args
    x1 = x[:, 0]
    x2 = x[:, 1]
    dx1 = x2
    dx2 = -k / m * x1 - d / m * x2
    return jnp.stack((dx1, dx2), axis=1)


def van_der_pol(t, x, args):
    mu = args
    x1 = x[:, 0]
    x2 = x[:, 1]
    dx1 = x2
    dx2 = mu * (1.0 - (x1**2.0)) * x2 - x1
    return jnp.stack((dx1, dx2), axis=1)


def generate_data(function, args, t, x0):
    T0 = t[0]
    T1 = t[-1]
    h = t[1] - t[0]
    saveat = diffrax.SaveAt(ts=t)
    term = diffrax.ODETerm(function)
    solver = diffrax.Dopri5()
    solution = diffrax.diffeqsolve(term, solver, t0=T0, t1=T1, dt0=h, y0=x0, saveat=saveat, args=args, adjoint=diffrax.RecursiveCheckpointAdjoint())
    return solution.ys


def plot_vector_field(vector_field_function, width=10, step=0.1):
    X = jnp.arange(-width, width, step)
    X1, X2 = jnp.meshgrid(X, X, indexing="xy")
    XX = jnp.stack((X1.flatten(), X2.flatten()), axis=1)

    YY = vector_field_function(XX)

    Y1 = YY[:, 0].reshape(X1.shape)
    Y2 = YY[:, 1].reshape(X2.shape)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)

    plt.figure()
    plt.streamplot(X1, X2, Y1, Y2, density=1, linewidth=None, color="#A23BEC")
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses):
    epochs = train_losses.shape[0]
    epochs = np.arange(1, epochs + 1, 1)
    plt.figure()
    plt.plot(epochs, train_losses, val_losses)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Loss")
    plt.legend(["Train", "Val"])
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


def model_init(model_def, key):
    subkeys = jax.random.split(key, num=(len(model_def) - 1) * 2)
    params = []
    for i in range(len(model_def) - 1):
        layer = {
            "weights": jax.random.normal(subkeys[i], (model_def[i], model_def[i + 1])),
            "bias": jax.random.normal(subkeys[i + len(model_def) - 1], (model_def[i + 1],)),
        }
        params.append(layer)
    return params


def model_forward(x, params):
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def node_forward(x0, t, params):
    T0 = t[0]
    T1 = t[-1]
    h = t[1] - t[0]
    saveat = diffrax.SaveAt(ts=t)
    term = diffrax.ODETerm(lambda t, x, args: model_forward(x, args))
    solver = diffrax.Dopri5()
    args = params
    solution = diffrax.diffeqsolve(term, solver, t0=T0, t1=T1, dt0=h, y0=x0, saveat=saveat, args=args)
    return solution.ys


def trapezoid(f, h):
    return h * (jnp.sum(f[1:-1]) + (f[0] + f[-1]) / 2.0)


def line_integral_loss(params, t, x, x_dot):
    h = t[1] - t[0]
    f = jax.vmap(model_forward, in_axes=(1, None), out_axes=1)(x[0:-1, :, :], params)
    f_norm = jnp.linalg.norm(f, ord=2, axis=2, keepdims=True)
    x_dot_norm = jnp.linalg.norm(x_dot, ord=2, axis=2, keepdims=True)
    ff = jnp.linalg.vecdot(f / f_norm, x_dot / x_dot_norm, axis=2)
    losses = jax.vmap(trapezoid, in_axes=(1, None))(ff, h)
    return jnp.mean(-losses) / (t[-1] - t[0])


def loss(x_pred, x_true):
    return jnp.mean((x_pred - x_true) ** 2)


def step(params, t, x):
    x0 = x[0, :, :]
    x_pred = jax.vmap(node_forward, in_axes=(0, None, None), out_axes=1)(x0, t, params)
    return loss(x_pred, x)


@jax.value_and_grad
def train_step(params, t, x, beta, x_dot):
    loss_value = step(params, t, x)
    loss_value += beta * line_integral_loss(params, t, x, x_dot)
    return loss_value


@partial(jax.jit, static_argnums=2)
def train(params, opt_state, optimizer, t, x, beta, x_dot):
    loss_value, loss_grads = train_step(params, t, x, beta, x_dot)
    updates, opt_state = optimizer.update(loss_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss_value, params, opt_state


@jax.jit
def validate(params, t, x):
    return step(params, t, x)


def main():
    seed = 42069
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    # function = mass_spring_damper
    # m = 1.0
    # d = 1.0
    # k = 1.0
    # args = m, d, k

    function = van_der_pol
    mu = 0.1
    args = mu

    T0 = 0.0
    T1 = 10.0
    h = 0.1
    t = jnp.arange(T0, T1, h)

    key, subkey = jax.random.split(key)
    data_points = 10
    x0 = jax.random.uniform(key, shape=(data_points, 2), minval=-10.0, maxval=10.0)

    x_data = generate_data(function, args, t, x0)

    train_size = int(data_points * 0.8)
    x_train = x_data[:, :train_size, :]
    x_val = x_data[:, train_size:, :]

    key, subkey = jax.random.split(key)
    sensor_noise = jax.random.normal(key, shape=x_train.shape) * 1e-3
    x_train += sensor_noise

    key, subkey = jax.random.split(key)
    model_def = [2, 50, 2]
    params = model_init(model_def, subkey)

    optimizer = optax.adamw(learning_rate=5e-3)
    opt_state = optimizer.init(params)

    epochs = 100
    train_losses = np.zeros((epochs,))
    val_losses = np.zeros((epochs,))

    batch_size = 5

    beta = 1.0
    # beta = 0.0

    # x_dot = (x_train[1:, :, :] - x_train[:-1, :, :]) / h
    # f = jax.vmap(function, in_axes=(None, 1, None), out_axes=1)(None, x_train[0:-1], args)
    # f_norm = jnp.linalg.norm(f, ord=2, axis=2, keepdims=True)
    # x_dot_norm = jnp.linalg.norm(x_dot, ord=2, axis=2, keepdims=True)
    # ff = jnp.linalg.vecdot(f / f_norm, x_dot / x_dot_norm, axis=2)
    # losses = jax.vmap(trapezoid, in_axes=(1, None))(ff, h)
    # print(jnp.mean(-losses) / (t[-1] - t[0]))
    # plot_vector_field(lambda x: van_der_pol(None, x, args))
    # exit()

    for epoch in range(epochs):
        try:
            key, subkey = jax.random.split(key)
            x_train = jax.random.permutation(subkey, x_train, axis=1)

            train_loss_mean = 0.0
            for i in range(train_size // batch_size):
                x_batch = x_train[:, i * (batch_size) : (i + 1) * batch_size, :]
                x_dot_batch = (x_batch[1:, :, :] - x_batch[:-1, :, :]) / h

                train_loss = validate(params, t, x_batch)
                train_loss_mean += train_loss

                train_loss, params, opt_state = train(params, opt_state, optimizer, t, x_batch, beta, x_dot_batch)

            train_loss_mean /= train_size // batch_size

            val_loss = validate(params, t, x_val)

            train_losses[epoch] = train_loss_mean
            val_losses[epoch] = val_loss

            print(f"Epoch: {epoch + 1:3d}, Train Loss: {train_loss_mean.item():.4f}, Val Loss: {val_loss.item():.4f}")
        except KeyboardInterrupt:
            break

    x_dot = (x_train[1:, :, :] - x_train[:-1, :, :]) / h
    print(line_integral_loss(params, t, x_train, x_dot))

    plot_vector_field(lambda x: model_forward(x, params))
    # plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
