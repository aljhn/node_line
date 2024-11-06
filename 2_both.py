import random
from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns


def pendulum(t, x, args):
    l, g = args
    a = l / g
    x1 = x[:, 0]
    x2 = x[:, 1]
    dx1 = x2
    dx2 = a * jnp.sin(x1)
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
    sns.set_theme(style="darkgrid")
    plt.streamplot(X1, X2, Y1, Y2, density=1, linewidth=None, color="#A23BEC")
    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_mse_losses, val_ll_losses):
    epochs = train_losses.shape[0]
    train_epochs = np.arange(1, epochs + 1, 1)
    val_epochs = np.arange(1, epochs + 1, 10)
    plt.figure()
    sns.set_theme(style="darkgrid")
    plt.plot(train_epochs, train_losses)
    plt.plot(val_epochs, val_mse_losses)
    # plt.plot(val_epochs, val_ll_losses)
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Loss")
    # plt.legend(["Train", "Val MSE", "Val LL"])
    plt.legend(["Train", "Val MSE"])
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
    adjoint = diffrax.BacksolveAdjoint(solver=solver)
    args = params
    solution = diffrax.diffeqsolve(term, solver, t0=T0, t1=T1, dt0=h, y0=x0, saveat=saveat, adjoint=adjoint, args=args)
    return solution.ys


def trapezoid(f, h):
    return h * (jnp.sum(f[1:-1]) + (f[0] + f[-1]) / 2.0)


def line_integral_loss(params, t, x, x_dot):
    h = t[1] - t[0]
    f = jax.vmap(model_forward, in_axes=(1, None), out_axes=1)(x[0:-1, :, :], params)
    f_norm = jnp.linalg.norm(f, ord=2, axis=2, keepdims=True)
    x_dot_norm = jnp.linalg.norm(x_dot, ord=2, axis=2, keepdims=True)
    ff = jnp.linalg.vecdot(f / f_norm, x_dot / x_dot_norm, axis=2)
    # ff = jnp.linalg.vecdot(f, x_dot, axis=2)
    losses = jax.vmap(trapezoid, in_axes=(1, None))(ff, h)
    return jnp.mean(-losses) / (t[-1] - t[0])


def loss(x_true, x_pred):
    return jnp.mean((x_true - x_pred) ** 2)


def step(params, t, x):
    x0 = x[0, :, :]
    x_pred = jax.vmap(node_forward, in_axes=(0, None, None), out_axes=1)(x0, t, params)
    return loss(x, x_pred)


@jax.value_and_grad
def train_step(params, t, x, x_dot, beta):
    loss_value = step(params, t, x)
    loss_value += beta * line_integral_loss(params, t, x, x_dot)
    return loss_value


@partial(jax.jit, static_argnums=2)
def train(params, opt_state, optimizer, t, x, x_dot, beta):
    loss_value, loss_grads = train_step(params, t, x, x_dot, beta)
    updates, opt_state = optimizer.update(loss_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss_value, params, opt_state


@jax.jit
def validate(params, t, x, x_dot):
    mse_loss = step(params, t, x)
    ll_loss = line_integral_loss(params, t, x, x_dot)
    return mse_loss, ll_loss


def main():
    seed = 42069
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    function = pendulum
    l = 10.0
    g = 1.0
    args = l, g

    T0 = 0.0
    T1 = 1.0
    h = 0.01
    t = jnp.arange(T0, T1, h)

    key, subkey = jax.random.split(key)
    train_trajectories = 10
    val_trajectories = 100
    data_trajectories = train_trajectories + val_trajectories
    x_range = 10.0
    x0 = jax.random.uniform(key, shape=(data_trajectories, 2), minval=-x_range, maxval=x_range)

    x_data = generate_data(function, args, t, x0)

    x_train = x_data[:, :train_trajectories, :]
    x_val = x_data[:, train_trajectories:, :]

    x_dot_val = (x_val[1:, :, :] - x_val[:-1, :, :]) / h

    key, subkey = jax.random.split(key)
    sensor_noise = jax.random.normal(subkey, shape=x_train.shape) * 1e-3
    x_train += sensor_noise

    batch_size = 5
    train_iterations = train_trajectories // batch_size

    key, subkey = jax.random.split(key)
    model_def = [2, 20, 20, 20, 2]
    params0 = model_init(model_def, subkey)

    key, subkey = jax.random.split(key)
    params1 = model_init(model_def, subkey)

    initial_learning_rate = 1e-3
    # learning_rate = optax.schedules.exponential_decay(initial_learning_rate, 10, 0.99)
    learning_rate = optax.schedules.constant_schedule(initial_learning_rate)
    optimizer0 = optax.adamw(learning_rate=learning_rate)
    opt_state0 = optimizer0.init(params0)

    optimizer1 = optax.adamw(learning_rate=learning_rate)
    opt_state1 = optimizer1.init(params1)

    epochs = 1000
    train_losses = np.zeros((epochs, 2))

    validation_frequency = 10
    val_mse_losses = np.zeros((epochs // validation_frequency, 2))
    val_ll_losses = np.zeros((epochs // validation_frequency, 2))

    beta0 = 0.0
    beta1 = 1e2

    for epoch in range(epochs):
        try:
            key, subkey = jax.random.split(key)
            x_train = jax.random.permutation(subkey, x_train, axis=1)

            train_loss_mean0 = 0.0
            train_loss_mean1 = 0.0
            for i in range(train_iterations):
                x_batch = x_train[:, i * (batch_size) : (i + 1) * batch_size, :]
                x_dot_batch = (x_batch[1:, :, :] - x_batch[:-1, :, :]) / h

                train_loss0, params0, opt_state0 = train(params0, opt_state0, optimizer0, t, x_batch, x_dot_batch, beta0)
                train_loss_mean0 += train_loss0.item()

                train_loss1, params1, opt_state1 = train(params1, opt_state1, optimizer1, t, x_batch, x_dot_batch, beta1)
                train_loss_mean1 += train_loss1.item()

            train_loss_mean0 /= train_iterations
            train_loss_mean1 /= train_iterations

            train_losses[epoch, 0] = train_loss_mean0
            train_losses[epoch, 1] = train_loss_mean1

            if (epoch + 1) % validation_frequency == 0:
                val_mse_loss0, val_ll_loss0 = validate(params0, t, x_val, x_dot_val)
                val_mse_losses[epoch // validation_frequency, 0] = val_mse_loss0
                val_ll_losses[epoch // validation_frequency, 0] = val_ll_loss0

                val_mse_loss1, val_ll_loss1 = validate(params1, t, x_val, x_dot_val)
                val_mse_losses[epoch // validation_frequency, 1] = val_mse_loss1
                val_ll_losses[epoch // validation_frequency, 1] = val_ll_loss1

                print(f"Epoch: {epoch + 1:3d}")
                print(f"Model: 0, Train Loss: {train_loss_mean0:.4f}, Val MSE Loss: {val_mse_loss0.item():.4f}, Val LL Loss: {val_ll_loss0.item():.4f}")
                print(f"Model: 1, Train Loss: {train_loss_mean1:.4f}, Val MSE Loss: {val_mse_loss1.item():.4f}, Val LL Loss: {val_ll_loss1.item():.4f}")
                print()

            if (epoch + 1) % 100 == 0:
                beta1 *= 0.5

        except KeyboardInterrupt:
            break

    X = jnp.arange(-10, 10, 0.1)
    X1, X2 = jnp.meshgrid(X, X, indexing="xy")
    XX = jnp.stack((X1.flatten(), X2.flatten()), axis=1)

    YY_true = pendulum(None, XX, args)
    YY_pred0 = model_forward(XX, params0)
    YY_pred1 = model_forward(XX, params1)

    vector_field_simularity0 = loss(YY_true, YY_pred0)
    vector_field_simularity1 = loss(YY_true, YY_pred1)
    print()
    print(f"Vector field similarity 0: {vector_field_simularity0:.4f}")
    print(f"Vector field similarity 1: {vector_field_simularity1:.4f}")

    # plot_vector_field(lambda x: pendulum(None, x, args))
    plot_vector_field(lambda x: model_forward(x, params0))
    plot_vector_field(lambda x: model_forward(x, params1))
    # plot_losses(train_losses, val_mse_losses, val_ll_losses)


if __name__ == "__main__":
    main()
