import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_vector_field(function, filename, width=10.0, step=0.1):
    """Only works for 2-dimensional vector fields"""
    X = jnp.arange(-width, width, step)
    X1, X2 = jnp.meshgrid(X, X, indexing="xy")
    XX = jnp.stack((X1.flatten(), X2.flatten()), axis=1)

    YY = function(XX)

    Y1 = YY[:, 0].reshape(X1.shape)
    Y2 = YY[:, 1].reshape(X2.shape)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 14
    # plt.rcParams["figure.figsize"] = (9, 9)
    plt.rcParams["figure.dpi"] = 600
    sns.set_theme(style="darkgrid")

    plt.figure()
    plt.streamplot(X1, X2, Y1, Y2, density=1, linewidth=1, color="#A23BEC")
    plt.xlim(-width, width)
    plt.ylim(-width, width)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.pdf")
    # plt.show()


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
