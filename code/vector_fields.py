from abc import ABC, abstractmethod
from typing import Callable, Sequence

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from plotting import plot_vector_field


class VectorField(ABC):

    def __init__(self, args: Sequence[Float]):
        self.args = args
        self.function = jax.jit(
            jax.vmap(self._get_function(), in_axes=(None, 0, None), out_axes=0)
        )

    @abstractmethod
    def _get_function(self) -> Callable[[Float, Array, Sequence[Float]], Array]:
        pass

    def generate(
        self, x0: Float[Array, "batch dim"], t: Float[Array, "time"], dt0: Float
    ) -> Float[PyTree, "time batch dim"]:
        T0 = t[0]
        T1 = t[-1]

        term = diffrax.ODETerm(self.function)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=t)
        stepsize_controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)
        adjoint = diffrax.RecursiveCheckpointAdjoint()

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=T0,
            t1=T1,
            dt0=dt0,
            y0=x0,
            args=self.args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
        )
        return solution.ys


class MassSpringDamper(VectorField):

    def __init__(self, m: Float, d: Float, k: Float):
        super().__init__((m, d, k))

    def _get_function(self) -> Callable[[Float, Array, Sequence[Float]], Array]:
        def function(t, x, args):
            m, d, k = args
            x1, x2 = x
            dx1 = x2
            dx2 = -k / m * x1 - d / m * x2
            return jnp.array((dx1, dx2))

        return function


class SinglePendulum(VectorField):

    def __init__(self, l: Float, g: Float):
        super().__init__((l, g))

    def _get_function(self) -> Callable[[Float, Array, Sequence[Float]], Array]:
        def function(t, x, args):
            l, g = args
            x1, x2 = x
            dx1 = x2
            dx2 = -g / l * jnp.sin(x1)
            return jnp.array((dx1, dx2))

        return function


class DoublePendulum(VectorField):

    def __init__(self, m1: Float, m2: Float, l1: Float, l2: Float, g: Float):
        super().__init__((m1, m2, l1, l2, g))

    def _get_function(self) -> Callable[[Float, Array, Sequence[Float]], Array]:
        def function(t, x, args):
            m1, m2, l1, l2, g = args

            theta_1 = x[0]
            theta_2 = x[1]
            theta_dot_1 = x[2]
            theta_dot_2 = x[3]

            M_1_1 = (m1 + m2) * (l1**2)
            M_1_2 = m2 * l1 * l2 * jnp.cos(theta_1 - theta_2)
            M_2_1 = m2 * l1 * l2 * jnp.cos(theta_1 - theta_2)
            M_2_2 = m2 * (l2**2)

            M_det = M_1_1 * M_2_2 - M_1_2 * M_2_1

            M_inv_1_1 = M_2_2 / M_det
            M_inv_1_2 = -M_1_2 / M_det
            M_inv_2_1 = -M_2_1 / M_det
            M_inv_2_2 = M_1_1 / M_det

            ff_1 = -m2 * l1 * l2 * jnp.sin(theta_1 - theta_2) * (theta_dot_2**2) - (
                m1 + m2
            ) * g * l1 * jnp.sin(theta_1)
            ff_2 = m2 * l1 * l2 * jnp.sin(theta_1 - theta_2) * (
                theta_dot_1**2
            ) - m2 * g * l2 * jnp.sin(theta_2)

            theta_ddot_1 = M_inv_1_1 * ff_1 + M_inv_1_2 * ff_2
            theta_ddot_2 = M_inv_2_1 * ff_1 + M_inv_2_2 * ff_2

            return jnp.array((theta_dot_1, theta_dot_2, theta_ddot_1, theta_ddot_2))

        return function


if __name__ == "__main__":
    m = 1.0
    d = 1.0
    k = 1.0
    vector_field = MassSpringDamper(m, d, k)

    seed = 42069
    key = jax.random.PRNGKey(seed)

    batch_size = 10
    dim = 2
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, (batch_size, dim))
    t = jnp.arange(0.0, 4.0, 1.0)
    dt0 = 0.01
    x = vector_field.generate(x0, t, dt0)
    print(x.shape)

    plot_vector_field(lambda x: vector_field.function(None, x, vector_field.args))
