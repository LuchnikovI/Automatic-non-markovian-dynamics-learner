import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.maps import xmap
from typing import Iterable, Callable, Any
from functools import partial

from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian
from dynamics_learner.general_utils import rho2bloch


# this dict shows how nested data points are organized
axes_order = {
    'de': 0,
    'training_set_size': 1,
    'discrete_time_steps': 2,
    'K': 3,
    'key': 4,
    'tau': 5,
    'hamiltonian_amplitude': 6,
    'dissipative_amplitude': 7,
    'sigma': 8,
    }


def trace_distance(
    rho1: jnp.ndarray,
    rho2: jnp.ndarray,
) -> jnp.ndarray:
    """[This function calculates the trace distance between sets of density matrices]

    Args:
        rho1 (complex vlaued jnp.ndarray of shape (..., n, n)): [first set of density matrices]
        rho2 (complex valued jnp.ndarray of shape (..., n, n)): [second set of density matrices]

    Returns:
        complex valued jnp.ndarray of shape (...,): [distances between corresponding density matrices]
    """

    return 0.5 * jnp.linalg.norm(rho2bloch(rho1) - rho2bloch(rho2), axis=-1)


def logrange(min: float, max: float, n: int) -> jnp.ndarray:
    """The equivalent of jnp.arange, but in the logarithmic scale."""
    return jnp.exp(jnp.linspace(jnp.log(min), jnp.log(max), n))


def log_mean(x: jnp.ndarray, but_axis: str) -> jnp.ndarray:
    """This function calculates the logarithmic mean of a bunch of values."""
    indices = tuple(value for key, value in axes_order.items() if key != but_axis)
    return jnp.exp(jnp.log(x).mean(indices))



dynamic_in_axes = JITDynamicParamsRandomLindbladian(
    key = {0: 'key'},
    tau = {0: 'tau'},
    hamiltonian_amplitude = {0: 'hamiltonian_amplitude'},
    dissipative_amplitude = {0: 'dissipative_amplitude'},
    sigma = {0: 'sigma'},
    )


dynamic_out_axes = ['key', 'tau', 'hamiltonian_amplitude', 'dissipative_amplitude', 'sigma', ...]


@partial(xmap, in_axes=(dynamic_in_axes,), out_axes=dynamic_out_axes)
def axes_mesh(dynamic_params: JITDynamicParamsRandomLindbladian):
    return dynamic_params


def experiments_vectorizer(
    func: Callable,
) -> Any:
    """This decorator transforms function that performs an experiment in
    such a way, that the new function performs an experiment for each
    combination of dynamic parameters."""

    func = vmap(vmap(vmap(vmap(vmap(func,
           in_axes=(0, None)), in_axes=(0, None)),
           in_axes=(0, None)), in_axes=(0, None)),
           in_axes=(0, None))
    def vectorized_func(dynamic_params, static_params):
        return func(axes_mesh(dynamic_params), static_params)
    vectorized_func = jit(vectorized_func, static_argnums=1)
    return vectorized_func
