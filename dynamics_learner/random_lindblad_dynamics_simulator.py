import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import random, jit, lax, vmap
from functools import partial
from typing import Union
from math import sqrt
from dynamics_learner.random_lindblad_utils import _basis, _random_hamiltonian, _random_gamma


def random_lindbladian(
    key: jnp.ndarray,
    hamiltonian_amplitude: Union[jnp.ndarray, float],
    dissipative_amplitude: Union[jnp.ndarray, float],
    dsde: int,
) -> jnp.ndarray:
    """[This function generates a random Lindbladian.]

    Args:
        key (jnp.ndarray): [PRNGKey]
        hamiltonian_amplitude (Union[jnp.ndarray, float]): [amplitude of the Hamiltonian part]
        dissipative_amplitude (Union[jnp.ndarray, float]): [amplitude of the dissipative part]
        dsde (int): [the dimension of the system and environment Hilbert space]

    Returns:
        jnp.ndarray: [random Lindbladian]
    """

    subkey1, subkey2 = random.split(key)
    h = _random_hamiltonian(subkey1, dsde)
    gamma = _random_gamma(subkey2, dsde ** 2 - 1)
    i = jnp.eye(dsde)
    f = _basis(dsde)

    commutator = jnp.kron(h, i) - jnp.kron(i, h.T)
    ff = jnp.einsum('nm,nki,mkj->ij', gamma, f.conj(), f)

    anticommutator = jnp.kron(ff, i) + jnp.kron(i, ff.T)
    f_f = jnp.einsum('mn,nij,mkl->ikjl', gamma, f, f.conj()).reshape((dsde**2, dsde**2))

    return -1j * hamiltonian_amplitude * commutator + dissipative_amplitude * (f_f - 0.5 * anticommutator)


def environment_steady_state(
    lindbladian: jnp.ndarray,
    de: int,
) -> jnp.ndarray:
    """[This function computes the partial density matrix of the environment
        at the steady state of the given Lindbladian.]

    Args:
        lindbladian (complex valued jnp.ndarray of shape (dsde**2, dsde**2)): [Lindbladian]
        de (int): dimension of the environment

    Returns:
        complex valued jnp.ndarray of shape (de, de): [partial density matrix of the environemnt]
    """

    dsde = lindbladian.shape[0]
    ds = int(sqrt(dsde)) // de
    _, _, vh = jnp.linalg.svd(lindbladian, full_matrices=False)
    rho = vh[-1].conj()
    rho = rho.reshape((ds, de, ds, de))
    rho = jnp.trace(rho, axis1=0, axis2=2)
    rho /= jnp.trace(rho)
    return rho


@partial(vmap, in_axes=(None, None, 0, None, None))
def run_dynamics(
    lindbladian: jnp.ndarray,
    steady_environment_state: jnp.ndarray,
    system_initial_state: jnp.ndarray,
    tau: Union[jnp.ndarray, float],
    total_discrete_time: int,
) -> jnp.ndarray:
    """[This function runs dynamics of the system with a random Lindbladian.]

    Args:
        lindbladian (complex valued jnp.ndarray of shape (dsde**2, dsde**2)): [Lindbladian]
        complex valued jnp.ndarray of shape (de, de): [partial density matrix of the environemnt]
        system_initial_state (complex valued jnp.ndarray of shape (ds, ds)): [initial system state]
        tau (Union[jnp.ndarray, float]): [time step]
        total_discrete_time (int): [total number of time steps]

    Returns:
        complex valued jnp.ndarray of shape (total_discrete_time, ds, ds): [resulting dynamics of the system's density matrix]
    """

    ds = system_initial_state.shape[-1]
    de = steady_environment_state.shape[-1]
    phi = expm(tau * lindbladian)
    in_state = steady_environment_state[jnp.newaxis, :, jnp.newaxis] * system_initial_state[:, jnp.newaxis, :, jnp.newaxis]
    in_system_state = jnp.trace(in_state, axis1=1, axis2=3)
    in_state = in_state.reshape((-1,))
    def iter(carry, xs):
        # carry is the currents sys + env state
        # y is the current sys state
        carry = phi.dot(carry)
        y = carry.reshape((ds, de, ds, de))
        y = jnp.trace(y, axis1=1, axis2=3)
        return carry, y
    _, dynamics = lax.scan(iter, in_state, None, length=total_discrete_time-1)
    return jnp.concatenate([in_system_state[jnp.newaxis], dynamics], axis=0)
