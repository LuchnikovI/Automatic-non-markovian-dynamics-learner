import jax.numpy as jnp
from jax import random
from math import sqrt
from typing import Union

sigma = jnp.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def random_pure_density_matrices(
    key: jnp.ndarray,
    dim: int,
    number: int,
) -> jnp.ndarray:
    """[This function generates a random set of pure density matrices.]

    Args:
        key (jnp.ndarray): [PRNGKey]
        dim (int): [Hilbert space dimension]
        number (int): [number of density matrices]

    Returns:
        complex valued jnp.ndarray of shape (number, dim, dim): [set of pure density matrices]
    """

    psi = random.normal(key, (number, dim, 2))
    psi = psi[..., 0] + 1j * psi[..., 1]
    psi /= jnp.linalg.norm(psi, axis=1, keepdims=True)
    return psi[..., jnp.newaxis] * psi[:, jnp.newaxis].conj()


def rho2bloch(
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """[This function transforms set of density matrices to set of Bloch vectors.]

    Args:
        rho (complex valued jnp.ndarray of shape (..., 2, 2)): [set of density matrices]

    Returns:
        real valued jnp.ndarray of shape (..., 3): [set of Bloch vectors]
    """

    return jnp.tensordot(rho, sigma, axes=[[-1, -2], [1, 2]]).real


def bloch2rho(
    bloch: jnp.ndarray,
) -> jnp.ndarray:
    """[This function transforms set of Bloch vectors to set of density matrices.]

    Args:
        bloch (real valued jnp.ndarray of shape (..., 3)): [set of Bloch vectors]

    Returns:
        complex valued jnp.ndarray of shape (..., 2, 2): [set of density matrices]
    """

    return (jnp.tensordot(bloch, sigma, axes=[[-1], [0]]) + jnp.eye(2)) / 2


# TODO: add tests to this function
def flatten_rho(
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """[This function flattens rho into vector]

    Args:
        rho (complex valued jnp.ndarray of shape (..., n, n)): [set of density matrices]

    Returns:
        complex valued jnp.ndarray of shape (..., n ** 2): [set of vectorized density matrices]
    """

    shape = rho.shape
    return rho.reshape((*shape[:-2], shape[-1] ** 2))


# TODO: add tests to this function
def unflatten_rho(
    vec_rho: jnp.ndarray,
) -> jnp.ndarray:
    """[This function unflattens vectorized rho back into the matrix form.]

    Args:
        vec_rho (complex valued jnp.ndarray of shape (..., n ** 2)): [set of vectorized density matrices]

    Returns:
        complex valued jnp.ndarray of shape (..., n, n): [set of density matrices]
    """

    shape = vec_rho.shape
    n = int(sqrt(shape[-1]))
    return vec_rho.reshape((*shape[:-1], n, n))


#TODO: add tests to this function
def addnoise(
    key: jnp.ndarray,
    rho: jnp.ndarray,
    sigma: Union[jnp.ndarray, float],
):
    """[This function adds complex valued normal i.i.d noise to the set of density matrices]

    Args:
        key (jnp.ndarray): [PRNGKey]
        rho (complex valued jnp.ndarray of shape (..., n, n)): [set of density matrices]
        sigma (Union[jnp.ndarray, float]): [standard deviation of the noise]
    """

    shape = rho.shape
    noise = random.normal(key, (*shape, 2))
    noise = noise[..., 0] + 1j * noise[..., 1]
    noise *= sigma
    return rho + noise
