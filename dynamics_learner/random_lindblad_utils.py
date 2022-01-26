import jax.numpy as jnp
from jax import random


def _basis(
    n: int,
) -> jnp.ndarray:
    """[This function returns a basis in the space of complex
    valued matrices of size n x n.]

    Args:
        n (int): [matrix dimension]

    Returns:
        real valued jnp.ndarray 0f shape (n**2 - 1, n, n): [basis,
            0th index enumerates basis elements]
    """

    q, _ = jnp.linalg.qr(jnp.eye(n).reshape((n ** 2, 1)), mode='complete')
    q = q[1:]
    return q.reshape((-1, n, n))


def _random_hamiltonian(
    key: jnp.ndarray,
    n: int,
) -> jnp.ndarray:
    """[This function generates a random Hamiltonian]

    Args:
        key (jnp.ndarray): [PRNGKey]
        n (int): [size of a Hamiltonian]

    Returns:
        complex valued jnp.ndarray of shape (n, n): [Hamiltonian]
    """
    
    h = random.normal(key, (n, n, 2))
    h = h[..., 0] + 1j * h[..., 1]
    h = (h + h.conj().T) / 2
    return h


def _random_gamma(
    key: jnp.ndarray,
    n: int,
) -> jnp.ndarray:
    """[This function generates a random matrix (gamma matrix) that
        defines the dissipative part of a Lindbladian.]

    Args:
        key (jnp.ndarray): [PRNGKey]
        n (int): [size of a gamma matrix]

    Returns:
        complex valued jnp.ndarray of shape (n, n): [gamma matrix]
    """

    subkey1, subkey2 = random.split(key)
    u = random.normal(subkey1, (n, n, 2))
    u = u[..., 0] + 1j * u[..., 1]
    u, _ = jnp.linalg.qr(u)
    lmbd = random.uniform(subkey2, (n,))
    return (u * lmbd) @ u.conj().T
