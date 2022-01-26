from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import pytest
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import expm
from dynamics_learner.random_lindblad_utils import _basis, _random_hamiltonian, _random_gamma

key = random.PRNGKey(42)

@pytest.mark.parametrize("n", [10])
def test_basis(
    n,
):
    f = _basis(n)
    dot_prod_matrix = (f[:, jnp.newaxis].conj() * f[jnp.newaxis]).sum((-2, -1))
    assert jnp.linalg.norm(dot_prod_matrix - jnp.eye(n**2 - 1)) < 1e-10, "Basis elements are not orthogonal to each other!"
    assert jnp.linalg.norm(jnp.trace(f, axis1=1, axis2=2)) < 1e-10, "Basis elements are not traceless!"

@pytest.mark.parametrize("key,n", [(key, 10)])
def test_random_hamiltonian(
    key,
    n,
):
    h = _random_hamiltonian(key, n)
    assert jnp.linalg.norm(h - h.conj().T) < 1e-10, "The generated Hamiltonian is not a Hermitian matrix!"

@pytest.mark.parametrize("key,n", [(key, 10)])
def test_random_gamma(
    key,
    n,
):
    gamma = _random_gamma(key, n)
    assert jnp.linalg.norm(gamma - gamma.conj().T) < 1e-10, "The generated gamma is not a Hermitian matrix!"
    assert jnp.min(jnp.linalg.eigvalsh(gamma)) > 0, "The generated gamma in not positive!"
