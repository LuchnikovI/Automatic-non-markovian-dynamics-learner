from base64 import decode
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import pytest
import jax.numpy as jnp
from jax import random

from dynamics_learner.general_utils import random_pure_density_matrices, bloch2rho, rho2bloch

key = random.PRNGKey(42)

@pytest.mark.parametrize("key,dim,number", [(key, 3, 3)])
def test_random_pure_density_matrices(
    key,
    dim,
    number,
):
    rho = random_pure_density_matrices(key, dim, number)
    assert jnp.linalg.norm(rho - rho.conj().transpose((0, 2, 1))) < 1e-10, "Generated density matrices are not Hermitian!"
    eigvals = jnp.linalg.eigvalsh(rho)
    eigvals = jnp.sort(eigvals, axis=1)
    print(eigvals)
    assert jnp.linalg.norm(eigvals[:, -1] - 1) < 1e-10, "Leading eigenvalue of generated density matrices is not 1!"
    assert jnp.linalg.norm(eigvals[:, :-1]) < 1e-10, "The rest of eigenvalues of generated density matrices are not 1!"

@pytest.mark.parametrize("key,number", [(key, 10)])
def test_rho2bloch_and_bloch2rho(
    key,
    number,
):
    rho = random_pure_density_matrices(key, 2, number)
    bloch = rho2bloch(rho)
    rho_reconstructed = bloch2rho(bloch)
    assert jnp.linalg.norm(rho - rho_reconstructed) < 1e-10, "bloch2rho or/and rho2bloch work incorrectly!"
