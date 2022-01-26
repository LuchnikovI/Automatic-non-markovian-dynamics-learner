from base64 import decode
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import pytest
import jax.numpy as jnp
from jax import random

from dynamics_learner.data_processing_utils import _hankelizer, _dehankelizer, _trunc_svd, _exact_dmd, _ridge_regression

key = random.PRNGKey(42)


def test_hankelizer_simple():
    tensor = jnp.arange(10).reshape((1, -1, 1))
    true_hankel = jnp.array([[0, 1, 2, 3, 4],
                             [1, 2, 3, 4, 5],
                             [2, 3, 4, 5, 6],
                             [3, 4, 5, 6, 7],
                             [4, 5, 6, 7, 8],
                             [5, 6, 7, 8, 9]])
    true_hankel = true_hankel.reshape((1, 6, 5, 1))
    assert jnp.linalg.norm(true_hankel - _hankelizer(tensor, 5)) < 1e-10, "Hankel matrix is incorrect!"


@pytest.mark.parametrize("key,batch_size,K,n,m", [(key, 3, 7, 15, 6)])
def test_hankelizer_dehankelizer(
    key,
    batch_size,
    K,
    n,
    m,
):
    tensor = random.normal(key, (batch_size, n, m, 2))
    tensor = tensor[..., 0] + 1j * tensor[..., 1]
    hankel = _hankelizer(tensor, K)
    for i in range(n - K + 1):
        assert jnp.linalg.norm(hankel[:, i] - tensor[:, i:K+i]) < 1e-10, "Slice #{} of the Hanke tensor is incorrect!".format(i)
    assert jnp.linalg.norm(tensor -_dehankelizer(hankel)) < 1e-10, "Dehankelizer does not work correctly!"


@pytest.mark.parametrize("key,sigma,rank,n,m", [(key, 0.01, 10, 20, 30)])
def test_trunc_svd(
    key,
    sigma,
    rank,
    n,
    m,
):
    subkey1, subkey2, subkey3 = random.split(key, 3)
    a = random.normal(subkey1, (n, rank, 2))
    a = a[..., 0] + 1j * a[..., 1]
    b = random.normal(subkey2, (rank, m, 2))
    b = b[..., 0] + 1j * b[..., 1]
    data_matrix = a @ b
    noise = random.normal(subkey3, (n, m))
    noisy_data_matrix = data_matrix + sigma * noise
    u, s, vh = _trunc_svd(noisy_data_matrix, sigma)
    assert s.shape[0] == rank, "Rank is found incorrectly!"
    err1 = jnp.linalg.norm(noise)
    err2 = jnp.linalg.norm((u * s) @ vh - data_matrix)
    assert err1 > err2, "Denoising does not work!"


@pytest.mark.parametrize("key,d_sq,K,rank,n,batch_size", [(key, 3, 6, 12, 10, 10)])
def test_exact_dmd(
    key,
    d_sq,
    K,
    rank,
    n,
    batch_size,
):
    subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)
    # random low-rank transition matrix
    a = random.normal(subkey1, (K * d_sq, rank, 2))
    a = a[..., 0] + 1j * a[..., 1]
    b = random.normal(subkey2, (rank, K * d_sq, 2))
    b = a[..., 0] + 1j * b[..., 1]
    transition_matrix = a @ b
    # sorted spectrum of the generated matrix
    spec = jnp.linalg.eigvals(transition_matrix)
    order = jnp.argsort(jnp.abs(spec))
    spec = spec[order]
    spec = spec[-rank:]
    # random data and test sets in the column space of the generated matrix
    u, _, _ = jnp.linalg.svd(transition_matrix, full_matrices=False)
    u = u[:, :rank]
    x = random.normal(subkey3, (rank, n * batch_size, 2))
    x = x[..., 0] + 1j * x[..., 1]
    x = u @ x
    x_test = random.normal(subkey4, (rank, n * batch_size, 2))
    x_test = x_test[..., 0] + 1j * x_test[..., 1]
    x_test = u @ x_test
    y = transition_matrix @ x
    y_test = transition_matrix @ x_test
    x = x.reshape((K, d_sq, batch_size, n)).transpose((2, 3, 0, 1))
    y = y.reshape((K, d_sq, batch_size, n)).transpose((2, 3, 0, 1))
    # data fitting by dmd
    dynamics_generator, _ = _exact_dmd(x, y, 1e-10, K)
    decoder, eigvals, encoder = dynamics_generator.decoder, dynamics_generator.eigvals, dynamics_generator.encoder
    # sorting eigen vectors and values
    order = jnp.argsort(jnp.abs(eigvals))
    decoder = decoder[:, order]
    encoder = encoder[order]
    eigvals = eigvals[order]
    # asserts
    assert eigvals.shape[0] == rank, "Rank is incorrect!"
    assert jnp.linalg.norm(spec - eigvals) < 1e-10, "Reconstructed eigenvalues are incorrect!"
    assert jnp.linalg.norm(encoder @ decoder - jnp.eye(rank)) < 1e-10, "Reconstructed left and right eigenvectors are not mutually orthonormal!"
    assert jnp.linalg.norm(y_test - (decoder * eigvals) @ (encoder @ x_test)) < 1e-10, "The reconstructed transition matrix is incorrect!"


@pytest.mark.parametrize("key,d_sq,K,n,batch_size", [(key, 3, 6, 10, 10)])
def test_ridge_regression(
    key,
    d_sq,
    K,
    n,
    batch_size,
):
    subkey1, subkey2 = random.split(key)
    transition_matrix = random.normal(subkey1, (K * d_sq, K * d_sq, 2))
    transition_matrix = transition_matrix[..., 0] + 1j * transition_matrix[..., 1]
    x = random.normal(subkey2, (K * d_sq, n * batch_size, 2))
    x = x[..., 0] + 1j * x[..., 1]
    y = transition_matrix @ x
    x = x.reshape((K, d_sq, batch_size, n)).transpose((2, 3, 0, 1))
    y = y.reshape((K, d_sq, batch_size, n)).transpose((2, 3, 0, 1))
    reconstructed_transition_matrix = _ridge_regression(x, y, 0, K)
    assert jnp.linalg.norm(reconstructed_transition_matrix - transition_matrix) < 1e-10, "Ridge regression does not work correctly!"
