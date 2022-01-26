import jax.numpy as jnp
from jax.lax import gather, GatherDimensionNumbers
from jax import vmap
from functools import partial
from typing import Tuple, Union
from dynamics_learner.dataclasses import DMDDynamicsGenerator


@partial(vmap, in_axes=(0, None), out_axes=0)
@partial(vmap, in_axes=(1, None), out_axes=2)
def _1dhankel(
    tensor: jnp.ndarray,
    K: int,
) -> jnp.ndarray:
    start_indices = jnp.arange(tensor.shape[0] - K + 1)[:, jnp.newaxis]
    hankel_tensor = gather(
        tensor,
        start_indices,
        GatherDimensionNumbers((0,), (), (0,)),
        (K,),
    )
    return hankel_tensor.T


def _hankelizer(
    tensor: jnp.ndarray,
    K: int,
) -> jnp.ndarray:
    """[This function transforms an ordinary tensor to the corresponding Hankel tensor.]

    Args:
        tensor (complex valued jnp.ndarray of shape (batch_size, n, m)): [ordinary tensor]
        K (int): [size of a chunk]

    Returns:
        complex valued jnp.ndarray of shape (batch_size, n-K, K, m): [Hankel tensor]
    """

    return _1dhankel(tensor, K)


def _dehankelizer(
    hankel: jnp.ndarray,
) -> jnp.ndarray:
    """[This function is the inverse of _hankelizer function.]

    Args:
        hankel (complex valued jnp.ndarray of shape (batch_size, n-K, K, m)): [Hankel tensor]

    Returns:
        complex valued jnp.ndarray of shape (batch_size, n, m): [ordinary tensor]
    """

    return jnp.concatenate([hankel[:, :, 0], hankel[:, -1, 1:]], axis=1)


# TODO: add jit support
def _trunc_svd(
    matrix: jnp.ndarray,
    sigma: Union[jnp.ndarray, float],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """[Perform a truncated svd decomposition of a matrix with the asymptotically
    optimal hard threshold according to the paper:
    "Gavish M, Donoho DL. The optimal hard threshold for singular values is $4/\sqrt {3} $.
    IEEE Transactions on Information Theory. 2014 Jun 30;60(8):5040-53.".]

    Args:
        matrix (complex valued jnp.ndarray of shape (n, m)): [matrix to be decomposed]
        sigma (Union[jnp.ndarray, float]): [std of complex valued i.i.d. normal noise]

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: [truncated u, s and vh factors of the svd decomposition.]
    """

    u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
    n, m = matrix.shape
    beta = m / n
    lmbd = jnp.sqrt(2 * (beta + 1) + (8 * beta) / (beta + 1 + jnp.sqrt(beta ** 2 + 14 * beta + 1)))
    threshold = lmbd * jnp.sqrt(2 * n) * sigma
    rank = (s > threshold).sum()
    return u[:, :rank], s[:rank], vh[:rank]


def _hankel2xy(
    hankelized_trajectories: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """[summary]

    Args:
        hankelized_trajectories (complex valued jnp.ndarray of shape (batch_size, n-K, K, d^2)):
            [input set of hankelized trajectories]

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [two complex valued jnp.ndarray x and y of shape 
            (batch_size, n-K-1, K, d^2)), where y is one time step forward wrt x]
    """

    return hankelized_trajectories[:, :-1], hankelized_trajectories[:, 1:]


def _exact_dmd(
    x: jnp.ndarray,
    y: jnp.ndarray,
    sigma: Union[jnp.ndarray, float],
    K: int,
    jit_compatible=False,
) -> Tuple[DMDDynamicsGenerator, jnp.ndarray]:
    """[Applies exact DMD algorithm to reconstruct transition matrix between x and y in
    a low-rank factorized form.]

    Args:
        x (complex valued jnp.ndarray of shape (batch_size, n-K-1, K, d^2)): [set of hankelized trajectories]
        y (complex valued jnp.ndarray of shape (batch_size, n-K-1, K, d^2)): [set of hankelized trajectories,
            that one time step forward wrt x]
        sigma (Union[jnp.ndarray, float]): [std of complex valued i.i.d. normal noise]
        K (int): [guessed memory depth]

    Returns:
        Tuple[DMDDynamicsGenerator, jnp.ndarray]: [DMD dynamics generator and denoised x]
    """

    # preprocessing and truncation
    batch_size, _, _, d_sq = x.shape
    x = x.reshape((-1, K*d_sq)).T
    y = y.reshape((-1, K*d_sq)).T
    u, s, vh = _trunc_svd(x, sigma)
    # denoising
    x_denoised = (u * s) @ vh
    x_denoised = x_denoised.T.reshape((batch_size, -1, K, d_sq))
    # exacr dmd
    s_inv = 1 / s
    a_tild = (u.conj().T @ y) @ (vh.conj().T * s_inv)
    lmbd, right = jnp.linalg.eig(a_tild)
    left_adj = jnp.linalg.inv(right)
    right = (y @ (vh.conj().T * s_inv)) @ right / lmbd
    left_adj = left_adj @ (u.conj().T)
    return DMDDynamicsGenerator(decoder=right, eigvals=lmbd, encoder=left_adj, rank=s.shape[0]), x_denoised


# TODO: consider reduced Y as an input in order to improve computational efficiency
def _ridge_regression(
    x: jnp.ndarray,
    y: jnp.ndarray,
    lmbd: Union[jnp.ndarray, float],
    K: int,
) -> jnp.ndarray:
    """[Applies ridge regression to reconstruct transition matrix between y anx x.]

    Args:
        x (complex valued jnp.ndarray of shape (batch_size, n-K-1, K, d^2)): [set of hankelized trajectories]
        y (complex valued jnp.ndarray of shape (batch_size, n-K-1, K, d^2)): [set of hankelized trajectories,
            that one time step forward wrt x]
        lmbd (Union[jnp.ndarray, float]): [regularization coefficient]
        K (int): [guessed memory depth]

    Returns:
        complex valued jnp.ndarray of shape (K*d^2, K*d^2): [transition matrix]
    """

    batch_size, _, _, d_sq = x.shape
    x = x.reshape((-1, K * d_sq))
    y = y.reshape((-1, K * d_sq))
    u, s, vh = jnp.linalg.svd(x, full_matrices=False)
    s_inv = s / (s ** 2 + lmbd)
    return (y.T @ (u.conj() * s_inv)) @ vh.conj()
