import jax.numpy as jnp
from jax import jit
from typing import Any, Tuple
from functools import partial

from dynamics_learner.data_processing_utils import _exact_dmd, _hankelizer, _hankel2xy, _dehankelizer, _ridge_regression
from dynamics_learner.dataclasses import DMDDynamicsGenerator


def dmd_fit(
    noisy_training_set: jnp.ndarray,
    dynamic_params: Any,
    static_params: Any,
) -> Tuple[DMDDynamicsGenerator, jnp.ndarray]:
    """[This function fits data set using DMD based model with automatic rank selection.]

    Args:
        noisy_training_set (complex valued jnp.ndarray of shape
            (batch_size, number_of_discrete_time_steps, n, n)): [set of quantum trajectories]
        dynamic_params (Any): [parameters that are not compile time constants]
        static_params (Any): [parameters that are compie time constants]

    Returns:
        Tuple[DMDDynamicsGenerator, jnp.ndarray]: [DMD based system model and set of denoised trajectories 
            (without last in time datapoint)]
    """

    shape = noisy_training_set.shape
    noisy_training_set = noisy_training_set.reshape((*shape[:-2], -1))
    noisy_training_set = _hankelizer(noisy_training_set, static_params.K)
    x, y = _hankel2xy(noisy_training_set)
    dynamics_generator, denoised_trajectories = _exact_dmd(x, y, dynamic_params.sigma, static_params.K)
    denoised_trajectories = _dehankelizer(denoised_trajectories)
    denoised_trajectories = denoised_trajectories.reshape((shape[0], shape[1]-1, *shape[2:]))
    return dynamics_generator, denoised_trajectories


def tt_fit(
    noisy_training_set: jnp.ndarray,
    dynamic_params: Any,
    static_params: Any,
) -> jnp.ndarray:
    """[This function fits data using ridge regression, that is essentiallt a Transfer-Tensor (TT)
        based fit.]

    Args:
        noisy_training_set (complex valued jnp.ndarray of shape
            (batch_size, number_of_discrete_time_steps, n, n)): [set of quantum trajectories]
        dynamic_params (Any): [parameters that are not compile time constants]
        static_params (Any): [parameters that are compie time constants]

    Returns:
        complex valued jnp.ndarray of shape (ds ** 2, K * ds ** 2): [the transition matrix]
    """

    shape = noisy_training_set.shape
    noisy_training_set = noisy_training_set.reshape((*shape[:-2], -1))
    noisy_training_set = _hankelizer(noisy_training_set, static_params.K)
    x, y = _hankel2xy(noisy_training_set)
    transition_matrix = _ridge_regression(x, y, dynamic_params.sigma, static_params.K)
    return transition_matrix[-shape[-1] ** 2:]
