import jax.numpy as jnp

from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian
from dynamics_learner.general_utils import rho2bloch

def mesh_dynamic_params(
    dynamic_params: JITDynamicParamsRandomLindbladian,
) -> JITDynamicParamsRandomLindbladian:
    """[This function builds new set of dynamic paramaters that contains
        all combinations of parameters values of input dynamic parameters.
        For example, having 3 fields in dynamic_params [0, 1, 2], [3, 4, 5], [6, 7, 8]
        the function returns a new dynamic_params with fields 
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        [3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3, 4, 4, 4, 5, 5, 5, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8]]

    Args:
        dynamic_params (JITDynamicParamsRandomLindbladian): [dynamic parameters]

    Returns:
        JITDynamicParamsRandomLindbladian: [new dynamic parameters]
    """
    idx = jnp.arange(dynamic_params.key.shape[0])
    sigma, dissipative_amplitude, hamiltonian_amplitude, tau, idx = jnp.meshgrid(
        dynamic_params.sigma,
        dynamic_params.dissipative_amplitude,
        dynamic_params.hamiltonian_amplitude,
        dynamic_params.tau,
        idx,
        indexing='ij',
    )
    new_dynamic_params = JITDynamicParamsRandomLindbladian(
        key =                   dynamic_params.key[idx],
        tau =                   tau,
        hamiltonian_amplitude = hamiltonian_amplitude,
        dissipative_amplitude = dissipative_amplitude,
        sigma =                 sigma,
    )
    return new_dynamic_params

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
