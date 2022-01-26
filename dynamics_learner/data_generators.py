import jax.numpy as jnp
from jax import jit, random
from typing import Tuple
from functools import partial

from dynamics_learner.random_lindblad_dynamics_simulator import random_lindbladian, environment_steady_state, run_dynamics
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian, JITStaticParamsRandomLindbladian, DataSet
from dynamics_learner.general_utils import random_pure_density_matrices, addnoise


@partial(jit, static_argnums=1)
def get_random_lindbladian_data(
    dynamic_params: JITDynamicParamsRandomLindbladian,
    static_params: JITStaticParamsRandomLindbladian,
) -> DataSet:
    """[This function generates data set of trajectories from the random Lindbladian model.]

    Args:
        dynamic_params (JITDynamicParamsRandomLindbladian): [parameters that are not compile time constants]
        static_params (JITStaticParamsRandomLindbladian): [parameters that are compie time constants]

    Returns:
        DataSet: [data set containing clean/noisy training/test sets]
    """

    subkey1, subkey2, subkey3 = random.split(dynamic_params.key, 3)
    batch_size = static_params.training_set_size + static_params.test_set_size
    dsde = static_params.ds * static_params.de
    lindbladian = random_lindbladian(subkey1, dynamic_params.hamiltonian_amplitude, dynamic_params.dissipative_amplitude, dsde)
    environment_initial_state = environment_steady_state(lindbladian, static_params.de)
    system_initial_states = random_pure_density_matrices(subkey2, static_params.ds, batch_size)
    dynamics_of_rho = run_dynamics(lindbladian, environment_initial_state, system_initial_states, dynamic_params.tau, static_params.discrete_time_steps)
    noisy_dynamics_of_rho = addnoise(subkey3, dynamics_of_rho, dynamic_params.sigma)
    data_set = DataSet(
        clean_training_set = dynamics_of_rho[:static_params.training_set_size],
        noisy_training_set = noisy_dynamics_of_rho[:static_params.training_set_size],
        clean_test_set = dynamics_of_rho[static_params.training_set_size:],
        noisy_test_set = noisy_dynamics_of_rho[static_params.training_set_size:],
        lindbladian = lindbladian,
    )
    return data_set
    