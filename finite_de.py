from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
from pickle import dump
from dynamics_learner.data_generators import get_random_lindbladian_data
from dynamics_learner.data_learners import dmd_fit
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian, JITStaticParamsRandomLindbladian, MainExperimentResult
from dynamics_learner.dynamics_predictors import dmd_predict_dynamics


# ==================================================================== #
# This is the experiments runner for validation of the proposed methon #
# ==================================================================== #

def finite_de_experiment_runner(
    dynamic_params: JITDynamicParamsRandomLindbladian,
    static_params: JITStaticParamsRandomLindbladian,
) -> MainExperimentResult:
    """[This function runs experiment with predefined effective dimension non-Markovian dynamics]

    Args:
        dynamic_params (JITDynamicParamsRandomLindbladian): [run time dynamic params]
        static_params (JITStaticParamsRandomLindbladian): [compile time constant params]

    Returns:
        MainExperimentResult: [results]
    """

    data_set = get_random_lindbladian_data(dynamic_params, static_params)
    dmd_model, denoised_training_set = dmd_fit(data_set.noisy_training_set, dynamic_params, static_params, jit_compatible=True)
    dmd_prediction = dmd_predict_dynamics(dmd_model, data_set.noisy_test_set[:, :static_params.K], static_params.discrete_time_steps)
    results = MainExperimentResult(
        exact_rank = (static_params.de * static_params.ds) ** 2,
        learned_rank = dmd_model.rank,
        exact_eigenvalues = jnp.exp(dynamic_params.tau * jnp.linalg.eigvals(data_set.lindbladian)),
        learned_eigenvalues = dmd_model.eigvals,
        noisy_training_set = data_set.noisy_training_set,
        denoised_training_set = denoised_training_set,
        clean_training_set = data_set.clean_training_set,
        noisy_test_set = data_set.noisy_test_set,
        clean_test_set = data_set.clean_test_set,
        predicted_test_set = dmd_prediction,
    )
    return results

# parameters
ds = 2
test_set_size = 1

de_list = [2, 3, 4, 5, 6]
training_set_size_list = [4]
discrete_time_steps_list = [150, 200]
K_list = [5, 15, 25, 35, 45, 55, 65, 75, 85]
key_list = [44]
tau_list = [0.2]
hamiltonian_amplitude_list = [1.]
dissipative_amplitude_list = [0.003]
sigma_list = [1e-13, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


# experiments (this extremely ugly construction however simplify data plotting later)
data_dict = {}
for de in de_list:
    training_set_size_dict = {}
    for training_set_size in training_set_size_list:
        discrete_time_steps_dict = {}
        for discrete_time_steps in discrete_time_steps_list:
            K_dict = {}
            for K in K_list:
                key_dict = {}
                for key in key_list:
                    tau_dict = {}
                    for tau in tau_list:
                        hamiltonian_amplitude_dict = {}
                        for hamiltonian_amplitude in hamiltonian_amplitude_list:
                            dissipative_amplitude_dict = {}
                            for dissipative_amplitude in dissipative_amplitude_list:
                                sigma_dict = {}
                                for sigma in sigma_list:

                                    temp_static_params = JITStaticParamsRandomLindbladian(
                                        ds = ds,
                                        test_set_size = test_set_size,
                                        de = de,
                                        training_set_size = training_set_size,
                                        discrete_time_steps = discrete_time_steps,
                                        K = K,
                                    )
                                    
                                    temp_dynamic_params = JITDynamicParamsRandomLindbladian(
                                        key = random.PRNGKey(key),
                                        tau = tau,
                                        hamiltonian_amplitude = hamiltonian_amplitude,
                                        dissipative_amplitude = dissipative_amplitude,
                                        sigma = sigma,
                                    )
                                    results =  finite_de_experiment_runner(temp_dynamic_params, temp_static_params)
                                    sigma_dict[sigma] = results
                                dissipative_amplitude_dict[dissipative_amplitude] = sigma_dict
                            hamiltonian_amplitude_dict[hamiltonian_amplitude] = dissipative_amplitude_dict
                        tau_dict[tau] = hamiltonian_amplitude_dict
                    key_dict[key] = tau_dict
                K_dict[K] = key_dict
            discrete_time_steps_dict[discrete_time_steps] = K_dict
        training_set_size_dict[training_set_size] = discrete_time_steps_dict
    data_dict[de] = training_set_size_dict
with open('finite_de_results.pickle', 'wb') as f:
    dump(data_dict, f)
