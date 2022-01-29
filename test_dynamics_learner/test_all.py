import pytest
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import expm
from dynamics_learner.data_generators import get_random_lindbladian_data
from dynamics_learner.data_learners import dmd_fit, tt_fit
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian, JITStaticParamsRandomLindbladian
from dynamics_learner.dynamics_predictors import dmd_predict_dynamics, tt_predict_dynamics


dynamic_params = JITDynamicParamsRandomLindbladian(
    key = random.PRNGKey(42),
    tau = 0.2,
    hamiltonian_amplitude = 1.,
    dissipative_amplitude = 0.005,
    sigma = 0.01,
)

static_params = JITStaticParamsRandomLindbladian(
    ds = 2,
    de = 4,
    discrete_time_steps = 200,
    training_set_size = 4,
    test_set_size = 4,
    K = 100,
)

@pytest.mark.parametrize("static_params,dynamic_params", [(static_params, dynamic_params)])
def test_all(
    static_params,
    dynamic_params,
):
    data_set = get_random_lindbladian_data(dynamic_params, static_params)

    dmd_model, denoised_training_set = dmd_fit(data_set.noisy_training_set, dynamic_params, static_params)
    dmd_model_jit, denoised_training_set_jit = dmd_fit(data_set.noisy_training_set, dynamic_params, static_params, True)
    tt_model = tt_fit(data_set.noisy_training_set, dynamic_params, static_params)
    
    assert jnp.linalg.norm(denoised_training_set_jit - denoised_training_set) < 1e-10, "Denosing for jit compatible == True/False is different!"

    err_noisy_vs_clean = jnp.linalg.norm(data_set.noisy_training_set[:, :-1] - data_set.clean_training_set[:, :-1])
    err_denoised_vs_clean = jnp.linalg.norm(denoised_training_set - data_set.clean_training_set[:, :-1])
    assert err_noisy_vs_clean > err_denoised_vs_clean, "Denoising fails!"

    exact_eigvals = jnp.linalg.eigvals(expm(dynamic_params.tau * data_set.lindbladian))
    reconstrucxted_eigvals = dmd_model.eigvals
    reconstrucxted_eigvals_jit = dmd_model_jit.eigvals
    eig_diff = reconstrucxted_eigvals[jnp.newaxis] - exact_eigvals[:, jnp.newaxis]
    
    assert jnp.max(jnp.abs(eig_diff).sort(0)[0]) < dynamic_params.sigma, "Reconstructed eigenvalues are too far from the exact ones!"

    dmd_test_predicted_from_clean = dmd_predict_dynamics(dmd_model, data_set.clean_test_set[:, :static_params.K], static_params.discrete_time_steps)
    dmd_test_predicted_from_clean_jit = dmd_predict_dynamics(dmd_model_jit, data_set.clean_test_set[:, :static_params.K], static_params.discrete_time_steps)
    tt_test_predicted_from_clean = tt_predict_dynamics(tt_model, data_set.clean_test_set[:, :static_params.K], static_params.discrete_time_steps)
    test_exact = data_set.clean_test_set
    
    assert jnp.linalg.norm(dmd_test_predicted_from_clean - dmd_test_predicted_from_clean_jit) < 1e-10, "Prediction for jit compatible == True/False is different!"

    assert jnp.linalg.norm(dmd_test_predicted_from_clean - test_exact) / jnp.linalg.norm(test_exact) < 3 * dynamic_params.sigma, "Prediction is too far from the exact test trajectories!"
    assert jnp.linalg.norm(tt_test_predicted_from_clean - test_exact) / jnp.linalg.norm(test_exact) < 3 * dynamic_params.sigma, "Prediction is too far from the exact test trajectories!"
