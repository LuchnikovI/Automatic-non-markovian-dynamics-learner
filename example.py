import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import expm
from dynamics_learner.general_utils import rho2bloch
from dynamics_learner.data_generators import get_random_lindbladian_data
from dynamics_learner.data_learners import dmd_fit, tt_fit
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian, JITStaticParamsRandomLindbladian
from dynamics_learner.dynamics_predictors import dmd_predict_dynamics, tt_predict_dynamics


dynamic_params = JITDynamicParamsRandomLindbladian(
    key = random.PRNGKey(42),
    tau = 0.2,
    hamiltonian_amplitude = 1.,
    dissipative_amplitude = 0.005,
    sigma = 0.03,
)

static_params = JITStaticParamsRandomLindbladian(
    ds = 2,
    de = 4,
    discrete_time_steps = 200,
    training_set_size = 4,
    test_set_size = 4,
    K = 100,
)

data_set = get_random_lindbladian_data(dynamic_params, static_params)

dmd_model, denoised_training_set = dmd_fit(data_set.noisy_training_set, dynamic_params, static_params)
tt_model = tt_fit(data_set.noisy_training_set, dynamic_params, static_params)

bloch_training_denoised = rho2bloch(denoised_training_set)
bloch_training_noisy = rho2bloch(data_set.noisy_training_set)
bloch_training_clean = rho2bloch(data_set.clean_training_set)

plt.figure()
plt.plot(bloch_training_denoised[0, :, 2], 'r')
plt.plot(bloch_training_noisy[0, :, 2], '--k')
plt.plot(bloch_training_clean[0, :, 2], 'b')
plt.ylabel(r'$\sigma_z$', fontsize=14)
plt.xlabel('Discrete time')
plt.legend(['Denoised data', 'Noisy data', 'Clean data'])


exact_eigvals = jnp.linalg.eigvals(expm(dynamic_params.tau * data_set.lindbladian))
reconstrucxted_eigvals = dmd_model.eigvals

plt.figure()
plt.plot(exact_eigvals.real, exact_eigvals.imag, 'ok')
plt.plot(reconstrucxted_eigvals.real, reconstrucxted_eigvals.imag, '+r')
plt.legend(['Exact eigenvalues', 'Reconstructed eigenvalues'])

dmd_test_predicted = dmd_predict_dynamics(dmd_model, data_set.noisy_test_set[:, :static_params.K], static_params.discrete_time_steps)
tt_test_predicted = tt_predict_dynamics(tt_model, data_set.noisy_test_set[:, :static_params.K], static_params.discrete_time_steps)
test_exact = data_set.clean_test_set
bloch_dmd_test_predicted = rho2bloch(dmd_test_predicted)
bloch_tt_test_predicted = rho2bloch(tt_test_predicted)
bloch_test_exact = rho2bloch(test_exact)

plt.figure()
plt.plot(bloch_dmd_test_predicted[0, :, 2], 'b')
plt.plot(bloch_tt_test_predicted[0, :, 2], '--k')
plt.plot(bloch_test_exact[0, :, 2], 'r')
plt.ylabel(r'$\sigma_z$', fontsize=14)
plt.xlabel('Discrete time')
plt.legend(['DMD prediction', 'TT prediction', 'Exact'])

plt.show()
