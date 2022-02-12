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

# ======================================================================== #
# This piece of code illustrates a simple use case of the proposed method. #
# It generates a data set with parameters specified belowe, learns a model #
# of non-Markovian dynamics from generated data and generates several      #
# plots illustrating results.                                              #
# ======================================================================== #

# This is a wrapper for parameters that are jit compatible
# (they are not compile time constants)
dynamic_params = JITDynamicParamsRandomLindbladian(
    key = random.PRNGKey(42),       # random seed (PRNGKey)
    tau = 0.2,                      # time step
    hamiltonian_amplitude = 1.,     # Hamiltonian part amplitude of the generated Lindbladian
    dissipative_amplitude = 0.005,  # dissipative part amplitude of the generated Lindbladian
    sigma = 0.03,                   # data set noise amplitude
)

# This is a wrapper for parameters that are not jit compatible
# (compile time constant parameters)
static_params = JITStaticParamsRandomLindbladian(
    ds = 2,                     # dimension of the target system (2 for qubit, do not change!)
    de = 4,                     # dimension of the environment
    discrete_time_steps = 200,  # number of time steps in data set's trajectories
    training_set_size = 4,      # number of trajectories in a data set
    test_set_size = 4,          # number of trajectories a in test set
    K = 100,                    # memory depth (hyperparameter of the proposed method)
)

# data set generation
data_set = get_random_lindbladian_data(dynamic_params, static_params)

# model learning from data (proposed method, returns model and denoised data set)
dmd_model, denoised_training_set = dmd_fit(
    data_set.noisy_training_set,
    dynamic_params,
    static_params
)

# model learning (Transfer-Tensor method)
tt_model = tt_fit(
    data_set.noisy_training_set,
    dynamic_params,
    static_params
)

# density matrices to bloch vectors conversion
bloch_training_denoised = rho2bloch(denoised_training_set)
bloch_training_noisy = rho2bloch(data_set.noisy_training_set)
bloch_training_clean = rho2bloch(data_set.clean_training_set)

# plotting noisy data set, clean data set and denoised data set trajectories (<\sigma_z> vs discrete time)
plt.figure()
plt.plot(bloch_training_denoised[0, :, 2], 'r')
plt.plot(bloch_training_noisy[0, :, 2], '--k')
plt.plot(bloch_training_clean[0, :, 2], 'b')
plt.ylabel(r'$\sigma_z$', fontsize=14)
plt.xlabel('Discrete time')
plt.legend(['Denoised data', 'Noisy data', 'Clean data'])
plt.show()

# exact and learned eigenvalues
exact_eigvals = jnp.linalg.eigvals(expm(dynamic_params.tau * data_set.lindbladian))
reconstrucxted_eigvals = dmd_model.eigvals

# plotting exact and learned eigenvalues
plt.figure()
plt.plot(exact_eigvals.real, exact_eigvals.imag, 'ok')
plt.plot(reconstrucxted_eigvals.real, reconstrucxted_eigvals.imag, '+r')
plt.legend(['Exact eigenvalues', 'Reconstructed eigenvalues'])
plt.show()

# building prediction of the test set's trajectories (proposed method)
dmd_test_predicted = dmd_predict_dynamics(
    dmd_model,
    data_set.noisy_test_set[:, :static_params.K],
    static_params.discrete_time_steps
)

# building prediction of the test set's trajectories (transfer-Tensor method)
tt_test_predicted = tt_predict_dynamics(tt_model, data_set.noisy_test_set[:, :static_params.K], static_params.discrete_time_steps)

# test set exact trajectories
test_exact = data_set.clean_test_set

# density matrices to bloch vectors conversion
bloch_dmd_test_predicted = rho2bloch(dmd_test_predicted)
bloch_tt_test_predicted = rho2bloch(tt_test_predicted)
bloch_test_exact = rho2bloch(test_exact)

# plotting comparison of the propose method based prediction,
# Transfer-Tensor method based prediction,
# and exact test trajectories
plt.figure()
plt.plot(bloch_dmd_test_predicted[0, :, 2], 'b')
plt.plot(bloch_tt_test_predicted[0, :, 2], '--k')
plt.plot(bloch_test_exact[0, :, 2], 'r')
plt.ylabel(r'$\sigma_z$', fontsize=14)
plt.xlabel('Discrete time')
plt.legend(['DMD prediction', 'TT prediction', 'Exact'])

plt.show()
