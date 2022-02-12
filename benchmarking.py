from jax.config import config
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random
from typing import Tuple
from pickle import dump
from dynamics_learner.data_generators import get_random_lindbladian_data
from dynamics_learner.data_learners import dmd_fit, tt_fit
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian, JITStaticParamsRandomLindbladian
from dynamics_learner.dynamics_predictors import dmd_predict_dynamics, tt_predict_dynamics
from experiments_utils import trace_distance, logrange, experiments_vectorizer


# ================================================================================================= #
# This is the experiments runner for benchmarking of the proposed method of non-Markovian dynamics
# identification. By default it runs 300 000 experiments and evaluates accuracy for the proposed
# method and for the Transfer-Tensor method.
# ================================================================================================= #

# PRNGKeys
key = random.PRNGKey(42)
keys = random.split(key, 10)


@experiments_vectorizer
def benchmark(
    dynamic_params: JITDynamicParamsRandomLindbladian,
    static_params: JITStaticParamsRandomLindbladian,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """[For a given parameters this function calculates the accuracy of prediction
    for TT based and DMD based methods.]

    Args:
        dynamic_params (JITDynamicParamsRandomLindbladian): [run time dynamic params]
        static_params (JITStaticParamsRandomLindbladian): [compile time constant params]

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [dmd acuracy and tt acuracy]
    """
    
    data_set = get_random_lindbladian_data(dynamic_params, static_params)
    dmd_model, _ = dmd_fit(data_set.noisy_training_set, dynamic_params, static_params, jit_compatible=True)
    tt_model = tt_fit(data_set.noisy_training_set, dynamic_params, static_params)
    dmd_prediction = dmd_predict_dynamics(dmd_model, data_set.noisy_test_set[:, :static_params.K], static_params.discrete_time_steps)
    tt_prediction = tt_predict_dynamics(tt_model, data_set.noisy_test_set[:, :static_params.K], static_params.discrete_time_steps)
    dmd_dist = trace_distance(dmd_prediction[:, static_params.K:], data_set.clean_test_set[:, static_params.K:])
    dmd_dist = dmd_dist.mean()
    tt_dist = trace_distance(tt_prediction[:, static_params.K:], data_set.clean_test_set[:, static_params.K:])
    tt_dist = tt_dist.mean()
    return dmd_dist, tt_dist


# 'jitted' parameters
dynamic_params = JITDynamicParamsRandomLindbladian(
    key = keys,
    tau = logrange(0.01, 1., 5),
    hamiltonian_amplitude = jnp.array([1.,]),
    dissipative_amplitude = logrange(0.001, 0.1, 5),
    sigma = logrange(0.001, 0.1, 5),
)


# not 'jitted' parameters
static_params = JITStaticParamsRandomLindbladian(
    ds = 2,
    test_set_size = 4,
    de = (2, 3, 4, 5),
    training_set_size = (1, 2, 3, 4, 5),
    discrete_time_steps = (100, 150, 200),
    K = (5, 35, 65, 95),
)


# saving experiments' parameters
with open('benchmarking_parameters.pickle', 'wb') as f:
    dump((dynamic_params, static_params), f)


total_number_of_iterations = len(static_params.de) * len(static_params.training_set_size) \
                           * len(static_params.discrete_time_steps) * len(static_params.K)


# experiments
benchmarking_data = []
iter = 0
for de in static_params.de:
    for training_set_size in static_params.training_set_size:
        for discrete_time_steps in static_params.discrete_time_steps:
            for K in static_params.K:

                temp_static_params = JITStaticParamsRandomLindbladian(
                    ds = static_params.ds,
                    test_set_size = static_params.test_set_size,
                    de = de,
                    training_set_size = training_set_size,
                    discrete_time_steps = discrete_time_steps,
                    K = K,
                )
                dmd_err, tt_err =  benchmark(dynamic_params, temp_static_params)
                benchmarking_data.append((dmd_err, tt_err))
                with open('benchmarking_results.pickle', 'wb') as f:
                    dump(benchmarking_data, f)
                iter += 1
                print("Iteration #{} is done! The total number of iterations is {}".format(iter, total_number_of_iterations))
