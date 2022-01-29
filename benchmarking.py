from jax.config import config
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random, vmap, jit
from functools import partial
from typing import Tuple
from pickle import dump
from dynamics_learner.data_generators import get_random_lindbladian_data
from dynamics_learner.data_learners import dmd_fit, tt_fit
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian, JITStaticParamsRandomLindbladian
from dynamics_learner.dynamics_predictors import dmd_predict_dynamics, tt_predict_dynamics
from dynamics_learner.benchmarking_utils import trace_distance, mesh_dynamic_params

key = random.PRNGKey(42)
keys = random.split(key, 10)

def logrange(min: float, max: float, n: int) -> jnp.ndarray:
    return jnp.exp(jnp.linspace(jnp.log(min), jnp.log(max), n))

@partial(jit, static_argnums=1)
@partial(vmap, in_axes=(0, None))
@partial(vmap, in_axes=(0, None))
@partial(vmap, in_axes=(0, None))
@partial(vmap, in_axes=(0, None))
@partial(vmap, in_axes=(0, None))
def tt_vs_dmd(
    dynamic_params: JITDynamicParamsRandomLindbladian,
    static_params: JITStaticParamsRandomLindbladian,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """[For a given parameters this function compares the accuracy of prediction
    for tt based and dmd based methods.]

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

dynamic_params = JITDynamicParamsRandomLindbladian(
    key = keys,
    tau = logrange(0.01, 1., 5),
    hamiltonian_amplitude = jnp.array([1.,]),
    dissipative_amplitude = logrange(0.001, 0.1, 5),
    sigma = logrange(0.001, 0.1, 5),
)

dynamic_params = mesh_dynamic_params(dynamic_params)

ds = 2
test_set_size =4
de_list = (2, 3, 4, 5)
training_set_size_list = (1, 2, 3, 4, 5)
discrete_time_steps_list = (100, 150, 200)
K_list = (5, 35, 65, 95)
total_number_of_iterations = len(de_list) * len(training_set_size_list) * len(discrete_time_steps_list) * len(K_list)

benchmarking_dict = {}
iter = 0

for de in de_list:
    for training_set_size in training_set_size_list:
        for discrete_time_steps in discrete_time_steps_list:
            for K in K_list:

                static_params = JITStaticParamsRandomLindbladian(
                    ds = ds,
                    de = de,
                    discrete_time_steps = discrete_time_steps,
                    training_set_size = training_set_size,
                    test_set_size = test_set_size,
                    K = K,
                )

                dmd_err, tt_err = tt_vs_dmd(dynamic_params, static_params)
                benchmarking_dict[static_params] = (dmd_err, tt_err)
                with open('banchmarking_results.pickle', 'wb') as f:
                    dump(benchmarking_dict, f)
                iter += 1
                print("Iteration #{} is done! The total number of iterations is {}".format(iter, total_number_of_iterations))
