from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import pytest
import jax.numpy as jnp
from jax import random

from dynamics_learner.benchmarking_utils import mesh_dynamic_params
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian

key = random.PRNGKey(42)
subkey1, subkey2 = random.split(key)

def test_mesh_dynamic_params():
    dynamic_params = JITDynamicParamsRandomLindbladian(
        key =                   jnp.array([subkey1, subkey2]),
        tau =                   jnp.array([2, 3]),
        hamiltonian_amplitude = jnp.array([4, 5]),
        dissipative_amplitude = jnp.array([6, 7]),
        sigma =                 jnp.array([8, 9]),
    )
    new_dynamic_params = mesh_dynamic_params(dynamic_params)
    print(new_dynamic_params.key)
    assert jnp.all(new_dynamic_params.key == jnp.array([subkey1, subkey2]).reshape((1, 1, 1, 1, 2, 2))), "key field is incorrect!"
    assert jnp.all(new_dynamic_params.tau == jnp.array([2, 3]).reshape((1, 1, 1, 2, 1))), "tay field is incorrect!"
    assert jnp.all(new_dynamic_params.hamiltonian_amplitude == jnp.array([4, 5]).reshape((1, 1, 2, 1, 1))), "hamiltonian_amplitude is incorrect!"
    assert jnp.all(new_dynamic_params.dissipative_amplitude == jnp.array([6, 7]).reshape((1, 2, 1, 1, 1))), "dissipative amplitude is incorrect!"
    assert jnp.all(new_dynamic_params.sigma == jnp.array([8, 9]).reshape((2, 1, 1, 1, 1))), "sigma field is incorrect!"
