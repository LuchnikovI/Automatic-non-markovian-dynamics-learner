from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import pytest
import jax.numpy as jnp
from jax import random
from jax.scipy.linalg import expm
from dynamics_learner.random_lindblad_dynamics_simulator import random_lindbladian, environment_steady_state, run_dynamics
from dynamics_learner.general_utils import random_pure_density_matrices

key = random.PRNGKey(42)


@pytest.mark.parametrize("key,dsde,hamiltonian_amplitude,dissipative_amplitude", [(key, 10, 1.5, 1.5)])
def test_random_lindbladian(
    key,
    dsde,
    hamiltonian_amplitude,
    dissipative_amplitude,
):
    lindbladian = random_lindbladian(key, hamiltonian_amplitude, dissipative_amplitude, dsde)
    phi = expm(lindbladian)
    phi = phi.reshape((dsde, dsde, dsde, dsde))
    assert jnp.linalg.norm(jnp.trace(phi, axis1=0, axis2=1) - jnp.eye(dsde)) < 1e-10, "The generated lindbladian does not preserve trace!"
    phi = phi.transpose((0, 2, 1, 3))
    phi = phi.reshape((dsde ** 2, dsde ** 2))
    assert jnp.linalg.norm(phi - phi.conj().T) < 1e-10, "The Choi matrix for time=1 channel build on top of the generated Lindbladian is not Hermitian!"
    assert jnp.min(jnp.linalg.eigvalsh(phi)) >= -1e-10, "The generated Lindbladian is not completely positive!"


@pytest.mark.parametrize("key,dsde,de,hamiltonian_amplitude,dissipative_amplitude", [(key, 9, 3, 1.5, 1.5)])
def test_environment_steady_state(
    key,
    dsde,
    de,
    hamiltonian_amplitude,
    dissipative_amplitude,
):
    lindbladian = random_lindbladian(key, hamiltonian_amplitude, dissipative_amplitude, dsde)
    rho = environment_steady_state(lindbladian, de)
    assert jnp.linalg.norm(rho - rho.conj().T) < 1e-10, "Environment density matrix is no Hermitian!"
    assert jnp.abs(jnp.trace(rho) - 1) < 1e-10, "Trace of the environment density matrix is not equal to 1!"
    assert jnp.min(jnp.linalg.eigvalsh(rho)) > -1e-10, "Environment density matrix is not non-negative!"
    lindbladian = random_lindbladian(key, hamiltonian_amplitude, dissipative_amplitude, dsde)
    rho = environment_steady_state(lindbladian, dsde)
    assert jnp.linalg.norm(lindbladian.dot(rho.reshape((-1,)))) < 1e-10, "The returned state is not steady!"


@pytest.mark.parametrize("key,dsde,de,hamiltonian_amplitude,dissipative_amplitude,tau,total_discrete_time", [(key, 9, 3, 1.5, 1.5, 0.5, 20)])
def test_run_dynamics(
    key,
    dsde,
    de,
    hamiltonian_amplitude,
    dissipative_amplitude,
    tau,
    total_discrete_time,
):
    lindbladian = random_lindbladian(key, hamiltonian_amplitude, dissipative_amplitude, dsde)
    env_steady = environment_steady_state(lindbladian, de)
    initial_state = random_pure_density_matrices(key, dsde // de, 3)
    print(initial_state.shape)
    dynamics = run_dynamics(lindbladian, env_steady, initial_state, tau, total_discrete_time)
    assert jnp.linalg.norm(jnp.trace(dynamics, axis1=2, axis2=3) - 1) < 1e-10, "Traces of simulated density matrices are incorrect!"
    assert jnp.linalg.norm(dynamics - dynamics.conj().transpose((0, 1, 3, 2))) < 1e-10, "Simulated density matrices are not Hermitian!"
    assert jnp.min(jnp.linalg.eigvalsh(dynamics)) > -1e-10, "Simulated density matrices are not non-negative!"
