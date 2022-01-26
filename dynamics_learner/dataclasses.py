import chex
from typing import Tuple
from dataclasses import dataclass

@chex.dataclass
class DMDDynamicsGenerator:
    decoder: chex.ArrayDevice
    eigvals: chex.ArrayDevice
    encoder: chex.ArrayDevice
    rank: chex.ArrayDevice

@chex.dataclass
class DataSet:
    clean_training_set: chex.ArrayDevice
    noisy_training_set: chex.ArrayDevice
    clean_test_set: chex.ArrayDevice
    noisy_test_set: chex.ArrayDevice
    lindbladian: chex.ArrayDevice

@chex.dataclass
class JITDynamicParamsRandomLindbladian:
    """
    key: PRNGKey
    tau: time step size
    haniltonian_amplitude: amplitude of the Hamiltonian part of the Lindbladian
    dissipative_amplitude: amplitude of the dissipative part of the Lindbladian
    sigma: std of noise
    """

    key: chex.ArrayDevice
    tau: chex.ArrayDevice
    hamiltonian_amplitude: chex.ArrayDevice
    dissipative_amplitude: chex.ArrayDevice
    sigma: chex.ArrayDevice
    
@dataclass(frozen=True)
class JITStaticParamsRandomLindbladian:
    """
    ds: Hilbert space's dimension of the system
    de: Hilbert space's dimension of the environment
    K: memory depth guess
    discrete_time_steps: number of discrete time steps in a trajectory
    training_set_size: number of trajectories in a training set
    test_set_size: number of trajectories in a test set
    """

    ds: int
    de: int
    K: int
    discrete_time_steps: int
    training_set_size: int
    test_set_size: int
