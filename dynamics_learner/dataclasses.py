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


@chex.dataclass
class MainExperimentResult:
    """
    exact_rank: exact system and effective environment dimension
    learned_rank: reconstructed system and effective environment dimension
    exact_eigenvalues: exact eigenvalues of the Lindbladian
    learned_eigenvalues: learned eigenvalues of the Lindbladian
    noisy_training_set: noisy training set
    denoised_training_set: denoised training set
    clean_training_set: clean training set
    noisy_test_set: noisy test set
    clean_test_set: clean test set
    predicted_test_set: prediction based on noisy test set
    """

    exact_rank: chex.ArrayDevice
    learned_rank: chex.ArrayDevice
    exact_eigenvalues: chex.ArrayDevice
    learned_eigenvalues: chex.ArrayDevice
    noisy_training_set: chex.ArrayDevice
    denoised_training_set: chex.ArrayDevice
    clean_training_set: chex.ArrayDevice
    noisy_test_set: chex.ArrayDevice
    clean_test_set: chex.ArrayDevice
    predicted_test_set: chex.ArrayDevice
