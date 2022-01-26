import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from dynamics_learner.dataclasses import DMDDynamicsGenerator

def dmd_predict_dynamics(
    dynamics_generator: DMDDynamicsGenerator,
    input_prehistory: jnp.ndarray,
    discrete_time_steps: int
) -> jnp.ndarray:
    """[This function predicts dynamics based on the reconstructed DMD based model.]

    Args:
        dynamics_generator (DMDDynamicsGenerator): [reconstructed model]
        input_prehistory (complex valued jnp.ndarray of shape (batch_size, K, n, n)): [prehistory]
        discrete_time_steps (int): [number of time steps one builds prediction for]

    Returns:
        complex valued jnp.ndarray of shape (batch_size, discrete_time_steps, n, n): [predicted dynamics including the prehistory part]
    """

    shape = input_prehistory.shape
    hidden_state = jnp.tensordot(input_prehistory.reshape((*shape[:-3], -1)), dynamics_generator.encoder, axes=[[-1], [-1]])
    def iter(carry, xs):
        # carry is the current hidden state
        # ys is the current system state
        carry = carry * dynamics_generator.eigvals
        ys = jnp.tensordot(carry, dynamics_generator.decoder[-shape[-1] ** 2:], axes=[[-1], [-1]])
        ys = ys.reshape((*shape[:-3], shape[-2], shape[-1]))
        return carry, ys
    _, rhos = lax.scan(iter, hidden_state, None, length=discrete_time_steps-shape[-3])
    rhos = rhos.transpose((1, 0, 2, 3))
    return jnp.concatenate([input_prehistory, rhos], axis=-3)


def tt_predict_dynamics(
    dynamics_generator: jnp.ndarray,
    input_prehistory: jnp.ndarray,
    discrete_time_steps: int
) -> jnp.ndarray:

    """[This function predicts dynamics based on the reconstructed TT model.]

    Args:
        dynamics_generator (complex valued jnp.ndarray of shape (d ** 2, K * d ** 2)): [reconstructed model]
        input_prehistory (complex valued jnp.ndarray of shape (batch_size, K, n, n)): [prehistory]
        discrete_time_steps (int): [number of time steps one builds prediction for]

    Returns:
        complex valued jnp.ndarray of shape (batch_size, discrete_time_steps, n, n): [predicted dynamics including the prehistory part]
    """

    shape = input_prehistory.shape
    state = input_prehistory.reshape((*shape[:-3], -1))
    def iter(carry, xs):
        # carry is the current simulator state
        # ys is the current system state
        ys = jnp.tensordot(carry, dynamics_generator, axes=[[-1], [-1]])
        carry = jnp.concatenate([carry[..., shape[-1] ** 2:], ys], axis=-1)
        ys = ys.reshape((*shape[:-3], shape[-2], shape[-1]))
        return carry, ys
    _, rhos = lax.scan(iter, state, None, length=discrete_time_steps-shape[-3])
    rhos = rhos.transpose((1, 0, 2, 3))
    return jnp.concatenate([input_prehistory, rhos], axis=-3)
