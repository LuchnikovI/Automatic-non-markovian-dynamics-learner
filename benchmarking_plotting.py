from typing import Iterable
from jax.config import config
from jax import random
config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
import jax.numpy as jnp
from pickle import load
from math import e
from dynamics_learner.dataclasses import JITDynamicParamsRandomLindbladian, JITStaticParamsRandomLindbladian

key = random.PRNGKey(42)

def logrange(min: float, max: float, n: int) -> jnp.ndarray:
    return jnp.exp(jnp.linspace(jnp.log(min), jnp.log(max), n))

def log_mean(x: jnp.ndarray, indices:Iterable[int]) -> jnp.ndarray:
    return jnp.exp(jnp.log(x).mean(indices))

ds = 2
test_set_size = 4
de_list = (2, 3, 4, 5)
training_set_size_list = (1, 2, 3, 4, 5)
discrete_time_steps_list = (100, 150, 200)
K_list = (5, 35, 65, 95)
sigma = logrange(0.001, 0.1, 5)
dissipative_amplitude = logrange(0.001, 0.1, 5)
hamiltonian_amplitude = jnp.array([1.,])
tau = logrange(0.01, 1., 5)
keys = random.split(key, 10) 

with open('banchmarking_results.pickle', 'rb') as f:
    data = load(f)
dmd_errs, tt_errs = zip(*data.values())
dmd_errs = jnp.concatenate(tuple(map(lambda x: x[jnp.newaxis], dmd_errs)), axis=0)
dmd_errs = dmd_errs.reshape((4, 5, 3, 4, *dmd_errs.shape[1:]))
tt_errs = jnp.concatenate(tuple(map(lambda x: x[jnp.newaxis], tt_errs)), axis=0)
tt_errs = tt_errs.reshape((4, 5, 3, 4, *tt_errs.shape[1:]))

dmd_errs = dmd_errs.mean(-1)[..., 0, :]
tt_errs = tt_errs.mean(-1)[..., 0, :]


fig = plt.figure()
ax = fig.add_subplot(2, 4, 1)
plt.ylim(top=1.3e-1, bottom=0.5e-2)
plt.plot(tau, log_mean(dmd_errs, (0, 1, 2, 3, 4, 5)), 'ok-')
plt.plot(tau, log_mean(tt_errs, (0, 1, 2, 3, 4, 5)), 'or-')
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$\left\langle\left\langle{\cal D}_{\rm noiseless}\right\rangle\right\rangle$', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=14)

ax = fig.add_subplot(2, 4, 2)
plt.ylim(top=1.3e-1, bottom=0.5e-2)
plt.plot(dissipative_amplitude, log_mean(dmd_errs, (0, 1, 2, 3, 4, 6)), 'ok-')
plt.plot(dissipative_amplitude, log_mean(tt_errs, (0, 1, 2, 3, 4, 6)), 'or-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$a_{\rm diss}$', fontsize=14)

ax = fig.add_subplot(2, 4, 3)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(sigma, log_mean(dmd_errs, (0, 1, 2, 3, 5, 6)), 'ok-')
plt.plot(sigma, log_mean(tt_errs, (0, 1, 2, 3, 5, 6)), 'or-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\sigma$', fontsize=14)

ax = fig.add_subplot(2, 4, 4)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(K_list, log_mean(dmd_errs, (0, 1, 2, 4, 5, 6)), 'ok-')
plt.plot(K_list, log_mean(tt_errs, (0, 1, 2, 4, 5, 6)), 'or-')
plt.xticks([5, 35, 65, 95])
plt.yscale('log')
plt.xlabel(r'$K$', fontsize=14)

ax = fig.add_subplot(2, 3, 4)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(discrete_time_steps_list, log_mean(dmd_errs, (0, 1, 3, 4, 5, 6)), 'ok-')
plt.plot(discrete_time_steps_list, log_mean(tt_errs, (0, 1, 3, 4, 5, 6)), 'or-')
plt.yscale('log')
plt.legend([r'${\rm Proposed \ method}$', r'${\rm Transfer-Tensor}$'], frameon=False, fontsize=9, loc=(0, 0), handlelength=0.4)
plt.ylabel(r'$\left\langle\left\langle{\cal D}_{\rm noiseless}\right\rangle\right\rangle$', fontsize=14)
plt.xticks([100, 150, 200])
plt.xlabel(r'$N$', fontsize=14)

ax = fig.add_subplot(2, 3, 5)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(training_set_size_list, log_mean(dmd_errs, (0, 2, 3, 4, 5, 6)), 'ok-')
plt.plot(training_set_size_list, log_mean(tt_errs, (0, 2, 3, 4, 5, 6)), 'or-')
plt.xticks([1, 2, 3, 4, 5])
plt.yscale('log')
plt.xlabel(r'$L$', fontsize=14)

ax = fig.add_subplot(2, 3, 6)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(de_list, log_mean(dmd_errs, (1, 2, 3, 4, 5, 6)), 'ok-')
plt.plot(de_list, log_mean(tt_errs, (1, 2, 3, 4, 5, 6)), 'or-')
plt.xticks([2, 3, 4, 5])
plt.yscale('log')
plt.xlabel(r'$d_{\rm E}$', fontsize=14)

fig.subplots_adjust(hspace=0.4, wspace=0.5)

plt.savefig("benchmarking.pdf")
