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

from experiments_utils import log_mean

key = random.PRNGKey(42)

with open('benchmarking_parameters.pickle', 'rb') as f:
    dynamic_parameters, static_parameters = load(f)

with open('benchmarking_results.pickle', 'rb') as f:
    dmd_errs, tt_errs = zip(*load(f))

# turning data in jnp.ndarrays
shape = (len(static_parameters.de),
         len(static_parameters.training_set_size),
         len(static_parameters.discrete_time_steps),
         len(static_parameters.K),
         *dmd_errs[0].shape)

dmd_errs = jnp.concatenate(tuple(map(lambda x: x[jnp.newaxis], dmd_errs)), axis=0)
dmd_errs = dmd_errs.reshape(shape)
tt_errs = jnp.concatenate(tuple(map(lambda x: x[jnp.newaxis], tt_errs)), axis=0)
tt_errs = tt_errs.reshape((shape))


fig = plt.figure()
ax = fig.add_subplot(2, 4, 1)
plt.ylim(top=1.3e-1, bottom=0.5e-2)
plt.plot(dynamic_parameters.tau, log_mean(dmd_errs, 'tau'), 'ok-')
plt.plot(dynamic_parameters.tau, log_mean(tt_errs, 'tau'), 'or-')
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$\left\langle\left\langle{\cal D}_{\rm noiseless}\right\rangle\right\rangle^{\log}$', fontsize=14)
plt.xlabel(r'$\tau$', fontsize=14)

ax = fig.add_subplot(2, 4, 2)
plt.ylim(top=1.3e-1, bottom=0.5e-2)
plt.plot(dynamic_parameters.dissipative_amplitude, log_mean(dmd_errs, 'dissipative_amplitude'), 'ok-')
plt.plot(dynamic_parameters.dissipative_amplitude, log_mean(tt_errs, 'dissipative_amplitude'), 'or-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$a_{\rm diss}$', fontsize=14)

ax = fig.add_subplot(2, 4, 3)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(dynamic_parameters.sigma, log_mean(dmd_errs, 'sigma'), 'ok-')
plt.plot(dynamic_parameters.sigma, log_mean(tt_errs, 'sigma'), 'or-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\sigma$', fontsize=14)

ax = fig.add_subplot(2, 4, 4)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(static_parameters.K, log_mean(dmd_errs, 'K'), 'ok-')
plt.plot(static_parameters.K, log_mean(tt_errs, 'K'), 'or-')
plt.xticks([5, 35, 65, 95])
plt.yscale('log')
plt.xlabel(r'$K$', fontsize=14)

ax = fig.add_subplot(2, 3, 4)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(static_parameters.discrete_time_steps, log_mean(dmd_errs, 'discrete_time_steps'), 'ok-')
plt.plot(static_parameters.discrete_time_steps, log_mean(tt_errs, 'discrete_time_steps'), 'or-')
plt.yscale('log')
plt.legend([r'${\rm Proposed \ method}$', r'${\rm Transfer-Tensor}$'], frameon=False, fontsize=9, loc=(0, 0), handlelength=0.4)
plt.ylabel(r'$\left\langle\left\langle{\cal D}_{\rm noiseless}\right\rangle\right\rangle^{\log}$', fontsize=14)
plt.xticks([100, 150, 200])
plt.xlabel(r'$N$', fontsize=14)

ax = fig.add_subplot(2, 3, 5)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(static_parameters.training_set_size, log_mean(dmd_errs, 'training_set_size'), 'ok-')
plt.plot(static_parameters.training_set_size, log_mean(tt_errs, 'training_set_size'), 'or-')
plt.xticks([1, 2, 3, 4, 5])
plt.yscale('log')
plt.xlabel(r'$L$', fontsize=14)

ax = fig.add_subplot(2, 3, 6)
plt.ylim(top=1.3e-1, bottom=1e-2)
plt.plot(static_parameters.de, log_mean(dmd_errs, 'de'), 'ok-')
plt.plot(static_parameters.de, log_mean(tt_errs, 'de'), 'or-')
plt.xticks([2, 3, 4, 5])
plt.yscale('log')
plt.xlabel(r'$d_{\rm E}$', fontsize=14)

fig.subplots_adjust(hspace=0.4, wspace=0.5)

plt.savefig("benchmarking.pdf")
