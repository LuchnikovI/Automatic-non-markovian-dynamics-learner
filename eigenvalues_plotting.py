from jax.config import config
import matplotlib as mpl
config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.style.use('seaborn-paper')
import jax.numpy as jnp
from pickle import load

from dynamics_learner.dataclasses import JITStaticParamsRandomLindbladian
from experiments_utils import log_mean

with open('finite_de_results.pickle', 'rb') as f:
    results = load(f)


# Visualization of egenvalues coincidence
# ============== Parameters ======================#
ds = 2
test_set_size = 1
de = 4
training_set_size = 4
discrete_time_steps = 200
K = 75
key = 44
tau = 0.2
hamiltonian_amplitude = 1.
dissipative_amplitude = 0.003
sigma_list = [1e-13, 1e-2]
# ================================================ #

fig, ax1 = plt.subplots()

legend = []

# main plot
legend.append(r'${\rm Spectrum \ of} \ \exp\left(\tau {\cal L} \right)$')
data_point = results[de][training_set_size][discrete_time_steps][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma_list[0]]
plt.scatter(data_point.exact_eigenvalues.real,
            data_point.exact_eigenvalues.imag,
            color='blue',
            marker='+',
            s=200)
styles = ['x', '*']
colors = ['r', 'k']
sizes = [80, 80]
for sigma, color, style, size in zip(sigma_list, colors, styles, sizes):
    data_point = results[de][training_set_size][discrete_time_steps][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma]
    plt.scatter(data_point.learned_eigenvalues.real,
                data_point.learned_eigenvalues.imag,
                marker=style,
                color=color,
                s=size)
    legend.append(r'$\Lambda_r, \ \sigma={}$'.format(sigma if sigma > 1e-13 else 0))
plt.xlabel(r'${\rm Re}\lambda$', fontsize=15)
plt.ylabel(r'${\rm Im}\lambda$', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid(True)

# inset plot
left, bottom, width, height = [0.225, 0.34, 0.29, 0.45]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.set_xlim(left=0.96, right=1.002)
ax2.set_ylim(bottom=-0.3, top=0.3)

data_point = results[de][training_set_size][discrete_time_steps][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma_list[0]]
ax2.scatter(data_point.exact_eigenvalues.real,
            data_point.exact_eigenvalues.imag,
            color='blue',
            marker='+',
            s=200)
styles = ['x', '*']
colors = ['r', 'k']
sizes = [80, 80]
for sigma, color, style, size in zip(sigma_list, colors, styles, sizes):
    data_point = results[de][training_set_size][discrete_time_steps][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma]
    ax2.scatter(data_point.learned_eigenvalues.real,
                data_point.learned_eigenvalues.imag,
                marker=style,
                color=color,
                s=size)
ax2.tick_params(axis='both', which='major', labelsize=12)
plt.legend(legend,
           fontsize=13,
           framealpha=1,
           loc=(1.05, 0.3))
plt.grid(True)
plt.tight_layout()
plt.savefig('eigenvalues.pdf')
