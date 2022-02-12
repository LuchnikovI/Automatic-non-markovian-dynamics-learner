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

with open('finite_de_results.pickle', 'rb') as f:
    results = load(f)


# Visualization of how rank reconstruction works
# ============== Parameters ======================#
ds = 2
test_set_size = 1
de_list = [2, 3, 4, 5, 6]
training_set_size = 4
discrete_time_steps_list = [150, 200]
K = 75
key = 44
tau = 0.2
hamiltonian_amplitude = 1.
dissipative_amplitude = 0.003
sigma_list = [1e-13, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
# ================================================ #

cmap = mpl.cm.get_cmap('gist_heat')
line_types = ['-', ':']
styles = ['', 'o', '^', 'D', 's', '<']
sizes = [7, 7, 7, 7, 7, 7]
colors = [cmap(i) for i in jnp.linspace(0., 0.8, 6)]

legend = []
exact_rank_vs_de = []
plt.figure()
for de in de_list:
    data_point = results[de][training_set_size][discrete_time_steps_list[0]][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma_list[0]]
    exact_rank_vs_de.append(data_point.exact_rank)
plt.plot(de_list,
            exact_rank_vs_de,
            color='blue',
            markersize=7,
            linewidth=3)
legend.append(r'${\rm Exact \ value \ of \ }d^2d_{\rm E}^2$')
for discrete_time_steps, line_type in zip(discrete_time_steps_list, line_types):
    for sigma, color, style, size in zip(sigma_list, colors, styles, sizes):
        learned_rank_vs_de = []
        for de in de_list:
            data_point = results[de][training_set_size][discrete_time_steps][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma]
            learned_rank_vs_de.append(data_point.learned_rank)
        plt.plot(de_list, learned_rank_vs_de,
                 line_type + style,
                 color=color,
                 markersize=size,
                 linewidth=3)
        legend.append(r'$N={}$, $\sigma={}$'.format(discrete_time_steps, sigma if sigma>1e-13 else 0))
plt.legend(legend,
           fontsize=11,
           labelspacing=0.07,
           framealpha=1)
plt.xticks(de_list)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel(r'$d_{\rm E}$', fontsize=15)
plt.ylabel(r'$r$', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('rank_vs_de.pdf')
