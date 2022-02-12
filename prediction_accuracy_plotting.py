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

from experiments_utils import trace_distance
with open('finite_de_results.pickle', 'rb') as f:
    results = load(f)


# Visualization of prediction accuracy
# ============== Parameters ================#
ds = 2
test_set_size = 1
de = 4
training_set_size = 4
discrete_time_steps = 200
K_list = [5, 15, 25, 35, 45, 55, 65, 75, 85]
key = 44
tau = 0.2
hamiltonian_amplitude = 1.
dissipative_amplitude = 0.003
sigma_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
# ================================================ #

styles = ['o', '^', 'D', 's', '<']
sizes = [7, 7, 7, 7, 7]
cmap = mpl.cm.get_cmap('gist_heat')
colors = list(map(cmap, list(jnp.linspace(0, 0.8, 5))))

lines1 = []
lines2 = []
plt.figure()
for sigma, color, style, size in zip(sigma_list, colors, styles, sizes):
    dist_noisy = []
    dist_noiseless = []
    for K in K_list:
        data_point = results[de][training_set_size][discrete_time_steps][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma]
        noisy_test_set = data_point.noisy_test_set[:, K:]
        clean_test_set = data_point.clean_test_set[:, K:]
        predicted_test_set = data_point.predicted_test_set[:, K:]
        dist_noiseless.append(trace_distance(predicted_test_set, clean_test_set).mean())
        dist_noisy.append(trace_distance(predicted_test_set, noisy_test_set).mean())
    line, = plt.plot(K_list, dist_noiseless,
                '-' + style,
                color=color,
                markersize=size,
                linewidth=3)
    lines1.append(line)
    line, = plt.plot(K_list, dist_noisy,
                ':' + style,
                color=color,
                markersize=size,
                linewidth=3)
    lines2.append(line)
legend1 = plt.legend(lines1, list(map(lambda x: r'$\sigma={}$'.format(x), sigma_list)), fontsize=12, framealpha=1, loc=(0.01, 0.01), title=r'${\cal D}_{\rm noiseless}$', labelspacing=0)
legend2 = plt.legend(lines2, list(map(lambda x: r'$\sigma={}$'.format(x), sigma_list)), fontsize=12, framealpha=1, loc=(0.01, 0.38), title=r'${\cal D}_{\rm noisy}$', labelspacing=0)
plt.setp(legend1.get_title(),fontsize=15)
plt.setp(legend2.get_title(),fontsize=15)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.gcf().subplots_adjust(left=0.15)
plt.yscale('log')
plt.xticks(K_list)
plt.xlabel(r'$K$', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_accuracy.pdf')
