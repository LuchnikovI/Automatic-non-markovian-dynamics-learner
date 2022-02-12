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


# Visualization of denoising
# ============== Parameters ================#
ds = 2
test_set_size = 1
de_list = [3, 4, 5, 6]
training_set_size = 4
discrete_time_steps = 200
K = 75
key = 44
tau = 0.2
hamiltonian_amplitude = 1.
dissipative_amplitude = 0.003
sigma_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
# ================================================ #

styles = ['o', '^', 'D', 's', '<']
sizes = [7, 7, 7, 7, 7]
cmap = mpl.cm.get_cmap('gist_heat')
colors = list(map(cmap, list(jnp.linspace(0, 0.8, 4))))

lines1 = []
lines2 = []
plt.figure()
for de, color, style, size in zip(de_list, colors, styles, sizes):
    dist_noisy = []
    dist_noiseless = []
    for sigma in sigma_list:
        data_point = results[de][training_set_size][discrete_time_steps][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma]
        noisy_training_set = data_point.noisy_training_set[:, :-1]
        clean_training_set = data_point.clean_training_set[:, :-1]
        denoised_training_set = data_point.denoised_training_set
        dist_noiseless.append(trace_distance(denoised_training_set, clean_training_set).mean())
        dist_noisy.append(trace_distance(noisy_training_set, clean_training_set).mean())
    line, = plt.plot(sigma_list, dist_noiseless,
                '-' + style,
                color=color,
                markersize=size,
                linewidth=3)
    lines1.append(line)
    line, = plt.plot(sigma_list, dist_noisy,
                ':' + style,
                color='blue',
                markersize=size,
                linewidth=3)
    lines2.append(line)

legend1 = plt.legend(lines1, list(map(lambda x: r'$d_{\rm E}= \ $' + '$' + str(x) + '$', de_list)), fontsize=15, framealpha=1, title=r'$\langle{\cal D}_{\rm denoised}\rangle$', loc=4)
legend2 = plt.legend(lines2, list(map(lambda x: r'$d_{\rm E}= \ $' + '$' + str(x) + '$', de_list)), fontsize=15, framealpha=1, title=r'$\langle{\cal D}_{\rm noisy}\rangle$', loc=2)
plt.setp(legend1.get_title(),fontsize=15)
plt.setp(legend2.get_title(),fontsize=15)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.gcf().subplots_adjust(left=0.15)
plt.xticks(de_list)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\sigma$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig('denoising.pdf')
