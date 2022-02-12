from jax.config import config
config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.style.use('seaborn-paper')
import jax.numpy as jnp
from pickle import load

from dynamics_learner.general_utils import rho2bloch

with open('finite_de_results.pickle', 'rb') as f:
    results = load(f)


# Visualization of dynamics
# ============== Parameters ================#
ds = 2
test_set_size = 1
de = 3
training_set_size = 4
discrete_taus = 200
K = 75
key = 44
tau = 0.2
hamiltonian_amplitude = 1.
dissipative_amplitude = 0.003
sigma_list = [1e-3, 1e-2, 3e-2, 1e-1]
# ================================================ #

fig = plt.figure(figsize=(7, 12))
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
ax.set_ylabel(r'$\langle\sigma_x\rangle$', fontsize=18, labelpad=30)
ax.set_xlabel(r'${\rm Time}$', fontsize=18, labelpad=10)

data_point = results[de][training_set_size][discrete_taus][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma_list[0]]
noisy_test_set = data_point.noisy_test_set
clean_test_set = data_point.clean_test_set
predicted_test_set = data_point.predicted_test_set[:, K:]
rank1 = data_point.learned_rank
X_test, X_predicted, X_test_noisy = rho2bloch(clean_test_set), rho2bloch(predicted_test_set), rho2bloch(noisy_test_set)
time_pred = jnp.arange(0, (X_predicted.shape[1]) * tau, tau) + K * tau
time_test = jnp.arange(0, (X_test.shape[1]) * tau, tau)

upper_bound = X_test_noisy[0, :, 0].max() + 0.12
lower_bound = X_test_noisy[0, :, 0].min() - 0.12

ax1 = fig.add_subplot(411)
ax1.fill_between((-100 * tau, K * tau), -1, 1, facecolor='red', alpha=0.3)
ax1.fill_between((K * tau, (time_test.shape[0] + 100) * tau), -1, 1, facecolor='green', alpha=0.3)
ax1.set_ylim(top=upper_bound, bottom=lower_bound)
ax1.set_xlim(left=0, right=time_test.shape[0]*tau)
plt.tick_params(bottom=False, labelbottom=False)
plt.text(27, 0.33, r'$\sigma={}$'.format(sigma_list[0]) + ', ' + r'$r={}$'.format(rank1), fontsize=15, bbox=dict(facecolor='white'))
plt.plot(time_test, X_test_noisy[0, :, 0], 'k', alpha=0.2, linewidth=3)
plt.plot(time_test, X_test[0, :, 0], 'b')
plt.plot(time_pred, X_predicted[0, :, 0], 'r--')
plt.tick_params(axis='both', which='major', labelsize=15)

data_point = results[de][training_set_size][discrete_taus][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma_list[1]]
noisy_test_set = data_point.noisy_test_set
clean_test_set = data_point.clean_test_set
predicted_test_set = data_point.predicted_test_set[:, K:]
rank2 = data_point.learned_rank
X_test, X_predicted, X_test_noisy = rho2bloch(clean_test_set), rho2bloch(predicted_test_set), rho2bloch(noisy_test_set)
time_pred = jnp.arange(0, (X_predicted.shape[1]) * tau, tau) + K * tau
time_test = jnp.arange(0, (X_test.shape[1]) * tau, tau)


ax2 = fig.add_subplot(412)
ax2.fill_between((-100 * tau, K * tau), -1, 1, facecolor='red', alpha=0.3)
ax2.fill_between((K * tau, (time_test.shape[0] + 100) * tau), -1, 1, facecolor='green', alpha=0.3)
ax2.set_ylim(top=upper_bound, bottom=lower_bound)
ax2.set_xlim(left=0, right=time_test.shape[0]*tau)
plt.tick_params(bottom=False, labelbottom=False)
plt.text(27, 0.33, r'$\sigma={}$'.format(sigma_list[1]) + ', ' + r'$r={}$'.format(rank2), fontsize=15, bbox=dict(facecolor='white'))
plt.plot(time_test, X_test_noisy[0, :, 0], 'k', alpha=0.2, linewidth=3)
plt.plot(time_test, X_test[0, :, 0], 'b')
plt.plot(time_pred, X_predicted[0, :, 0], 'r--')
plt.tick_params(axis='both', which='major', labelsize=15)

data_point = results[de][training_set_size][discrete_taus][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma_list[2]]
noisy_test_set = data_point.noisy_test_set
clean_test_set = data_point.clean_test_set
predicted_test_set = data_point.predicted_test_set[:, K:]
rank2 = data_point.learned_rank
X_test, X_predicted, X_test_noisy = rho2bloch(clean_test_set), rho2bloch(predicted_test_set), rho2bloch(noisy_test_set)
time_pred = jnp.arange(0, (X_predicted.shape[1]) * tau, tau) + K * tau
time_test = jnp.arange(0, (X_test.shape[1]) * tau, tau)


ax3 = fig.add_subplot(413)
ax3.fill_between((-100 * tau, K * tau), -1, 1, facecolor='red', alpha=0.3)
ax3.fill_between((K * tau, (time_test.shape[0] + 100) * tau), -1, 1, facecolor='green', alpha=0.3)
ax3.set_ylim(top=upper_bound, bottom=lower_bound)
ax3.set_xlim(left=0, right=time_test.shape[0]*tau)
plt.tick_params(bottom=False, labelbottom=False)
plt.text(27, 0.33, r'$\sigma={}$'.format(sigma_list[2]) + ', ' + r'$r={}$'.format(rank2), fontsize=15, bbox=dict(facecolor='white'))
plt.plot(time_test, X_test_noisy[0, :, 0], 'k', alpha=0.2, linewidth=3)
plt.plot(time_test, X_test[0, :, 0], 'b')
plt.plot(time_pred, X_predicted[0, :, 0], 'r--')
plt.tick_params(axis='both', which='major', labelsize=15)

data_point = results[de][training_set_size][discrete_taus][K][key][tau][hamiltonian_amplitude][dissipative_amplitude][sigma_list[3]]
noisy_test_set = data_point.noisy_test_set
clean_test_set = data_point.clean_test_set
predicted_test_set = data_point.predicted_test_set[:, K:]
rank2 = data_point.learned_rank
X_test, X_predicted, X_test_noisy = rho2bloch(clean_test_set), rho2bloch(predicted_test_set), rho2bloch(noisy_test_set)
time_pred = jnp.arange(0, (X_predicted.shape[1]) * tau, tau) + K * tau
time_test = jnp.arange(0, (X_test.shape[1]) * tau, tau)


ax4 = fig.add_subplot(414)
ax4.fill_between((-100 * tau, K * tau), -1, 1, facecolor='red', alpha=0.3)
ax4.fill_between((K * tau, (time_test.shape[0] + 100) * tau), -1, 1, facecolor='green', alpha=0.3)
ax4.set_ylim(top=upper_bound, bottom=lower_bound)
ax4.set_xlim(left=0, right=time_test.shape[0]*tau)
plt.text(27, 0.33, r'$\sigma={}$'.format(sigma_list[3]) + ', ' + r'$r={}$'.format(rank2), fontsize=15, bbox=dict(facecolor='white'))
plt.plot(time_test, X_test_noisy[0, :, 0], 'k', alpha=0.2, linewidth=3)
plt.plot(time_test, X_test[0, :, 0], 'b')
plt.plot(time_pred, X_predicted[0, :, 0], 'r--')
plt.tick_params(axis='both', which='major', labelsize=15)

plt.tight_layout()
plt.savefig('dynamics.pdf')
