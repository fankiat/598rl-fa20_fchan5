import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import ListedColormap

SAVE_DIR = [
    'with_replay_with_target',
    'with_replay_without_target',
    'without_replay_with_target',
    'without_replay_without_target'
    ]
STEPS_PER_EPISODE = 100

fig = plt.figure()
ax = fig.add_subplot(111)
color_palette = sns.color_palette("Set2", 4)
cmap = ListedColormap(sns.color_palette(color_palette).as_hex())

# average_num = 50
# for count, case in enumerate(SAVE_DIR):
#     learning_curve = []
#     for i in range(10):
#         curve = np.load(os.path.join(case, "reward_trajectory_run{:01d}.npy".format(i)))[:-1]
#         n_sim_step = np.arange(0, curve.size, average_num) * STEPS_PER_EPISODE
#         # average every n episodes
#         curve = np.mean(curve.reshape(-1,average_num), axis=1)
#         learning_curve.append(curve)
#     learning_curve = np.array(learning_curve)
#     learning_mean_average = np.mean(learning_curve, axis=0)
#     learning_std_average = np.std(learning_curve, axis=0)
#     # n_sim_step = np.arange(0, learning_mean.size, average_num) * STEPS_PER_EPISODE
#     # learning_mean_average = np.mean(learning_mean.reshape(-1,average_num), axis=1)
#     # learning_std_average = np.mean(learning_std.reshape(-1,average_num), axis=1)
#     # draw lines
#     ax.plot(
#         n_sim_step, learning_mean_average,
#         label='{}'.format(case), color=cmap(count))
#     # draw std bands
#     ax.fill_between(n_sim_step, learning_mean_average - learning_std_average, learning_mean_average + learning_std_average, color=cmap(count), alpha=0.25)

for count, case in enumerate(SAVE_DIR):
    learning_curve = []
    for i in range(10):
        curve = np.load(os.path.join(case, "reward_trajectory_run{:01d}.npy".format(i)))[:-1]
        learning_curve.append(curve)
    learning_curve = np.array(learning_curve)
    learning_mean = np.mean(learning_curve, axis=0)
    learning_std = np.std(learning_curve, axis=0)
    average_num = 100
    n_sim_step = np.arange(0, learning_mean.size, average_num) * STEPS_PER_EPISODE
    learning_mean_average = np.mean(learning_mean.reshape(-1,average_num), axis=1)
    learning_std_average = np.mean(learning_std.reshape(-1,average_num), axis=1)
    # draw lines
    ax.plot(
        n_sim_step, learning_mean_average,
        label='{}'.format(case), color=cmap(count))
    # draw std bands
    ax.fill_between(n_sim_step, learning_mean_average - learning_std_average, learning_mean_average + learning_std_average, color=cmap(count), alpha=0.25)

ax.set_ylim([-10, 15])
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('results_learning_curve_all.png')