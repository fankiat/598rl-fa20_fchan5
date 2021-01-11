# python_live_plot.py

import random
from itertools import count
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
import os

# plt.style.use('fivethirtyeight')

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='test_MPI_save')
parser.add_argument('--run_id', type=int, default=0)
args = parser.parse_args()

x_values = []
y_values = []

index = count()

# def plot_learning_curve(data):
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     ax.plot(data['step'], data['reward'], linewidth=2, label='total undiscounted reward')
#     ax.grid()
#     ax.legend(fontsize=14)
#     ax.tick_params(labelsize=14)
#     ax.set_xlabel('simulation steps', fontsize=20)
#     plt.tight_layout()
#     plt.show()

fig, ax1 = plt.subplots(figsize=(8,6))
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Total undiscounted reward')
ax1.grid()
ax1.set_title('Learning curve (run id : {})'.format(args.run_id))
line1 = ax1.plot([], [], 'deepskyblue', lw=2, label='Total undiscounted reward')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('MSE Loss')
line2 = ax2.semilogy([], [], 'firebrick', lw=2, label='MSE Loss')
# ax1.set_xlim([1.9e7, 3.5e7])
# ax1.set_ylim([-1.5, 0])

lines = line1 + line2
labs = [l.get_label() for l in lines]
ax1.legend(lines, labs, loc=0)

def animate(i):
    # data = np.load('test_save/run000/training_data.npy',allow_pickle='TRUE').item()
    data = np.load(os.path.join(args.save_dir, 'run{:03d}'.format(args.run_id), 'training_data.npy'),allow_pickle='TRUE').item()

    ax1.plot(data['step'], data['reward'], color='deepskyblue')
    ax2.semilogy(data['step'], data['losses'], color='firebrick')
    return lines
    # plt.cla()
    # plt.plot(data['step'], data['reward'], linewidth=2, label='total undiscounted reward')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('simulation steps', fontsize=20)
    # plt.ylabel('reward')
    # plt.title('Particle trajectory {}'.format(args.run_id))
    # plt.gcf().autofmt_xdate()
    # plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, 5000)
# ani = FuncAnimation(fig, animate, 5000, blit=True)

plt.tight_layout()
plt.show()