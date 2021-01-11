# python_live_plot.py

import random
from itertools import count
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# plt.style.use('fivethirtyeight')

x_values = []
y_values = []

index = count()

# dslave_bubblematched = np.loadtxt('bubbledensitymatched/fish-diag_body_pos_tmpB01.dat')
# x_values_matched = dslave_bubblematched[:,1]
# y_values_matched = dslave_bubblematched[:,2]

# dslave_bubbleunmatched = np.loadtxt('bubbledensityunmatched/fish-diag_body_pos_tmpB01.dat')
# x_values_unmatched = dslave_bubbleunmatched[:,1]
# y_values_unmatched = dslave_bubbleunmatched[:,2]

# dslave_bubblematched_nobaro = np.loadtxt('bubbledensitymatched_nobaroclinic/fish-diag_body_pos_tmpB01.dat')
# x_values_matched_nobaro = dslave_bubblematched_nobaro[:,1]
# y_values_matched_nobaro = dslave_bubblematched_nobaro[:,2]

# dslave_bubblematched_nobaro_particledensitymatched = np.loadtxt('bubbledensitymatched_nobaroclinic_particledensitymatched/fish-diag_body_pos_tmpB01.dat')
# x_values_matched_nobaro_particledensitymatched = dslave_bubblematched_nobaro_particledensitymatched[:,1]
# y_values_matched_nobaro_particledensitymatched = dslave_bubblematched_nobaro_particledensitymatched[:,2]

# def plot_learning_curve(data):
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     ax.plot(data['step'], data['reward'], linewidth=2, label='total undiscounted reward')
#     ax.grid()
#     ax.legend(fontsize=14)
#     ax.tick_params(labelsize=14)
#     ax.set_xlabel('simulation steps', fontsize=20)
#     plt.tight_layout()
#     plt.show()

def animate(i):
    data = np.load('test_save/run000/training_data.npy',allow_pickle='TRUE').item()

    plt.cla()
    plt.plot(data['step'], data['reward'], linewidth=2, label='total undiscounted reward')

    # plt.xlim([0., 0.05])
    plt.ylim([-0.2, 1.])
    plt.grid()
    plt.legend()
    plt.xlabel('simulation steps', fontsize=20)
    plt.ylabel('reward')
    plt.title('Particle trajectory')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate, 5000)

plt.tight_layout()
plt.show()