import numpy as np
import torch
import gridworld
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

DATA_DIR = [
    'reinforce',
    'reinforce_baseline_causality',
    'reinforce_baseline_importance_sampling',
    'reinforce_causality_importance_sampling',
    'reinforce_baseline_causality_importance_sampling'
    ]
# DATA_DIR = ['test_save']

IMG_DIR = 'images'

def main():
    # Make folder to store plots and gifs
    if os.path.exists(IMG_DIR):
        os.system("rm -rf {}".format(IMG_DIR))
    os.makedirs(IMG_DIR)

    # initialize environment
    env = gridworld.GridWorld()

    # (1) Plot learning and loss function curves
    plot_learning_curve(DATA_DIR, IMG_DIR, num_training_runs=100)

    # (2) visualize policy
    visualize_policy(DATA_DIR, IMG_DIR, num_training_runs=100)


def plot_learning_curve(case_dir, save_dir, num_training_runs=1):
    fig, ax = plt.subplots(
        6, 1, figsize=(8, 14),
        gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 2]})
    color_palette = sns.color_palette("Set2", len(case_dir))
    cmap = ListedColormap(sns.color_palette(color_palette).as_hex())
    for count, case in enumerate(case_dir):
        learning_curve = []
        for i in range(num_training_runs):
            learning_curve.append(np.load(os.path.join(case, "learning_curve_run{:02d}.npy".format(i))))
        learning_curve = np.array(learning_curve)
        # compute the mean and standard dev from multiple runs
        learning_mean = np.mean(learning_curve, axis=0)
        learning_std = np.std(learning_curve, axis=0)
        # draw lines
        ax[count].plot(
            learning_mean[:,0], learning_mean[:,1],
            label='{}'.format(case.replace('_', ' ')), color=cmap(count))
        ax[-1].plot(
            learning_mean[:,0], learning_mean[:,1],
            label='{}'.format(case.replace('_', ' ')), color=cmap(count))
        # draw std bands
        ax[count].fill_between(
            learning_mean[:,0],
            learning_mean[:,1] - learning_std[:,1],
            learning_mean[:,1] + learning_std[:,1],
            color=cmap(count),
            alpha=0.25)
        # ax.plot(
        #     data[:,0], data[:,1],
        #     linewidth=2, label=f"{case.replace('_', ' ')}")
        ax[count].grid()
        ax[count].set_xlim(0, 2e6)
        ax[count].set_ylim(0, 200)
        ax[count].legend(loc="lower right", fontsize=14)
        ax[count].set_ylabel('total reward', fontsize=16)
    ax[-1].grid()
    ax[-1].set_xlim(0, 2e6)
    ax[-1].set_ylim(0, 200)
    ax[-1].legend(loc="lower right", fontsize=8)
    ax[-1].set_ylabel('total reward', fontsize=16)
    # ax[count].tick_params(labelsize=14)
    ax[-1].set_xlabel('simulation steps', fontsize=16)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_dir, 'results_learning_curve_all.png'))

def visualize_policy(case_dir, save_dir, num_training_runs=10):
    ## greedy method
    def convert_action_to_quivers(Q):
        pi = np.argmax(Q, axis=1)
        u = np.zeros(pi.size)
        v = np.zeros(pi.size)
        for i, a in enumerate(pi):
            # right
            if a == 0:
                u[i] = 1
            elif a == 1:
                v[i] = 1
            elif a == 2:
                u[i] = -1
            elif a == 3:
                v[i] = -1
        teleport_zones = [1, 3]
        for tp in teleport_zones:
            u[tp] = 0
            v[tp] = 0

        return np.flip(u.reshape(5,5), axis=0), np.flip(v.reshape(5,5), axis=0)

    x = np.arange(0.5, 5.5, 1)
    X, Y = np.meshgrid(x, x)

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for count, case in enumerate(case_dir):
        theta = []
        for i in range(num_training_runs):
            theta.append(np.load(os.path.join(case, "theta_run{:02d}.npy".format(i))))
        theta = np.array(theta)
        # compute the mean and standard dev from multiple runs
        theta_mean = np.mean(theta, axis=0)
        U, V = convert_action_to_quivers(theta_mean)
        ax[count].quiver(X, Y, U, V, pivot="mid")
        ax[count].set_title('{}'.format(case.replace('_', ' ')), fontsize=10)
        ax[count].set_aspect('equal')
        # Turn off tick labels
        ax[count].set_xticklabels([])
        ax[count].set_yticklabels([])
        ax[count].set_xticks(range(len(x)+1))
        ax[count].set_yticks(range(len(x)+1))
        ax[count].xaxis.set_ticks_position('none')
        ax[count].yaxis.set_ticks_position('none')
        ax[count].grid(color="black")
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(save_dir, 'results_policy_all.png'))


if __name__ == "__main__":
    main()