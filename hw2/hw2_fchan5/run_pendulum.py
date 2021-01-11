import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import discreteaction_pendulum
from QNet import QNet
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

DATA_DIR = [
    'with_replay_with_target',
    'with_replay_without_target',
    'without_replay_with_target',
    'without_replay_without_target'
    ]
# DATA_DIR = ['test_save']
IMG_DIR = 'images'
STEPS_PER_EPISODE = 100

def main():
    # Make folder to store plots and gifs
    if os.path.exists(IMG_DIR):
        os.system("rm -rf {}".format(IMG_DIR))
    os.makedirs(IMG_DIR)

    # initialize environment
    env = discreteaction_pendulum.Pendulum()

    # (1) Plot learning and loss function curves
    plot_learning_curve(DATA_DIR, IMG_DIR)
    plot_loss_curve(DATA_DIR, IMG_DIR)

    # (2) Example trajectory
    plot_trajectory(env, DATA_DIR, IMG_DIR)

    # (3) Animated gif of trajectory
    generate_animated_gif(env, DATA_DIR, IMG_DIR)

    # (4) Compute policy and visualize
    plot_policy(env, DATA_DIR, IMG_DIR, resolution=256)

    # (5) Compute state value function and visualize
    plot_value_function(env, DATA_DIR, IMG_DIR, resolution=256)

def plot_learning_curve(
    case_dir, save_dir, num_training_runs=10,
    num_episode_average=100, steps_per_episode=100):
    """
    Plot learning curve for ablation study
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_palette = sns.color_palette("Set2", 4)
    cmap = ListedColormap(sns.color_palette(color_palette).as_hex())

    for count, case in enumerate(case_dir):
        learning_curve = []
        for i in range(num_training_runs):
            curve = np.load(os.path.join(case, "reward_trajectory_run{:01d}.npy".format(i)))[:-1]
            learning_curve.append(curve)
        learning_curve = np.array(learning_curve)
        # compute the mean and standard dev from multiple runs
        learning_mean = np.mean(learning_curve, axis=0)
        learning_std = np.std(learning_curve, axis=0)
        # now average over `num_episode_average` episodes
        n_sim_step = np.arange(0, learning_mean.size, num_episode_average) * steps_per_episode
        learning_mean_average = np.mean(learning_mean.reshape(-1,num_episode_average), axis=1)
        learning_std_average = np.mean(learning_std.reshape(-1,num_episode_average), axis=1)
        # draw lines
        ax.plot(
            n_sim_step, learning_mean_average,
            label='{}'.format(case.replace('_', ' ')), color=cmap(count))
        # draw std bands
        ax.fill_between(
            n_sim_step,
            learning_mean_average - learning_std_average,
            learning_mean_average + learning_std_average,
            color=cmap(count), alpha=0.25)

    ax.set_ylim([-10, 15])
    ax.set_xlabel('Number of simulation steps')
    ax.set_ylabel('Total discounted reward (Averaged over {} episodes)'.format(num_episode_average))
    ax.set_title('Learning curve for pendulum with different conditions')
    ax.legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_learning_curve_all.png'))

def plot_loss_curve(
    case_dir, save_dir, num_training_runs=10,
    num_episode_average=10, steps_per_episode=100):
    """
    Plot learning curve for ablation study
    """
    num_cases = len(case_dir)
    fig, ax = plt.subplots(num_cases, 1, figsize=(9, 3*num_cases))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    color_palette = sns.color_palette("Set2", 4)
    cmap = ListedColormap(sns.color_palette(color_palette).as_hex())

    for count, case in enumerate(case_dir):
        loss_curve = []
        for i in range(num_training_runs):
            curve = np.load(os.path.join(case, "loss_trajectory_run{:01d}.npy".format(i)))[:-1]
            loss_curve.append(curve)
        loss_curve = np.array(loss_curve) / steps_per_episode
        # compute the mean and standard dev from multiple runs
        loss_mean = np.mean(loss_curve, axis=0)
        # loss_std = np.std(loss_curve, axis=0)
        # now average over `num_episode_average` episodes
        n_sim_step = np.arange(0, loss_mean.size, num_episode_average) * steps_per_episode
        loss_mean_average = np.mean(loss_mean.reshape(-1,num_episode_average), axis=1)
        # loss_std_average = np.mean(loss_std.reshape(-1,num_episode_average), axis=1)
        # draw lines
        ax[count].semilogy(
            n_sim_step, loss_mean_average,
            label='{}'.format(case.replace('_', ' ')), color=cmap(count), alpha=0.7)
        # draw std bands
        # ax.fill_between(
        #     n_sim_step,
        #     loss_mean_average, #- loss_std_average,
        #     loss_mean_average + loss_std_average,
        #     color=cmap(count), alpha=0.25)

        ax[count].set_xlabel('Number of simulation steps')
        ax[count].set_ylabel('Total loss (Averaged over {} episodes)'.format(num_episode_average), fontsize=8)
        ax[count].legend(loc="upper right")
        ax[count].set_ylim([0.1,100])

    ax[0].set_title('Loss function curve for pendulum {}'.format(case.replace('_', ' ')))
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_loss_curve_all.png'), dpi=300)


def plot_trajectory(env, case_dir, save_dir):
    """
    Plot trajectories for pendulum
    """
    num_cases = len(case_dir)
    fig, ax = plt.subplots(3, num_cases, figsize=(4*num_cases, 10))
    # initialize QNet
    Q_net = QNet(env.num_states, env.num_actions)
    # loop over all cases and plot them accordingly on the same plot
    for case_idx, case in enumerate(case_dir):
        # Load trained Q network
        trained_network = torch.load(os.path.join(os.getcwd(), case, 'Q_network.pt'))
        Q_net.load_state_dict(trained_network)

        # Initialize simulation
        s = env.reset()
        # Create dict to store data from simulation
        data = {
            't': [0],
            's': [s],
            'a': [],
            'r': [],
        }
        # Simulate until episode is done
        done = False
        while not done:
            # a = random.randrange(env.num_actions)
            with torch.no_grad():
                Q = Q_net(torch.Tensor(s))
                a = np.argmax(Q.numpy())
            (s, r, done) = env.step(a)
            data['t'].append(data['t'][-1] + 1)
            data['s'].append(s)
            data['a'].append(a)
            data['r'].append(r)

        # Parse data from simulation
        data['s'] = np.array(data['s'])
        theta = data['s'][:, 0]
        thetadot = data['s'][:, 1]
        tau = [env._a_to_u(a) for a in data['a']]

        # Plot data and save to png file
        # fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0][case_idx].plot(data['t'], theta, label='theta')
        ax[0][case_idx].plot(data['t'], thetadot, label='thetadot')
        ax[0][case_idx].legend()
        ax[0][case_idx].set_ylabel(r'$\theta, \dot{\theta}$', fontsize=20)
        ax[0][case_idx].set_title(case.replace('_',' '))
        ax[1][case_idx].plot(data['t'][:-1], tau, label='tau')
        ax[1][case_idx].legend()
        ax[1][case_idx].set_ylabel(r'$\tau$', fontsize=20)
        ax[2][case_idx].plot(data['t'][:-1], data['r'], label='r')
        ax[2][case_idx].legend()
        ax[2][case_idx].set_xlabel('time step')
        ax[2][case_idx].set_ylabel('Reward')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'results_trajectory_all.png'))

# Copied some code from environment class to plot pendulums with different training conditions
# The Q network is reloaded and simulation is re-ran here to preserve function modularity
def generate_animated_gif(env, case_dir, save_dir, writer='imagemagick'):
    """
    Animate trajectory for different trained agents
    """
    s_traj_all = []
    # initialize Q network
    Q_net = QNet(env.num_states, env.num_actions)
    for case_idx, case in enumerate(case_dir):
        # Initialize simulation
        s = env.reset()
        # Load trained Q network
        trained_network = torch.load(os.path.join(os.getcwd(), case, 'Q_network.pt'))
        Q_net.load_state_dict(trained_network)

        s_traj = [s]
        # Simulate until episode is done
        done = False
        while not done:
            with torch.no_grad():
                Q = Q_net(torch.Tensor(s))
                a = np.argmax(Q.numpy())
            (s, r, done) = env.step(a)
            # (s, r, done) = env.step(policy(s))
            s_traj.append(s)

        s_traj_all.append(s_traj)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax.set_aspect('equal')
    ax.grid()
    text = ax.set_title('')

    color_palette = sns.color_palette("Set2", len(case_dir))
    cmap = ListedColormap(sns.color_palette(color_palette).as_hex())
    lines = []
    for case_idx, case in enumerate(case_dir):
        lobj = ax.plot([], [], 'o-', lw=2, markersize=12, color=cmap(case_idx))[0]
        lines.append(lobj)

    def animate(i):
        x1, y1 = [], []
        x2, y2 = [], []
        for case_idx, case in enumerate(case_dir):
            theta = s_traj_all[case_idx][i][0]
            x1.append(0)
            y1.append(0)
            x2.append(-np.sin(theta))
            y2.append(np.cos(theta))
        xlist = np.array([x1, x2]).T
        ylist = np.array([y1, y2]).T

        for lnum, line in enumerate(lines):
            # print(lnum)
            # print(xlist)
            line.set_data(xlist[lnum], ylist[lnum])
            line.set_label(case_dir[lnum].replace('_', ' '))

        text.set_text(f'time = {i * env.dt:3.1f}')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.005),
          ncol=2, fancybox=True, shadow=True)
        return line, text

    anim = animation.FuncAnimation(
        fig, animate, len(s_traj), interval=(1000 * env.dt),
        blit=True, repeat=False)
    anim.save(os.path.join(save_dir, 'animated_trajectory_all.gif'), writer=writer, fps=10)

def plot_policy(env, case_dir, save_dir, resolution=32):
    """
    Visualize policy
    """
    # Sample theta and thetadot
    theta = np.linspace(-np.pi, np.pi, resolution)
    thetadot = np.linspace(-env.max_thetadot, env.max_thetadot, resolution)
    # initialize QNet
    Q_net = QNet(env.num_states, env.num_actions)

    num_cases = len(case_dir)
    fig, ax = plt.subplots(1, num_cases, figsize=(6*num_cases, 4.5))

    for case_idx, case in enumerate(case_dir):
        # Load trained Q network
        trained_network = torch.load(os.path.join(os.getcwd(), case, 'Q_network.pt'))
        Q_net.load_state_dict(trained_network)

        state_space = np.meshgrid(theta, thetadot)
        policy = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                state = env._x_to_s([theta[i], thetadot[j]])
                with torch.no_grad():
                    Q = Q_net(torch.Tensor(state))
                    action = np.argmax(Q.numpy())
                policy[i,j] = env._a_to_u(action)

        c = ax[case_idx].pcolor(state_space[0], state_space[1], policy)
        fig.colorbar(c, ax=ax[case_idx], label=r"$\tau$")
        ax[case_idx].set_xlabel(r'$\dot{\theta}$', fontsize=15)
        ax[case_idx].set_ylabel(r'$\theta$', fontsize=15)
        ax[case_idx].set_title(case.replace('_', ' '))
        ax[case_idx].set_aspect(1.0/ax[case_idx].get_data_ratio(), adjustable='box')


    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_policy_all.png'))


def plot_value_function(env, case_dir, save_dir, resolution=32):
    """
    Visualize value function
    """
    # Sample theta and thetadot
    theta = np.linspace(-np.pi, np.pi, resolution)
    thetadot = np.linspace(-env.max_thetadot, env.max_thetadot, resolution)
    # initialize QNet
    Q_net = QNet(env.num_states, env.num_actions)

    num_cases = len(case_dir)
    fig, ax = plt.subplots(1, num_cases, figsize=(6*num_cases, 4.5))

    for case_idx, case in enumerate(case_dir):
        # Load trained Q network
        trained_network = torch.load(os.path.join(os.getcwd(), case, 'Q_network.pt'))
        Q_net.load_state_dict(trained_network)

        state_space = np.meshgrid(theta, thetadot)
        V = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                state = env._x_to_s([theta[i], thetadot[j]])
                with torch.no_grad():
                    Q = Q_net(torch.Tensor(state))
                V[i,j] = np.max(Q.numpy())

        c = ax[case_idx].pcolor(state_space[0], state_space[1], V)
        fig.colorbar(c, ax=ax[case_idx], label='value function')
        ax[case_idx].set_xlabel(r'$\dot{\theta}$', fontsize=15)
        ax[case_idx].set_ylabel(r'$\theta$', fontsize=15)
        ax[case_idx].set_title(case.replace('_', ' '))
        ax[case_idx].set_aspect(1.0/ax[case_idx].get_data_ratio(), adjustable='box')


    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_value_function_all.png'))


if __name__ == "__main__":
    main()
