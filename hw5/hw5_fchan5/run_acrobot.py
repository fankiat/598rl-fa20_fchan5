import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from actor_critic_model import Actor
import acrobot_with_target
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation

case = 4
if case == 1:
    DATA_DIR = 'random_acrobot_fixed_target'
    TARGET_BEHAVIOR = 'fixed'
    TASK = 'reach'
elif case == 2:
    DATA_DIR = 'random_acrobot_randomly_moving_target'
    TARGET_BEHAVIOR = 'random'
    TASK = 'reach'
elif case == 3:
    DATA_DIR = 'random_acrobot_forcing_on_target'
    TARGET_BEHAVIOR = 'forcing'
    TASK = 'reach'
elif case == 4:
    DATA_DIR = 'random_acrobot_forcing_on_target_collecting'
    TARGET_BEHAVIOR = 'forcing'
    TASK = 'collect'
else:
    DATA_DIR = 'test_MPI_save'
    TARGET_BEHAVIOR = 'forcing'
    TASK = 'collect'
IMG_DIR = os.path.join('images', DATA_DIR)

def main():
    print(f'Doing {DATA_DIR} with target {TARGET_BEHAVIOR} and task {TASK}')

    # Make folder to store plots and gifs
    if os.path.exists(IMG_DIR):
        os.system("rm -rf {}".format(IMG_DIR))
    os.makedirs(IMG_DIR)

    # Find latest iteration for post processing for each run
    folders = os.listdir(DATA_DIR)
    runs = [folder for folder in folders if 'run' in folder]
    runs_iters = {}
    for run in runs:
        files = os.listdir(os.path.join(DATA_DIR,run))
        fnames = []
        for f in files:
            if 'actor' in f and not 'trained' in f:
                fnames.append(int(f.split('_')[-1].replace('.pt', '')))
        latest_it = max(fnames)
        runs_iters[run] = latest_it

    # initialize environment
    env = acrobot_with_target.Acrobot(target_behavior=TARGET_BEHAVIOR, task=TASK)

    # (1) Plot learning and loss function curves
    plot_learning_curve(DATA_DIR, IMG_DIR, runs_iters)

    # (2) Example trajectory
    # (3) Animated gif of trajectory
    # Plot both trajectory and animation all in the same gif
    # We pick the first run here
    generate_animated_gif_with_trajectory_plots(env, runs_iters['run000'], os.path.join(DATA_DIR, 'run000'), IMG_DIR)

def plot_learning_curve(case_dir, save_dir, runs_iters):
    """
    Plot learning curves for actor and critic
    """
    fig = plt.figure(figsize=(6,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    color_palette = sns.color_palette("Set2", 4)
    cmap = ListedColormap(sns.color_palette(color_palette).as_hex())

    learning_curve = []
    loss_curve = []
    # plot up to the minimum steps for all the runs conducted for averaging, just in case
    max_iter = min(runs_iters.values())
    for run in runs_iters.keys():
        data = np.load(os.path.join(case_dir, run, "training_data.npy"),allow_pickle='TRUE').item()
        learning_curve.append(data['reward'][:max_iter])
        loss_curve.append(data['losses'][:max_iter])
    learning_curve = np.array(learning_curve)
    # compute the mean and standard dev from multiple runs
    learning_mean = np.mean(learning_curve, axis=0)
    learning_std = np.std(learning_curve, axis=0)
    loss_curve = np.array(loss_curve)
    # compute the mean and standard dev from multiple runs
    loss_mean = np.mean(loss_curve, axis=0)
    loss_std = np.std(loss_curve, axis=0)

    # Plot learning curve
    ax1.plot(
        data['step'][:max_iter], learning_mean,
        label='Total undiscounted reward', color="skyblue")
    # draw std bands
    ax1.fill_between(
        data['step'][:max_iter],
        learning_mean - learning_std,
        learning_mean + learning_std,
        color="skyblue", alpha=0.25)
    ax1.set_xlabel('Number of simulation steps')
    ax1.set_ylabel('Total discounted reward')
    ax1.set_title('Learning curves for actor and critic')
    ax1.legend()

    # Plot loss curve
    ax2.loglog(
        data['step'][:max_iter], loss_mean,
        label='MSE Loss', color="firebrick")
    # draw std bands
    ax2.fill_between(
        data['step'][:max_iter],
        loss_mean - loss_std,
        loss_mean + loss_std,
        color="firebrick", alpha=0.25)
    ax2.set_xlabel('Number of simulation steps')
    ax2.set_ylabel('MSE Loss')
    # ax2.set_title('')
    ax2.legend()

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_learning_curve.png'), dpi=200)


def generate_animated_gif_with_trajectory_plots(env, it, case_dir, save_dir, writer='imagemagick'):
    """
    Animate trajectory with synchronized trajectory plots
    """
    # initialize actor
    actor = Actor(env.num_states, env.num_actions)
     # Load trained actor
    trained_actor = torch.load(os.path.join(os.getcwd(), case_dir, 'actor_{:04d}.pt'.format(it)))
    actor.load_state_dict(trained_actor)

    # Initialize simulation
    s = env.reset()
    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [[0., 0.]],
        'r': [],
        'target_pos': [env.target_pos]
    }
    # Simulate until episode is done
    done = False
    while not done:
        # a = random.randrange(env.num_actions)
        with torch.no_grad():
            (mu, std) = actor(torch.from_numpy(s))
            dist = torch.distributions.normal.Normal(mu, std)
            a = dist.sample().numpy()
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)
        data['target_pos'].append(env.target_pos)
    data['s'] = np.array(data['s'])

    from matplotlib import gridspec
    from celluloid import Camera
    # Setup figure and subplots
    fig = plt.figure(figsize=(13,8))
    gs = gridspec.GridSpec(4,3)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])
    ax5 = plt.subplot(gs[0:4,1:3], autoscale_on=False, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
    camera = Camera(fig)

    # Parse data from simulation
    theta1 = data['s'][:, 0]
    theta2 = data['s'][:, 1]
    thetadot1 = data['s'][:, 2]
    thetadot2 = data['s'][:, 3]
    tau = np.array([env._a_to_u(a) for a in data['a']])

    for t in range(len(data['r'])):
        # Plot data and and take a snapshot
        ax1.plot(data['t'], theta1, label='theta1', color='deepskyblue')
        ax1.plot(data['t'], theta2, label='theta2', color='firebrick')
        ax1.legend(['theta1', 'theta2'])
        ax1.plot(data['t'][t], theta1[t], 'o', color='deepskyblue')
        ax1.plot(data['t'][t], theta2[t], 'o', color='firebrick')
        ax1.set_ylabel(r'$\theta$', fontsize=20)

        ax2.plot(data['t'], thetadot1, label='thetadot1', color='deepskyblue')
        ax2.plot(data['t'], thetadot2, label='thetadot2', color='firebrick')
        ax2.legend(['thetadot1', 'thetadot2'])
        ax2.plot(data['t'][t], thetadot1[t], 'o', color='deepskyblue')
        ax2.plot(data['t'][t], thetadot2[t], 'o', color='firebrick')
        ax2.set_ylabel(r'$\dot{\theta}$', fontsize=20)

        ax3.plot(data['t'], tau[:, 0], label='tau1', color='deepskyblue')
        ax3.plot(data['t'], tau[:, 1], label='tau2', color='firebrick')
        ax3.legend(['tau1', 'tau2'])
        ax3.plot(data['t'][t], tau[t, 0], 'o', color='deepskyblue')
        ax3.plot(data['t'][t], tau[t, 1], 'o', color='firebrick')
        ax3.set_ylabel(r'$\tau$', fontsize=20)

        ax4.plot(data['t'][:-1], data['r'], label='r', color='deepskyblue')
        ax4.legend(['r'])
        ax4.plot(data['t'][t], data['r'][t], 'o', color='deepskyblue')
        ax4.set_xlabel('Time step')
        ax4.set_ylabel('Reward')

        # Animation of the acrobot
        # Initialize the plot
        p1 = [ env.params['l1'] * np.sin(theta1[t]), -env.params['l1'] * np.cos(theta1[t]) ]
        p2 = [ p1[0] + env.params['l2'] * np.sin(theta1[t] + theta2[t]), p1[1] - env.params['l2'] * np.cos(theta1[t] + theta2[t])]
        ax5.plot([0, p1[0], p2[0]], [0, p1[1], p2[1]], 'ko-', lw=2)
        col = env._a_to_u(data['a'][t])
        # np.array([[0, 0], [p1[0], p1[1]], [p2[0], p2[1]]])
        scat = ax5.scatter([0, p1[0], p2[0]], [0, p1[1], p2[1]], c=np.append(col, 0), s=100, edgecolor="k", cmap='coolwarm', zorder=10, vmin=-env.max_tau, vmax=env.max_tau)
        ax5.scatter([data['target_pos'][t][0]], [data['target_pos'][t][1]], c='red', s=100, edgecolor="k", zorder=10)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(scat, cax=cax, orientation='vertical')
        # cbar = plt.colorbar(scat)
        cbar.set_label(r'$\tau$', fontsize=25)
        # ax5.set_title('time = {:3.1f}'.format(t * env.dt))
        ax5.text(0.41, 1.01, 'Time = {:3.1f}'.format(t * env.dt), transform=ax5.transAxes, fontsize=15)
        ax5.grid()

        # Take a snapshot
        camera.snap()

    # plt.tight_layout()
    animation = camera.animate()
    animation.save(os.path.join(save_dir, 'animated_trajectory_with_plots.gif'), writer=writer, fps=10, dpi=100)

if __name__ == "__main__":
    main()
