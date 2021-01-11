import numpy as np
import matplotlib.pyplot as plt
from actor_critic_model import Actor, Critic
import pendulum
import torch
import os
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns

DATA_DIR = 'PPO_750iteration_20eps_1000mbs_20opstep'
IMG_DIR = 'images'

def main():
    # Make folder to store plots and gifs
    if os.path.exists(IMG_DIR):
        os.system("rm -rf {}".format(IMG_DIR))
    os.makedirs(IMG_DIR)

    # initialize environment
    env = pendulum.Pendulum()

    # (1) Actor learning curve
    # (2) Critic learning curve
    plot_learning_curve(DATA_DIR, IMG_DIR, num_training_runs=40)

    # (3) Example trajectory
    plot_trajectory(env, os.path.join(DATA_DIR, 'run004'), IMG_DIR)

    # (4) Animated gif
    generate_animated_gif(env, os.path.join(DATA_DIR, 'run004'), IMG_DIR)

    # (5) Policy visualization
    plot_policy(env, os.path.join(DATA_DIR, 'run004'), IMG_DIR, resolution=512)

    # (6) Value visualization
    plot_value_function(env, os.path.join(DATA_DIR, 'run004'), IMG_DIR, resolution=512)

def plot_learning_curve(
    case_dir, save_dir, num_training_runs=10,
    num_episode_average=100, steps_per_episode=100):
    """
    Plot learning curve for actor and critic
    """
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    color_palette = sns.color_palette("Set2", 4)
    cmap = ListedColormap(sns.color_palette(color_palette).as_hex())

    # for count, case in enumerate(case_dir):
    actor_learning_curve = []
    critic_learning_curve = []
    for i in range(num_training_runs):
        # curve = np.load(os.path.join(case, "reward_trajectory_run{:01d}.npy".format(i)))[:-1]
        # learning_curve.append(curve)
        data = np.load(os.path.join(case_dir, 'run{:03d}'.format(i), 'training_data.npy'),allow_pickle='TRUE').item()
        actor_learning_curve.append(data['reward'])
        critic_learning_curve.append(data['losses'])
    # actor stuff
    actor_learning_curve = np.array(actor_learning_curve)
    # compute the mean and standard dev from multiple runs
    actor_learning_mean = np.mean(actor_learning_curve, axis=0)
    actor_learning_std = np.std(actor_learning_curve, axis=0)

    # critic stuff
    critic_learning_curve = np.array(critic_learning_curve)
    # compute the mean and standard dev from multiple runs
    critic_learning_mean = np.mean(critic_learning_curve, axis=0)
    critic_learning_std = np.std(critic_learning_curve, axis=0)

    # # now average over `num_episode_average` episodes
    # n_sim_step = np.arange(0, learning_mean.size, num_episode_average) * steps_per_episode
    # learning_mean_average = np.mean(learning_mean.reshape(-1,num_episode_average), axis=1)
    # learning_std_average = np.mean(learning_std.reshape(-1,num_episode_average), axis=1)

    # actor plots
    ax1.plot(
        data['step'], actor_learning_mean,
        label='Total undiscounted reward', color='skyblue')
    # draw std bands
    ax1.fill_between(
        data['step'],
        actor_learning_mean - actor_learning_std,
        actor_learning_mean + actor_learning_std,
        color='skyblue', alpha=0.25)
    ax1.set_ylim([-5, 0])
    ax1.set_xlabel('Number of simulation steps')
    ax1.set_ylabel('Total undiscounted reward')
    ax1.set_title('Actor learning curve')
    ax1.legend()

    # critic plots
    ax2.plot(
        data['step'], critic_learning_mean,
        label='MSE Loss', color='skyblue')
    # draw std bands
    ax2.fill_between(
        data['step'],
        critic_learning_mean - critic_learning_std,
        critic_learning_mean + critic_learning_std,
        color='skyblue', alpha=0.25)
    # ax2.set_ylim([-10, 15])
    ax2.set_xlabel('Number of simulation steps')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Critic learning curve')
    ax2.legend()

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_learning_curve.png'))


def plot_trajectory(env, case_dir, save_dir):
    """
    Plot trajectories for pendulum
    """
    # initialize actor
    actor = Actor(env.num_states, env.num_actions)
     # Load trained actor
    trained_actor = torch.load(os.path.join(os.getcwd(), case_dir, 'actor_trained.pt'))
    actor.load_state_dict(trained_actor)

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
        # a = np.array([random.gauss(0, 1)])
        (mu, std) = actor(torch.from_numpy(s))
        dist = torch.distributions.normal.Normal(mu, std)
        a = dist.sample().numpy()
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    data['a'] = np.array(data['a'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = data['a'][:, 0]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    # Plot data and save to png file
    # fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[0].set_ylabel(r'$\theta, \dot{\theta}$', fontsize=20)
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[1].set_ylabel(r'$\tau$', fontsize=20)
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    ax[2].set_ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results_trajectory_pendulum.png'))


def generate_animated_gif(env, case_dir, save_dir, writer='imagemagick'):
    """
    Animate trajectory for different trained agents
    """
    # initialize actor
    actor = Actor(env.num_states, env.num_actions)
     # Load trained actor
    trained_actor = torch.load(os.path.join(os.getcwd(), case_dir, 'actor_trained.pt'))
    actor.load_state_dict(trained_actor)

    s = env.reset()
    s_traj = [s]
    done = False
    while not done:
        (mu, std) = actor(torch.from_numpy(s))
        dist = torch.distributions.normal.Normal(mu, std)
        a = dist.sample().numpy()
        (s, r, done) = env.step(a)
        s_traj.append(s)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax.set_aspect('equal')
    ax.grid()
    line, = ax.plot([], [], 'o-', lw=2)
    text = ax.set_title('')

    def animate(i):
        theta = s_traj[i][0]
        line.set_data([0, -np.sin(theta)], [0, np.cos(theta)])
        text.set_text(f'time = {i * env.dt:3.1f}')
        return line, text

    anim = animation.FuncAnimation(fig, animate, len(s_traj), interval=(1000 * env.dt), blit=True, repeat=False)
    anim.save(os.path.join(save_dir, 'animated_trajectory.gif'), writer=writer, fps=10)

    plt.close()


def plot_policy(env, case_dir, save_dir, resolution=32):
    """
    Visualize policy
    """
    # Sample theta and thetadot
    theta = np.linspace(-np.pi, np.pi, resolution)
    thetadot = np.linspace(-env.max_thetadot, env.max_thetadot, resolution)
    # initialize actor
    actor = Actor(env.num_states, env.num_actions)
     # Load trained actor
    trained_actor = torch.load(os.path.join(os.getcwd(), case_dir, 'actor_trained.pt'))
    actor.load_state_dict(trained_actor)

    # num_cases = len(case_dir)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    state_space = np.meshgrid(theta, thetadot)
    policy = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            state = env._x_to_s([theta[i], thetadot[j]])
            with torch.no_grad():
                (mu, std) = actor(torch.from_numpy(state))
                # dist = torch.distributions.normal.Normal(mu, std)
                # action = dist.sample().numpy()
                action = mu.numpy()
            # policy[i,j] = env._a_to_u(action)
            policy[i,j] = np.clip(action[0], -env.max_tau, env.max_tau)

    c = ax.pcolor(state_space[0], state_space[1], policy)
    fig.colorbar(c, ax=ax, label=r"$\tau$")
    ax.set_xlabel(r'$\dot{\theta}$', fontsize=15)
    ax.set_ylabel(r'$\theta$', fontsize=15)
    ax.set_title('Policy visualization')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_policy.png'))


def plot_value_function(env, case_dir, save_dir, resolution=32):
    """
    Visualize policy
    """
    # Sample theta and thetadot
    theta = np.linspace(-np.pi, np.pi, resolution)
    thetadot = np.linspace(-env.max_thetadot, env.max_thetadot, resolution)
    # initialize critic
    critic = Critic(env.num_states, env.num_actions)
     # Load trained actor
    trained_critic = torch.load(os.path.join(os.getcwd(), case_dir, 'critic_trained.pt'))
    critic.load_state_dict(trained_critic)

    # num_cases = len(case_dir)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    state_space = np.meshgrid(theta, thetadot)
    value_func = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            state = env._x_to_s([theta[i], thetadot[j]])
            # with torch.no_grad():
            #     (mu, std) = actor(torch.from_numpy(state))
            #     dist = torch.distributions.normal.Normal(mu, std)
            #     action = dist.sample().numpy()
            # policy[i,j] = env._a_to_u(action)
            value_func[i,j] = critic(torch.from_numpy(state)).item()

    c = ax.pcolor(state_space[0], state_space[1], value_func)
    fig.colorbar(c, ax=ax, label=r"$\tau$")
    ax.set_xlabel(r'$\dot{\theta}$', fontsize=15)
    ax.set_ylabel(r'$\theta$', fontsize=15)
    ax.set_title('Value function visualization')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(save_dir, 'results_value_function.png'))

if __name__ == '__main__':
    main()