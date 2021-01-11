import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
import os
from learning_algorithms import policy_evaluation, policy_improvement
from learning_algorithms import value_iteration
from learning_algorithms import sarsa_td0, qlearning_td0, value_evaluation_td0
import multiprocessing

# NUM_EPISODES = 50000
N_THETA = 31
N_THETADOT = 31
N_TAU = 31
SAVE_DIR = 'pendulumData'
IMG_DIR = 'pendulumData/images'

def main():
    # Create environment
    env = discrete_pendulum.Pendulum(n_theta=N_THETA, n_thetadot=N_THETADOT, n_tau=N_TAU)

    # Q1
    plot_learning_curve()

    # Q2
    plot_trajectory(env)

    alpha_epsilon = np.load(os.path.join(SAVE_DIR, 'alpha_epsilon_exploration.npy'), allow_pickle=True)
    alphaList = np.unique(alpha_epsilon[:,1])
    epsilonList = np.unique(alpha_epsilon[:,2])
    # Q3 - sarsa learning curve different epsilon
    plot_learning_curve_alpha_epsilon("sarsa", [0.1], epsilonList)
    # Q4 - sarsa learning curve different alpha
    plot_learning_curve_alpha_epsilon("sarsa", alphaList, [0.1])
    # Q5 - qlearning learning curve different epsilon
    plot_learning_curve_alpha_epsilon("qlearning", [0.1], epsilonList)
    # Q6 - qlearning learning curve different alpha
    plot_learning_curve_alpha_epsilon("qlearning", alphaList, [0.1])

    # Q7 - visualize policy
    visualize_policy(env)

    # Q8 - visualize value
    visualize_value(env)


def plot_learning_curve():
    # load R for different methods
    # R_policy_iteration = np.load(os.path.join(SAVE_DIR, 'policy_iteration_learning_curve.npy'))
    # R_value_iteration = np.load(os.path.join(SAVE_DIR, 'value_iteration_learning_curve.npy'))
    R_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.10_epsilon0.10_learning_curve.npy'))
    R_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.10_epsilon0.10_learning_curve.npy'))

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    # averaging over every 100 episodes, hence 100*100 simulation steps
    n_sim_step = np.arange(0, R_sarsa.size, 100)*100
    ax.plot(n_sim_step, np.mean(R_sarsa.reshape(-1,100), axis=1), label=r'SARSA ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')
    n_sim_step = np.arange(0, R_qlearning.size, 100)*100
    ax.plot(n_sim_step, np.mean(R_qlearning.reshape(-1,100), axis=1), label=r'Q-learning ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')
    # n_sim_step = np.arange(R_sarsa.size)*100
    # ax2.plot(n_sim_step, R_sarsa, label=r'SARSA ($\alpha=0.5, \epsilon=0.1, \gamma=0.95$)')
    # n_sim_step = np.arange(R_qlearning.size)*100
    # ax2.plot(n_sim_step, R_qlearning, label=r'Q-learning ($\alpha=0.5, \epsilon=0.1, \gamma=0.95$)')
    ax.set_ylim([-5,15])
    ax.legend()
    ax.set_xlabel('Number of simulation steps')
    ax.set_ylabel('Total discounted reward (Averaged over 100 episodes)')
    ax.set_title('Model-free learning curve for pendulum')

    # plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(IMG_DIR, 'pendulum_learning_curve.png'))

    return

def plot_trajectory(env):
    # load pi for different trained agent and generate trajectories
    Q_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.10_epsilon0.10_Q.npy'))
    Q_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.10_epsilon0.10_Q.npy'))

    log_sarsa = run_agent(env, Q_sarsa)
    log_qlearning = run_agent(env, Q_qlearning)

    fig, (
        (ax1, ax6),
        (ax2, ax7),
        (ax3, ax8),
        (ax4, ax9),
        (ax5, ax10)
        ) = plt.subplots(5, 2, sharex='col', figsize=(14,10))

    # Plot sarsa stuff
    ax1.plot(log_sarsa['t'], log_sarsa['s'])
    ax2.plot(log_sarsa['t'][:-1], log_sarsa['a'])
    ax3.plot(log_sarsa['t'][:-1], log_sarsa['r'])
    ax4.plot(log_sarsa['t'], log_sarsa['theta'])
    ax5.plot(log_sarsa['t'], log_sarsa['thetadot'])
    # Plot qlearning stuff
    ax6.plot(log_qlearning['t'], log_qlearning['s'])
    ax7.plot(log_qlearning['t'][:-1], log_qlearning['a'])
    ax8.plot(log_qlearning['t'][:-1], log_qlearning['r'])
    ax9.plot(log_qlearning['t'], log_qlearning['theta'])
    ax10.plot(log_qlearning['t'], log_qlearning['thetadot'])

    ax5.set_xlabel('Time (t)', fontsize=20)
    ax10.set_xlabel('Time (t)', fontsize=20)
    ax1.set_ylabel('s', fontsize=20)
    ax2.set_ylabel('a', fontsize=20)
    ax3.set_ylabel('r', fontsize=20)
    ax4.set_ylabel(r'$\theta$', fontsize=20)
    ax5.set_ylabel(r'$\dot{\theta}$',  fontsize=20)
    ax6.set_ylabel('s',  fontsize=20)
    ax7.set_ylabel('a', fontsize=20)
    ax8.set_ylabel('r', fontsize=20)
    ax9.set_ylabel(r'$\theta$', fontsize=20)
    ax10.set_ylabel(r'$\dot{\theta}$', fontsize=20)

    ax1.set_title(r'SARSA ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)', fontsize=20)
    ax6.set_title(r'Q-learning ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)', fontsize=20)

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:5]], visible=False)
    # plt.tight_layout()

    plt.show()

    fig.savefig(os.path.join(IMG_DIR, 'pendulum_trajectory_different_method.png'))

    return

def run_agent(env, Q):
    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    done = False
    while not done:
    #     a = random.randrange(env.n_tau)
        a = np.argmax(Q[s])
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])

    return log

def plot_learning_curve_alpha_epsilon(method, alphaList, epsilonList):
    if method != "sarsa" and method != "qlearning":
        raise ValueError("Method not defined.")
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    for alpha in alphaList:
        for epsilon in epsilonList:
            fname = \
                os.path.join(
                    SAVE_DIR,
                    '{}_alpha{:.2f}_epsilon{:.2f}_learning_curve.npy'.format(
                        method, alpha, epsilon))
            R = np.load(fname)
            # n_sim_step = np.arange(R.size)*100
            # averaging over every 10 episodes, hence 10*100 simulation steps
            n_sim_step = np.arange(0, R.size, 100)*100
            ax.plot(
                n_sim_step, np.mean(R.reshape(-1,100), axis=1),
                label=r'{} ($\alpha={}, \epsilon={}, \gamma=0.95$)'.format(
                    method, alpha, epsilon))
    ax.set_ylim([-20,20])
    ax.set_title(
        r'Learning curve for {} with $\alpha={}, \epsilon={}, \gamma=0.95$'.format(
                method,
                np.array2string(np.array(alphaList), separator=','),
                np.array2string(np.array(epsilonList), separator=',')))
    ax.legend()
    ax.set_xlabel('Number of simulation steps')
    ax.set_ylabel('Total discounted reward (Averaged over 100 episodes)')
    plt.show()

    if len(alphaList) == 1:
        # different epsilon
        figname = "pendulum_{}_different_epsilon_alpha{}.png".format(method, alphaList[0])
    elif len(epsilonList) == 1:
        # different alpha
        figname = "pendulum_{}_different_alpha_epsilon{}.png".format(method, epsilonList[0])
    fig.savefig(os.path.join(IMG_DIR, figname))


def visualize_policy(env):
    # load pi for different trained agent and generate action (torque) contours
    Q_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.10_epsilon0.10_Q.npy'))
    Q_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.10_epsilon0.10_Q.npy'))

    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    tau = env._a_to_u(np.argmax(Q_sarsa, axis=1))
    c = ax1.pcolor(
        tau.reshape(env.n_thetadot, env.n_theta),
        edgecolors='k', linewidths=1)
    fig.colorbar(c, ax=ax1)
    # ax1.axis('off')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(r'$\dot{\theta}$', fontsize=20)
    ax1.set_ylabel(r'$\theta$', fontsize=20)
    ax1.set_aspect('equal')
    ax1.set_title(r'SARSA ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')

    tau = env._a_to_u(np.argmax(Q_qlearning, axis=1))
    c = ax2.pcolor(
        tau.reshape(env.n_thetadot, env.n_theta),
        edgecolors='k', linewidths=1)
    fig.colorbar(c, ax=ax2)
    # ax1.axis('off')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel(r'$\dot{\theta}$', fontsize=20)
    ax2.set_ylabel(r'$\theta$', fontsize=20)
    ax2.set_aspect('equal')
    ax2.set_title(r'Q-learning ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')

    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(IMG_DIR, 'pendulum_visualize_policy.png'))

    return

def visualize_value(env):
    # load V fpr different trained agent
    V_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.10_epsilon0.10_V.npy'))
    V_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.10_epsilon0.10_V.npy'))


    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    c = ax1.pcolor(
        V_sarsa.reshape(env.n_thetadot, env.n_theta),
        edgecolors='k', linewidths=1)
    fig.colorbar(c, ax=ax1)
    # ax1.axis('off')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(r'$\dot{\theta}$', fontsize=20)
    ax1.set_ylabel(r'$\theta$', fontsize=20)
    ax1.set_aspect('equal')
    ax1.set_title(r'SARSA ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')

    c = ax2.pcolor(
        V_qlearning.reshape(env.n_thetadot, env.n_theta),
        edgecolors='k', linewidths=1)
    fig.colorbar(c, ax=ax2)
    # ax1.axis('off')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel(r'$\dot{\theta}$', fontsize=20)
    ax2.set_ylabel(r'$\theta$', fontsize=20)
    ax2.set_aspect('equal')
    ax2.set_title(r'Q-learning ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')

    plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(IMG_DIR, 'pendulum_visualize_value.png'))

    return


if __name__ == "__main__":
    main()