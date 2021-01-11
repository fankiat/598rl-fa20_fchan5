import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import os
from learning_algorithms import policy_evaluation, policy_improvement
from learning_algorithms import value_iteration
from learning_algorithms import sarsa_td0, qlearning_td0, value_evaluation_td0
import multiprocessing

NUM_EPISODES = 2000
SAVE_DIR = 'gridworldData'
IMG_DIR = 'gridworldData/images'

def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

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

    # Q7 and Q8- visualize policy and value
    visualize_policy_value()


def plot_learning_curve():
    # load R for different methods
    R_policy_iteration = np.load(os.path.join(SAVE_DIR, 'policy_iteration_learning_curve.npy'))
    R_value_iteration = np.load(os.path.join(SAVE_DIR, 'value_iteration_learning_curve.npy'))
    R_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.10_epsilon0.10_learning_curve.npy'))
    R_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.10_epsilon0.10_learning_curve.npy'))

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(R_policy_iteration, label=r'Policy iteration ($\gamma=0.95$)')
    ax1.plot(R_value_iteration,  label=r'Value iteration ($\gamma=0.95$)')
    ax1.legend()
    ax1.set_xlabel('Number of value iterations')
    ax1.set_ylabel('Mean value function')
    ax1.set_title('Model-based learning curve for gridworld')

    # averaging over every 10 episodes, hence 10*100 simulation steps
    n_sim_step = np.arange(0, R_sarsa.size, 10)*100
    ax2.plot(n_sim_step, np.mean(R_sarsa.reshape(-1,10), axis=1), label=r'SARSA ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')
    n_sim_step = np.arange(0, R_qlearning.size, 10)*100
    ax2.plot(n_sim_step, np.mean(R_qlearning.reshape(-1,10), axis=1), label=r'Q-learning ($\alpha=0.1, \epsilon=0.1, \gamma=0.95$)')
    # n_sim_step = np.arange(R_sarsa.size)*100
    # ax2.plot(n_sim_step, R_sarsa, label=r'SARSA ($\alpha=0.5, \epsilon=0.1, \gamma=0.95$)')
    # n_sim_step = np.arange(R_qlearning.size)*100
    # ax2.plot(n_sim_step, R_qlearning, label=r'Q-learning ($\alpha=0.5, \epsilon=0.1, \gamma=0.95$)')
    ax2.legend()
    ax2.set_xlabel('Number of simulation steps')
    ax2.set_ylabel('Total discounted reward (Averaged over 10 episodes)')
    ax2.set_title('Model-free learning curve for gridworld')

    # plt.tight_layout()
    plt.show()

    fig.savefig(os.path.join(IMG_DIR, 'gridworld_learning_curve.png'))

    return

def plot_trajectory(env):
    # load pi for different trained agent and generate trajectories
    Q_policy_iteration = np.load(os.path.join(SAVE_DIR, 'policy_iteration_Q.npy'))
    Q_value_iteration = np.load(os.path.join(SAVE_DIR, 'value_iteration_Q.npy'))
    Q_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.50_epsilon0.10_Q.npy'))
    Q_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.50_epsilon0.10_Q.npy'))

    log_policy_iteration = run_agent(env, Q_policy_iteration)
    log_value_iteration = run_agent(env, Q_value_iteration)
    log_sarsa = run_agent(env, Q_sarsa)
    log_qlearning = run_agent(env, Q_qlearning)

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(log_policy_iteration['t'], log_policy_iteration['s'])
    ax1.plot(log_policy_iteration['t'][:-1], log_policy_iteration['a'])
    ax1.plot(log_policy_iteration['t'][:-1], log_policy_iteration['r'])
    ax1.set_xlim([0, 100])
    ax1.set_ylim([-5, 25])
    ax1.legend(['s', 'a', 'r'])
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('s')
    ax1.set_title('Policy iteration')

    ax2.plot(log_value_iteration['t'], log_value_iteration['s'])
    ax2.plot(log_value_iteration['t'][:-1], log_value_iteration['a'])
    ax2.plot(log_value_iteration['t'][:-1], log_value_iteration['r'])
    ax2.set_xlim([0, 100])
    ax2.set_ylim([-5, 25])
    ax2.legend(['s', 'a', 'r'])
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('s')
    ax2.set_title('Value iteration')

    ax3.plot(log_sarsa['t'], log_sarsa['s'])
    ax3.plot(log_sarsa['t'][:-1], log_sarsa['a'])
    ax3.plot(log_sarsa['t'][:-1], log_sarsa['r'])
    ax3.set_xlim([0, 100])
    ax3.set_ylim([-5, 25])
    ax3.legend(['s', 'a', 'r'])
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('s')
    ax3.set_title(r'SARSA ($\alpha=0.5, \epsilon=0.1$)')

    ax4.plot(log_qlearning['t'], log_qlearning['s'])
    ax4.plot(log_qlearning['t'][:-1], log_qlearning['a'])
    ax4.plot(log_qlearning['t'][:-1], log_qlearning['r'])
    ax4.set_xlim([0, 100])
    ax4.set_ylim([-5, 25])
    ax4.legend(['s', 'a', 'r'])
    ax4.set_xlabel('Time (t)')
    ax4.set_ylabel('s')
    ax4.set_title(r'Q-learning ($\alpha=0.5, \epsilon=0.1$)')

    plt.show()

    fig.savefig(os.path.join(IMG_DIR, 'gridworld_trajectory_different_method.png'))

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
    }

    # Simulate until episode is done
    done = False
    while not done:
    #     a = random.randrange(4)
        a = np.argmax(Q[s])
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

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
            n_sim_step = np.arange(0, R.size, 10)*100
            ax.plot(
                n_sim_step, np.mean(R.reshape(-1,10), axis=1),
                label=r'{} ($\alpha={}, \epsilon={}, \gamma=0.95$)'.format(
                    method, alpha, epsilon))
    ax.set_title(
        r'Learning curve for {} with $\alpha={}, \epsilon={}, \gamma=0.95$'.format(
                method,
                np.array2string(np.array(alphaList), separator=','),
                np.array2string(np.array(epsilonList), separator=',')))
    ax.legend()
    ax.set_xlabel('Number of simulation steps')
    ax.set_ylabel('Total discounted reward (Averaged over 10 episodes)')
    plt.show()

    if len(alphaList) == 1:
        # different epsilon
        figname = "gridworld_{}_different_epsilon_alpha{}.png".format(method, alphaList[0])
    elif len(epsilonList) == 1:
        # different alpha
        figname = "gridworld_{}_different_alpha_epsilon{}.png".format(method, epsilonList[0])
    fig.savefig(os.path.join(IMG_DIR, figname))

def visualize_policy_value():
    # load pi for different trained agent and generate trajectories
    Q_policy_iteration = np.load(os.path.join(SAVE_DIR, 'policy_iteration_Q.npy'))
    Q_value_iteration = np.load(os.path.join(SAVE_DIR, 'value_iteration_Q.npy'))
    Q_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.50_epsilon0.10_Q.npy'))
    # Q_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.50_epsilon0.10_Q.npy'))

    V_policy_iteration = np.load(os.path.join(SAVE_DIR, 'policy_iteration_V.npy'))
    V_value_iteration = np.load(os.path.join(SAVE_DIR, 'value_iteration_V.npy'))
    V_sarsa = np.load(os.path.join(SAVE_DIR, 'sarsa_alpha0.50_epsilon0.10_V.npy'))
    # V_qlearning = np.load(os.path.join(SAVE_DIR, 'qlearning_alpha0.50_epsilon0.10_V.npy'))

    # greedy method
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

    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    c = ax1.pcolor(
            np.flip(V_policy_iteration.reshape(5,5), axis=0),
            edgecolors='white', linewidths=4)
    fig.colorbar(c, ax=ax1)
    x = np.arange(0.5, 5.5, 1)
    X, Y = np.meshgrid(x, x)
    U, V = convert_action_to_quivers(Q_policy_iteration)
    ax1.quiver(X, Y, U, V, pivot="mid")
    # ax1.set_xticks(np.arange(0,6,1))
    # ax1.set_yticks(np.arange(0,6,1))
    # ax1.grid()
    ax1.set_xlim([0,5])
    ax1.set_ylim([0,5])
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax1.set_title(r'Policy iteration ($\gamma=0.95$)')

    c = ax2.pcolor(
            np.flip(V_value_iteration.reshape(5,5), axis=0),
            edgecolors='white', linewidths=4)
    fig.colorbar(c, ax=ax2)
    x = np.arange(0.5, 5.5, 1)
    X, Y = np.meshgrid(x, x)
    U, V = convert_action_to_quivers(Q_value_iteration)
    ax2.quiver(X, Y, U, V, pivot="mid")
    # ax2.set_xticks(np.arange(0,6,1))
    # ax2.set_yticks(np.arange(0,6,1))
    # ax2.grid()
    ax2.set_xlim([0,5])
    ax2.set_ylim([0,5])
    ax2.axis('off')
    ax2.set_aspect('equal')
    ax2.set_title(r'Value iteration ($\gamma=0.95$)')

    c = ax3.pcolor(
            np.flip(V_sarsa.reshape(5,5), axis=0),
            edgecolors='white', linewidths=4)
    fig.colorbar(c, ax=ax3)
    x = np.arange(0.5, 5.5, 1)
    X, Y = np.meshgrid(x, x)
    U, V = convert_action_to_quivers(Q_sarsa)
    ax3.quiver(X, Y, U, V, pivot="mid")
    # ax3.set_xticks(np.arange(0,6,1))
    # ax3.set_yticks(np.arange(0,6,1))
    # ax3.grid()
    ax3.set_xlim([0,5])
    ax3.set_ylim([0,5])
    ax3.axis('off')
    ax3.set_aspect('equal')
    ax3.set_title(r'SARSA ($\alpha=0.5, \epsilon=0.1, \gamma=0.95$)')

    c = ax4.pcolor(
            np.flip(V_qlearning.reshape(5,5), axis=0),
            edgecolors='white', linewidths=4)
    fig.colorbar(c, ax=ax4)
    x = np.arange(0.5, 5.5, 1)
    X, Y = np.meshgrid(x, x)
    U, V = convert_action_to_quivers(Q_qlearning)
    ax4.quiver(X, Y, U, V, pivot="mid")
    # ax4.set_xticks(np.arange(0,6,1))
    # ax4.set_yticks(np.arange(0,6,1))
    # ax4.grid()
    ax4.set_xlim([0,5])
    ax4.set_ylim([0,5])
    ax4.axis('off')
    ax4.set_aspect('equal')
    ax4.set_title(r'Q-learning ($\alpha=0.5, \epsilon=0.1, \gamma=0.95$)')

    plt.show()

    fig.savefig(os.path.join(IMG_DIR, 'gridworld_visualize_policy_value.png'))

if __name__ == "__main__":
    main()