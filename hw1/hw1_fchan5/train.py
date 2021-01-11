import os
import numpy as np
from learning_algorithms import policy_evaluation, policy_improvement
from learning_algorithms import value_iteration
from learning_algorithms import sarsa_td0, qlearning_td0, value_evaluation_td0

GAMMA = 0.95

def train_policy_iteration(env, save_dir, gamma=GAMMA, *args, **kwargs):
    # Create environment
    # env = discrete_pendulum.Pendulum(n_theta=21, n_thetadot=21, n_tau=21)
    # env = gridworld.GridWorld(hard_version=False)

    # initializing
    # V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))

    policy_stable = False
    meanV_trajectory = []
    while not policy_stable:
        V, Q, meanV = policy_evaluation(env, Q, gamma)
        V, Q, policy_stable = policy_improvement(env, V, Q, gamma)
        meanV_trajectory.extend(meanV)

    np.save(
        os.path.join(save_dir, 'policy_iteration_learning_curve.npy'),
        meanV_trajectory, allow_pickle=False)
    np.save(
        os.path.join(save_dir, 'policy_iteration_Q.npy'),
        Q, allow_pickle=False)
    np.save(
        os.path.join(save_dir, 'policy_iteration_V.npy'),
        V, allow_pickle=False)

    return

def train_value_iteration(env, save_dir, gamma=GAMMA, theta=1e-8, *args, **kwargs):
    # Create environment
    # env = discrete_pendulum.Pendulum(n_theta=21, n_thetadot=21, n_tau=21)
    # env = gridworld.GridWorld(hard_version=False)

    # initializing
    V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))

    V, Q, meanV_trajectory = value_iteration(env, V, Q, gamma, theta)

    np.save(
        os.path.join(SAVEDIR, 'value_iteration_learning_curve.npy'),
        meanV_trajectory, allow_pickle=False)
    np.save(
        os.path.join(SAVEDIR, 'value_iteration_Q.npy'),
        Q, allow_pickle=False)
    np.save(
        os.path.join(SAVEDIR, 'value_iteration_V.npy'),
        V, allow_pickle=False)

    return

def train_sarsa(inputParams):
    # print("Worker id: {}".format(os.getpid()))
    env = inputParams[0]
    alpha = inputParams[1]
    epsilon = inputParams[2]

    # Create environment
    # env = discrete_pendulum.Pendulum(n_theta=21, n_thetadot=21, n_tau=21)
    # env = gridworld.GridWorld(hard_version=False)

    # initializing
    V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))

    # V, Q = value_iteration(env, V, Q, gamma, theta)
    Q, reward_trajectory = sarsa_td0(
        env, Q, num_episodes=NEPISODES, gamma=GAMMA, alpha=alpha,
        epsilon=epsilon)
    V = value_evaluation_td0(
        env, V, Q, num_episodes=NEPISODES, gamma=GAMMA, alpha=alpha)

    np.save(
        os.path.join(SAVEDIR, 'sarsa_alpha{:.2f}_epsilon{:.2f}_Q.npy'.format(alpha, epsilon)),
        Q, allow_pickle=False)
    np.save(
        os.path.join(SAVEDIR, 'sarsa_alpha{:.2f}_epsilon{:.2f}_learning_curve.npy'.format(alpha, epsilon)),
        reward_trajectory, allow_pickle=False)
    np.save(
        os.path.join(SAVEDIR, 'sarsa_alpha{:.2f}_epsilon{:.2f}_V.npy'.format(alpha, epsilon)),
        V, allow_pickle=False)

    # print("working pid {} done".format(os.getpid()))

    return

def train_qlearning(inputParams):
    alpha = inputParams[0]
    epsilon = inputParams[1]

    # Create environment
    # env = discrete_pendulum.Pendulum(n_theta=21, n_thetadot=21, n_tau=21)
    env = gridworld.GridWorld(hard_version=False)

    # initializing
    V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))

    # V, Q = value_iteration(env, V, Q, gamma, theta)
    Q, reward_trajectory = qlearning_td0(
        env, Q, num_episodes=NEPISODES, gamma=GAMMA, alpha=alpha,
        epsilon=epsilon)
    V = value_evaluation_td0(
        env, V, Q, num_episodes=NEPISODES, gamma=GAMMA, alpha=alpha)

    np.save(
        os.path.join(SAVEDIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_Q.npy'.format(alpha, epsilon)),
        Q, allow_pickle=False)
    np.save(
        os.path.join(SAVEDIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_learning_curve.npy'.format(alpha, epsilon)),
        reward_trajectory, allow_pickle=False)
    np.save(
        os.path.join(SAVEDIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_V.npy'.format(alpha, epsilon)),
        V, allow_pickle=False)

    # print("working pid {} done".format(os.getpid()))

    return
