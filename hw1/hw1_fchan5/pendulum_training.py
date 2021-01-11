import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
import os
from tqdm import tqdm
from learning_algorithms import policy_evaluation, policy_improvement
from learning_algorithms import value_iteration
from learning_algorithms import sarsa_td0, qlearning_td0, value_evaluation_td0
import multiprocessing

NUM_EPISODES = 50000
N_THETA = 31
N_THETADOT = 31
N_TAU = 31
SAVE_DIR = 'pendulumData'

NUM_PROCESS = 4

def main():
    print(
        "Training agent using sarsa and q-learning with following parameters:\n\
        NUM_EPISODES = {}\n (n_theta, n_thetedot, n_tau) = ({}, {}, {})\n".format(
            NUM_EPISODES, N_THETA, N_THETADOT, N_TAU))

    # Create environment
    env = discrete_pendulum.Pendulum(n_theta=N_THETA, n_thetadot=N_THETADOT, n_tau=N_TAU)

    # generating sweep for alpha and epsilon
    all_alpha = [0.1, 0.25, 0.5]
    all_epsilon = [0.1, 0.25, 0.5]

    # config contains tuple of (alpha, epsilon, learnValue)
    all_config = []
    # for alpha = 0.1, different epsilon
    for epsilon in all_epsilon:
        all_config.append([env, 0.1, epsilon])
    # for epsilon = 0.1, different alpha
    for alpha in all_alpha:
        if alpha == 0.1:
            continue
        all_config.append([env, alpha, 0.1])
    print(all_config)

    np.save(
        os.path.join(SAVE_DIR, 'alpha_epsilon_exploration.npy'),
        np.array(all_config), allow_pickle=True)

    # sarsa training
    p = multiprocessing.Pool(NUM_PROCESS)
    result = p.map(train_sarsa_parallel, all_config)
    # qlearning training
    p = multiprocessing.Pool(NUM_PROCESS)
    result = p.map(train_qlearning_parallel, all_config)


def train_sarsa_parallel(input_params):
    env = input_params[0]
    alpha = input_params[1]
    epsilon = input_params[2]

    # initializing
    V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))

    # V, Q = value_iteration(env, V, Q, gamma, theta)
    Q, reward_trajectory = sarsa_td0(
        env, Q, num_episodes=NUM_EPISODES, alpha=alpha,
        epsilon=epsilon)
    V = value_evaluation_td0(
        env, V, Q, num_episodes=NUM_EPISODES, alpha=alpha)

    np.save(
        os.path.join(SAVE_DIR, 'sarsa_alpha{:.2f}_epsilon{:.2f}_Q.npy'.format(alpha, epsilon)),
        Q, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'sarsa_alpha{:.2f}_epsilon{:.2f}_learning_curve.npy'.format(alpha, epsilon)),
        reward_trajectory, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'sarsa_alpha{:.2f}_epsilon{:.2f}_V.npy'.format(alpha, epsilon)),
        V, allow_pickle=False)

    return

def train_qlearning_parallel(input_params):
    env = input_params[0]
    alpha = input_params[1]
    epsilon = input_params[2]

    # initializing
    V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))
    gamma = 0.95 # discount factor

    # V, Q = value_iteration(env, V, Q, gamma, theta)
    Q, reward_trajectory = qlearning_td0(
        env, Q, num_episodes=NUM_EPISODES, alpha=alpha,
        epsilon=epsilon)
    V = value_evaluation_td0(
        env, V, Q, num_episodes=NUM_EPISODES, alpha=alpha)

    np.save(
        os.path.join(SAVE_DIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_Q.npy'.format(alpha, epsilon)),
        Q, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_learning_curve.npy'.format(alpha, epsilon)), reward_trajectory, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_V.npy'.format(alpha, epsilon)),
        V, allow_pickle=False)

    return

if __name__ == "__main__":
    main()
