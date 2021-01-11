import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import os
from learning_algorithms import policy_iteration
from learning_algorithms import value_iteration
from learning_algorithms import sarsa_td0, qlearning_td0, value_evaluation_td0
import multiprocessing
from train import train_policy_iteration

NUM_EPISODES = 2000
SAVE_DIR = 'gridworldData'

NUM_PROCESS = 4

def main():
    env = gridworld.GridWorld(hard_version=False)

    # POLICY ITERATION TRAINING
    V, Q, meanV_trajectory = policy_iteration(env)
    # save V, Q, meaveV_trajectory for post processing
    np.save(
        os.path.join(SAVE_DIR, 'policy_iteration_V.npy'),
        V, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'policy_iteration_Q.npy'),
        Q, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'policy_iteration_learning_curve.npy'),
        meanV_trajectory, allow_pickle=False)

    # VALUE ITERATION TRAINING
    V, Q, meanV_trajectory = value_iteration(env)
    # save V, Q, meaveV_trajectory for post processing
    np.save(
        os.path.join(SAVE_DIR, 'value_iteration_V.npy'),
        V, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'value_iteration_Q.npy'),
        Q, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'value_iteration_learning_curve.npy'),
        meanV_trajectory, allow_pickle=False)

    # SARSA & Q-LEARNING TRAINING FOR DIFFERENT ALPHA AND EPSILON
    # generating sweep for alpha and epsilon
    all_alpha = [0.05, 0.1, 0.25, 0.5]
    all_epsilon = [0.05, 0.1, 0.25, 0.5]

    # config contains tuple of (env, alpha, epsilon)
    all_config = []
    # for alpha = 0.1, different epsilon
    for epsilon in all_epsilon:
        all_config.append([env, 0.1, epsilon])
    # for epsilon = 0.1, different alpha
    for alpha in all_alpha:
        if alpha == 0.1:
            continue
        all_config.append([env, alpha, 0.1])
    # print(all_config)

    np.save(
        os.path.join(SAVE_DIR, 'alpha_epsilon_exploration.npy'),
        np.array(all_config), allow_pickle=True)

    # sarsa training
    p = multiprocessing.Pool(NUM_PROCESS)
    result = p.map(train_sarsa_parallel, all_config)

    # qlearning training
    p = multiprocessing.Pool(NUM_PROCESS)
    result = p.map(train_qlearning_parallel, all_config)

    # fig = plt.figure()
    # for i, V in enumerate(result):
    #     plt.plot(np.mean(V.reshape(-1,10), axis=1), label="alpha={}, epsilon={}".format(all_config[i][0], all_config[i][1]))
    #     # plt.plot(V, label="alpha={}, epsilon={}".format(all_config[i][0], all_config[i][1]))

    #     # plt.contourf(np.flip(V.reshape(5,5), axis=0), levels=100)
    #     # plt.colorbar()
    #     # plt.title("alpha={}, epsilon={}".format(all_config[i][0], all_config[i][1]))
    # plt.legend()
    # plt.show()

def train_sarsa_parallel(input_params):
    # print("Worker id: {}".format(os.getpid()))
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
        os.path.join(SAVE_DIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_learning_curve.npy'.format(alpha, epsilon)),
        reward_trajectory, allow_pickle=False)
    np.save(
        os.path.join(SAVE_DIR, 'qlearning_alpha{:.2f}_epsilon{:.2f}_V.npy'.format(alpha, epsilon)),
        V, allow_pickle=False)

    return


if __name__ == "__main__":
    main()