import random, os
import numpy as np
from tqdm import tqdm

GAMMA = 0.95

def policy_evaluation(env, V, Q, gamma=GAMMA):
    # initializing
    # V = np.zeros(env.num_states)
    meanV = []
    meanV.append(np.average(V))
    while True:
        delta = 0
        for curr_state in range(env.num_states):
            old_v = V[curr_state]
            update = 0
            curr_action = np.argmax(Q[curr_state])
            for next_state in range(env.num_states):
                update += env.p(next_state, curr_state, curr_action) * (env.r(curr_state, curr_action) + gamma * V[next_state])
            V[curr_state] = update
            delta = max(delta, abs(old_v - V[curr_state]))
        meanV.append(np.average(V))

        if delta <= 1e-8:
            break

    return V, Q, meanV


def policy_improvement(env, V, Q, gamma=GAMMA):
    policy_stable = True
    for curr_state in range(env.num_states):
        old_best_action = np.argmax(Q[curr_state]) # old pi
        for curr_action in range(env.num_actions):
            update = 0
            for next_state in range(env.num_states):
                update += env.p(next_state, curr_state, curr_action) * (env.r(curr_state, curr_action) + gamma * V[next_state])
            # greedy update?
            Q[curr_state, curr_action] = update
        new_best_action = np.argmax(Q[curr_state]) # new pi

        if old_best_action != new_best_action:
            policy_stable = False

    return V, Q, policy_stable

def policy_iteration(env, gamma=GAMMA):
    # initializing
    V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))

    policy_stable = False
    meanV_trajectory = []
    while not policy_stable:
        V, Q, meanV = policy_evaluation(env, V, Q, gamma)
        V, Q, policy_stable = policy_improvement(env, V, Q, gamma)
        meanV_trajectory.extend(meanV)

    return V, Q, meanV_trajectory

def value_iteration(env, gamma=GAMMA, theta=1e-5):
    # initializing
    V = np.zeros(env.num_states)
    # equal probability of taking any action
    Q = np.ones((env.num_states, env.num_actions)) / env.num_actions
    # Q = np.zeros((env.num_states, env.num_actions))

    meanV_trajectory = []
    meanV_trajectory.append(np.average(V))
    while True:
        delta = 0
        for curr_state in range(env.num_states):
            v = np.zeros(env.num_actions)
            for curr_action in range(env.num_actions):
                for next_state in range(env.num_states):
                    v[curr_action] += env.p(next_state, curr_state, curr_action) \
                                    * (env.r(curr_state, curr_action) + gamma * V[next_state])
            # update the Q table here right away?
            # we're gonna take argmax(Q[s]) later when deciding action anyway
            Q[curr_state] = v

            # pick the best value
            best_v = np.max(v)
            delta = max(delta, abs(best_v - V[curr_state]))
            # update V
            V[curr_state] = best_v
        meanV_trajectory.append(np.average(V))

        if delta < theta:
            break

    return V, Q, meanV_trajectory


def sarsa_td0(env, Q, num_runs=1, num_episodes=100, gamma=0.95, alpha=0.85, epsilon=0.1):

    def choose_action(state):
        if np.random.uniform(0,1) < epsilon:
            action = random.randrange(env.num_actions)
        else:
            action = np.argmax(Q[state])
        return action

    reward_trajectory = np.zeros(num_episodes)
    reward_trajectory_averaged = np.zeros(num_episodes)

    # start sarsa
    for curr_run in range(num_runs):
        # for curr_episode in range(num_episodes):
        for curr_episode in tqdm(range(num_episodes), desc="#{:02d} {:<12} alpha={:.2f}, epsilon={:.2f}".format(os.getpid(), '(sarsa td0)', alpha, epsilon)):
            curr_state = env.reset()
            # choose action
            curr_action = choose_action(curr_state)

            total_reward = 0
            done = False
            while not done:
                (next_state, reward, done) = env.step(curr_action)

                # choose next action
                next_action = choose_action(next_state)

                # learn and update Q
                Q[curr_state, curr_action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[curr_state, curr_action])

                curr_state = next_state
                curr_action = next_action

                total_reward += (gamma**env.num_steps) * reward

            reward_trajectory[curr_episode] = total_reward
        reward_trajectory_averaged += reward_trajectory
    reward_trajectory_averaged /= num_runs

    return Q, reward_trajectory_averaged

def value_evaluation_td0(env, V, Q, num_episodes=100, gamma=0.95, alpha=0.85):
    # for curr_episode in range(num_episodes):
    for curr_episode in tqdm(range(num_episodes), desc="#{} {:<12} alpha={:.2f}{:<14}".format(os.getpid(), '(value td0)', alpha, '')):
        curr_state = env.reset()

        done = False
        while not done:
            # choose action from Q
            curr_action = np.argmax(Q[curr_state])
            (next_state, reward, done) = env.step(curr_action)
            V[curr_state] += alpha * (reward + gamma * V[next_state] - V[curr_state])
            curr_state = next_state

    return V

def qlearning_td0(env, Q, num_runs=1, num_episodes=100, gamma=0.95, alpha=0.85, epsilon=0.9):
    # defining choose action with epsilon-greedy scheme
    def choose_action(state):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, env.num_actions)
        else:
            action = np.argmax(Q[state])
        return action

    reward_trajectory = np.zeros(num_episodes)
    reward_trajectory_averaged = np.zeros(num_episodes)
    # start sarsa
    for curr_run in range(num_runs):
        # for curr_episode in range(num_episodes):
        for curr_episode in tqdm(range(num_episodes), desc="#{:02d} {:<12} alpha={:.2f}{:<14}".format(os.getpid(), '(qlearning)', alpha, '')):
            curr_state = env.reset()

            total_reward = 0
            done = False
            while not done:
                # choose action
                curr_action = choose_action(curr_state)
                # take action
                (next_state, reward, done) = env.step(curr_action)
                # learn and update Q
                Q[curr_state, curr_action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[curr_state, curr_action])
                # update state
                curr_state = next_state

                total_reward += (gamma**env.num_steps) * reward
                # total_reward += reward

            reward_trajectory[curr_episode] = total_reward
        reward_trajectory_averaged += reward_trajectory
    reward_trajectory_averaged /= num_runs

    return Q, reward_trajectory
