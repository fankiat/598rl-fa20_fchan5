import numpy as np
import torch
import discreteaction_pendulum
from QNet import QNet
import random
import os
from tqdm import tqdm
from collections import deque
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='test_save')
parser.add_argument('--target_reset_frequency', type=int, default=250)
parser.add_argument('--replay_memory_buffer_size', type=int, default=10000)
parser.add_argument('--mini_batch_size', type=int, default=25)
parser.add_argument('--num_training_episodes', type=int, default=2000)
parser.add_argument('--num_training_runs', type=int, default=1)
parser.add_argument('--process_id', type=int, default=0)
parser.add_argument('--dump_qnet_frequency', type=int, default=50000)
args = parser.parse_args()

NUM_INITIAL_EXPLORE = 10000
GAMMA = 0.95
SAVE_DIR = args.save_dir
EPSILON = 0.1
OPTIMIZER_LR = 0.0025

def train_network(Q_net, Q_net_target, mini_batch, optimizer):
    """
    Train Q network
    """
    # decipher and restructure replay memory mini batch for learning
    # each memory element is in the format of (curr_state, curr_action, reward, next_state)
    curr_states_mb = np.stack(mini_batch[:,0])
    curr_actions_mb = np.stack(mini_batch[:,1])
    rewards_mb = np.stack(mini_batch[:,2])
    next_states_mb = np.stack(mini_batch[:,3])

    # convert to torch tensors
    curr_states_mb = torch.Tensor(curr_states_mb)
    curr_actions_mb = torch.LongTensor(curr_actions_mb)
    rewards_mb = torch.Tensor(rewards_mb)
    next_states_mb = torch.Tensor(next_states_mb)

    # compute Q values
    Q = Q_net(curr_states_mb)
    curr_actions_mb_idx = curr_actions_mb.unsqueeze(1) # making the idx array of size (mini_batch_size, 1)
    # gather the q values for the taken action and reshape it as size of mini_batch_size
    Q = Q.gather(1, curr_actions_mb_idx).view(-1)

    with torch.no_grad():
        Q_target = Q_net_target(next_states_mb)
        target_y = rewards_mb + GAMMA * Q_target.max(1)[0]

    loss_criterion = torch.nn.MSELoss()
    loss = loss_criterion(Q, target_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().numpy()

def choose_action_epsilon_greedy(Q, num_actions, epsilon):
    """
    Returns action based on epsilon greedy method
    """
    if np.random.uniform(0,1) <= epsilon:
        action = random.randrange(num_actions)
    else:
#         print(Q)
#         Qmax, Qmax_idx = torch.max(Q)
#         action = Qmax_idx.numpy()[0]
        action = np.argmax(Q.detach().numpy())
    return action

def main():
    # Initialize environment
    env = discreteaction_pendulum.Pendulum()

    # for some reason this makes the code run faster
    # Did not investigate this too much as to why this happens
    torch.set_num_threads(1)

    # create save directory
    if os.path.isdir(SAVE_DIR):
        os.system("rm -rf {}".format(SAVE_DIR))
    os.makedirs(SAVE_DIR)

    # perform multiple runs for averaging of results
    for curr_run in range(args.num_training_runs):
        # initialize Q network, optimizer, replay memory buffer
        Q_net = QNet(env.num_states, env.num_actions)
        Q_net_target = QNet(env.num_states, env.num_actions)
        Q_net_target.load_state_dict(Q_net.state_dict()) # make a copy
        optimizer = torch.optim.Adam(Q_net.parameters(), lr=OPTIMIZER_LR)
        replay_memory = deque(maxlen=args.replay_memory_buffer_size)

        epsilon = EPSILON

        # Create dict to store training log
        training_steps = 0
        log_reward = []
        log_reward.append(0)
        log_loss = []
        log_loss.append(0) # zero loss cos we know nothing yet at step 0?

        # for curr_episode in range(args.num_training_episodes):
        for curr_episode in tqdm(
            range(args.num_training_episodes),
            desc="Run #{} on {}".format(curr_run+1, os.getpid()),
            position=args.process_id+1):
            curr_state = env.reset()
            done = False
            total_discounted_reward = 0
            total_loss = 0
            while not done:
                training_steps += 1
                with torch.no_grad():
                    Q = Q_net(torch.Tensor(curr_state))
                curr_action = choose_action_epsilon_greedy(Q, env.num_actions, epsilon)
                next_state, reward, done = env.step(curr_action)

                replay_memory.append((curr_state, curr_action, reward, next_state))

                curr_state = next_state

                total_discounted_reward += (GAMMA**(env.num_steps-1)) * reward

                # explore fully for first num_initial_explore steps to fill up replay memory buffer, then train Q network
                if training_steps > NUM_INITIAL_EXPLORE:
                    # sample mini batch from replay memory
                    mini_batch = np.array(random.sample(replay_memory, args.mini_batch_size))
                    loss = train_network(Q_net, Q_net_target, mini_batch, optimizer)
                    total_loss += loss

                    # update target network every C (target_reset_frequency) steps
                    if training_steps % args.target_reset_frequency == 0:
                        Q_net_target.load_state_dict(Q_net.state_dict()) # make a copy

                # save the network every dump_qnet_frequency steps for qnet trajectory
                if training_steps % args.dump_qnet_frequency == 0:
                    torch.save(Q_net.state_dict(), os.path.join(SAVE_DIR, 'Q_network_{:05d}.pt'.format(training_steps)))

            log_reward.append(total_discounted_reward)
            log_loss.append(total_loss)
        # save the total discounted reward and loss trajectory
        np.save(
            os.path.join(
                SAVE_DIR, 'reward_trajectory_run{}.npy'.format(curr_run)),
            np.array(log_reward), allow_pickle=False)
        np.save(
            os.path.join(
                SAVE_DIR, 'loss_trajectory_run{}.npy'.format(curr_run)),
            np.array(log_loss), allow_pickle=False)

    # save the trained network (only the last run---not sure how to average Q-networks)
    torch.save(Q_net.state_dict(), os.path.join(SAVE_DIR, 'Q_network.pt'))

    print("Completed {} runs for case of {}".format(args.num_training_runs, args.save_dir))

if __name__ == "__main__":
    # time the training for estimation
    # start = time.time()
    main()
    # end = time.time()
    # print(f"Runtime of the program is {end - start}")
