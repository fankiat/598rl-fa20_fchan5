import numpy as np
import torch
import gridworld
import random
import os
from tqdm import tqdm
# from collections import deque
# import time
import matplotlib.pyplot as plt
import argparse

NUM_PROCESSES = 8

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='test_save')
parser.add_argument('--baseline', type=str2bool, default=False)
parser.add_argument('--causality', type=str2bool, default=False)
parser.add_argument('--importance_sampling', type=str2bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_rollouts', type=int, default=200)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--hard_version', type=str2bool, default=False)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--run_id', type=int, default=0)
args = parser.parse_args()

def main():

    # print(f'case : {args.save_dir}')
    # print(f'baseline: {args.baseline}')
    # print(f'causality: {args.causality}')
    # print(f'imp sampling: {args.importance_sampling}')
    # print(f'optimizer: {args.optimizer}')
    # print(f'alpha : {args.alpha}')

    SAVE_DIR = args.save_dir
    # create save directory
    if not os.path.isdir(SAVE_DIR):
        # os.system("rm -rf {}".format(SAVE_DIR))
        os.makedirs(SAVE_DIR)

    if args.save_dir == 'reinforce':
        run_case = 1
    elif args.save_dir == 'reinforce_baseline_causality':
        run_case = 2
    elif args.save_dir == 'reinforce_baseline_importance_sampling':
        run_case = 3
    elif args.save_dir == 'reinforce_causality_importance_sampling':
        run_case = 4
    elif args.save_dir == 'reinforce_baseline_causality_importance_sampling':
        run_case = 5
    elif args.save_dir == 'test_save':
        run_case = 0
    else:
        raise Exception("Unrecognized run case")

    # INITIALIZE
    rg = np.random.default_rng()
    env = gridworld.GridWorld(hard_version=args.hard_version)

    theta = torch.tensor(rg.standard_normal(size=(env.num_states, env.num_actions)), requires_grad=True)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([theta], lr=args.alpha)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam([theta], lr=args.alpha)
    else:
        raise Exception("Optimizer not specified!")

    data = {
        'step': [],
        'total_reward': [],
    }
    step = 0

    # BEGIN TRAINING
    for outer_iter in tqdm(
        range(args.num_rollouts),
        desc="Case {}, run #{} on {}".format(run_case, args.run_id, os.getpid()),
        position=(run_case + args.run_id*5) % NUM_PROCESSES):
    # for outer_iter in range(args.num_rollouts):
        with torch.no_grad():
            batch_s = []
            batch_a = []
            batch_w = []
            batch_log_pi = []
            r_total_average = 0
            # compute batch_size number of trajectories
            for inner_iter in range(args.batch_size):
                traj_s = []
                traj_a = []
                traj_r = []
                traj_log_pi = []
                s = env.reset()
                while True:
                    logits = theta[s]
                    dist = torch.distributions.categorical.Categorical(logits=logits)
                    a = dist.sample().item()
                    s_prime, r, done = env.step(a)
                    traj_s.append(s)
                    traj_a.append(a)
                    traj_r.append(r)
                    traj_log_pi.append(action_log_probability(a, s, theta.detach().numpy()))
                    s = s_prime
                    if done:
                        break
                batch_s.extend(traj_s)
                batch_a.extend(traj_a)
                # batch_w.extend([np.sum(traj_r)] * len(traj_r))
                batch_w.extend([np.sum(traj_r[int(args.causality*ri):]) for ri in range(len(traj_r))])
                batch_log_pi.extend(traj_log_pi)
                r_total_average += np.sum(traj_r)
            r_total_average /= args.batch_size

        step += args.batch_size * env.max_num_steps
        data['step'].append(step)
        data['total_reward'].append(r_total_average)
        # if (outer_iter + 1) % 10 == 0:
        #     print(f'{step} : {r_total_average}')

        # Convert each batch from list to tensor
        batch_s = torch.tensor(batch_s, requires_grad=False)
        batch_a = torch.tensor(batch_a, requires_grad=False)
        batch_w = torch.tensor(batch_w, requires_grad=False)
        batch_log_pi = torch.tensor(batch_log_pi, requires_grad=False)

        # number of times to take advantage of offline policy training
        # hardcoded to take 5 times for now
        num_offline_train = 1 + args.importance_sampling * 4

        for epoch in range(num_offline_train):
            # Update weights
            optimizer.zero_grad()
            logits = theta[batch_s]
            dist = torch.distributions.categorical.Categorical(logits=logits)
            # log_pi = dist.log_prob(batch_a)
            with torch.no_grad():
                log_pi = dist.log_prob(batch_a)
                ratio = torch.exp(log_pi - batch_log_pi)
            loss = -(env.max_num_steps * dist.log_prob(batch_a) \
                      * (batch_w - args.baseline * np.mean(batch_w.numpy())) \
                      * (args.importance_sampling * ratio + (not args.importance_sampling)*1.)).mean()
            loss.backward()
            optimizer.step()

        # print(f'\ntheta = {theta.detach().numpy().tolist()}\n')
        # print(policy_as_string(env, theta))
        # plot_learning_curve(data, args.save_dir.replace('_', ' '))

    # SAVE DATA
    np.save(os.path.join(SAVE_DIR, 'theta_run{:02d}.npy'.format(args.run_id)), theta.detach().numpy())
    learning_curve = np.vstack((np.array(data['step']), np.array(data['total_reward']))).T
    np.save(os.path.join(SAVE_DIR, 'learning_curve_run{:02d}.npy'.format(args.run_id)), learning_curve)

    print("Completed run #{} for case of {}".format(args.run_id, args.save_dir))


def action_log_probability(a, s, theta):
    return theta[s, a] - np.log(np.exp(theta[s]).sum())

def action_greedy(s, theta):
    return theta[s].argmax()

def policy_as_string(env, theta):
    output = 'greedy policy:\n'
    for s in range(env.num_states):
        a = action_greedy(s, theta)
        if a == 0:
            direction = "right"
        elif a == 1:
            direction = "up"
        elif a == 2:
            direction = "left"
        elif a == 3:
            direction = "down"
        output += f' pi({s}) = {direction}\n'
    return output

def plot_learning_curve(data, label):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(data['step'], data['total_reward'], linewidth=2, label=label)
    ax.grid()
#     ax.set_ylim(0, 20)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('simulation steps', fontsize=20)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()