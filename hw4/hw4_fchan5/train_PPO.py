import numpy as np
import torch
from actor_critic_model import Actor, Critic
import pendulum
import argparse
import os
from tqdm import tqdm

gamm = 0.99
lamb = 0.95 # <--- cannot use "lambda" as a variable name in python code!
NUM_PROCESSES = 4

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='test_save')
parser.add_argument('--mini_batch_size', type=int, default=2500)
parser.add_argument('--alpha', type=float, default=0.001)
parser.add_argument('--epsilon', type=float, default=0.5)
parser.add_argument('--num_iters', type=int, default=2000)
parser.add_argument('--num_eps_per_iter', type=int, default=50)
parser.add_argument('--max_num_steps_per_ep', type=int, default=100)
parser.add_argument('--actor_epochs_per_iter', type=int, default=25)
parser.add_argument('--critic_epochs_per_iter', type=int, default=25)
parser.add_argument('--sparse_reward', type=str2bool, default=False)
parser.add_argument('--dump_frequency', type=int, default=1)
parser.add_argument('--run_id', type=int, default=-1)
args = parser.parse_args()

def main():
    """
    Train the agent using PPO
    """

    # print('Running with parameters:')
    # print(f'episode / iteration : {args.num_eps_per_iter}')
    # print(f'mini batch size : {args.mini_batch_size}')
    # print(f'optimizer epoch : {args.actor_epochs_per_iter}')

    # Set up save directory
    SAVE_DIR = args.save_dir if (args.run_id == -1) else os.path.join(args.save_dir, 'run{:03d}'.format(args.run_id))
    if not os.path.isdir(SAVE_DIR):
        # os.system("rm -rf {}".format(SAVE_DIR))
        os.makedirs(SAVE_DIR)

    # Initialize environment, etc.
    env = pendulum.Pendulum(sparse_reward=args.sparse_reward)
    rg = np.random.default_rng()

    # Initialize actor related stuff
    actor = Actor(env.num_states, env.num_actions)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.alpha)

    # Initialize critic related stuff
    critic = Critic(env.num_states, env.num_actions)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.alpha)

    step = 0
    data = {'step': [], 'reward':[], 'losses':[]}

    # for it in range(args.num_iters):
    for it in tqdm(
        range(args.num_iters),
        desc="Run #{} on {}".format(args.run_id, os.getpid()),
        position=args.run_id % NUM_PROCESSES):
        # Get a batch of trajectories
        batch = get_batch(env, args.num_eps_per_iter, args.max_num_steps_per_ep, actor, critic)

        # Actor optimization
        for epoch in range(args.actor_epochs_per_iter):
            sample_idx = np.random.choice(range(batch['r'].shape[0]), args.mini_batch_size, replace=False)
            optimizer_actor.zero_grad()
            # Compute surrogate loss
            (mu, std) = actor(batch['s'][sample_idx])
            dist = torch.distributions.normal.Normal(mu, std)
            log_pi = dist.log_prob(batch['a'][sample_idx]).sum(axis=-1)
            ratio = torch.exp(log_pi - batch['log_pi'][sample_idx])
            loss = ratio * batch['w'][sample_idx]
            ratio_clipped = torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon)
            loss_clipped = ratio_clipped * batch['w'][sample_idx]
            loss = -torch.mean(torch.min(loss, loss_clipped))
            loss.backward()
            # Optimize using surrogate loss
            optimizer_actor.step()

        # Critic optimization
        for epoch in range(args.critic_epochs_per_iter):
            optimizer_critic.zero_grad()
            # Compute critic loss
            V = critic(batch['s'])
            loss = torch.nn.MSELoss()(V.squeeze(1), batch['V_target'])
            loss.backward()
            optimizer_critic.step()
        data['losses'].append(loss.item())
        data['reward'].append(batch['r'].mean())# * N)
        step += batch['r'].shape[0]
        data['step'].append(step)

        # save actor and critic network every few iterations
        if it % args.dump_frequency == 0:
            torch.save(actor.state_dict(), os.path.join(SAVE_DIR, 'actor_{:04d}.pt'.format(it)))
            torch.save(critic.state_dict(), os.path.join(SAVE_DIR, 'critic_{:04d}.pt'.format(it)))
            # np.save(os.path.join(SAVE_DIR, 'training_data_{:04d}.npy'.format(it)), data)
            np.save(os.path.join(SAVE_DIR, 'training_data.npy'), data)


    # Save final training stuff
    torch.save(actor.state_dict(), os.path.join(SAVE_DIR, 'actor_trained.pt'))
    torch.save(critic.state_dict(), os.path.join(SAVE_DIR, 'critic_trained.pt'))
    np.save(os.path.join(SAVE_DIR, 'training_data.npy'), data)

    print("Completed run #{} for case of {}".format(args.run_id, args.save_dir))


def get_batch(env, num_eps, num_steps, actor, critic):
    with torch.no_grad():
        batch = {'s': [], 'a': [], 'r': [], 'w': [], 'V_target': [], 'log_pi': []}
        for ep in range(num_eps):
            traj = {'s': [], 'a': [], 'r': [], 'V': [], 'log_pi': []}
            done = False
            s = env.reset()
            for step in range(num_steps):
                (mu, std) = actor(torch.from_numpy(s))
                dist = torch.distributions.normal.Normal(mu, std)
                a = dist.sample().numpy()
                s_prime, r, done = env.step(a)
                traj['s'].append(s)
                traj['a'].append(a)
                traj['r'].append(r)
                traj['V'].append(critic(torch.from_numpy(s)).item())
                traj['log_pi'].append(dist.log_prob(torch.tensor(a)))
                s = s_prime
                if done:
                    break

            # Get advantages and value targets
            # - get length of trajectory
            T = len(traj['r'])
            # - create copy of r and V, appended with zero (could append with bootstrap)
            r = np.append(traj['r'], 0.)
            V = np.append(traj['V'], 0.)
            # - compute deltas
            delta = r[:-1] + (gamm * V[1:]) - V[:-1]
            # - compute advantages as reversed, discounted, cumulative sum of deltas
            A = delta.copy()
            for t in reversed(range(T - 1)):
                A[t] = A[t] + (gamm * lamb * A[t + 1])
            # - get value targets
            for t in reversed(range(T)):
                V[t] = r[t] + (gamm * V[t + 1])
            V = V[:-1]

            batch['s'].extend(traj['s'])
            batch['a'].extend(traj['a'])
            batch['r'].extend(traj['r'])
            batch['w'].extend(A)
            batch['V_target'].extend(V)
            batch['log_pi'].extend(traj['log_pi'])
        batch['num_steps'] = len(batch['r'])
        batch['s'] = torch.tensor(batch['s'], requires_grad=False, dtype=torch.double)
        batch['a'] = torch.tensor(batch['a'], requires_grad=False, dtype=torch.double)
        batch['r'] = torch.tensor(batch['r'], requires_grad=False, dtype=torch.double)
        batch['w'] = torch.tensor(batch['w'], requires_grad=False, dtype=torch.double)
        batch['V_target'] = torch.tensor(batch['V_target'], requires_grad=False, dtype=torch.double)
        batch['log_pi'] = torch.tensor(batch['log_pi'], requires_grad=False, dtype=torch.double)
    return batch


if __name__ == "__main__":
    main()