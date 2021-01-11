from mpi4py import MPI
import secrets
import numpy as np
import torch
from actor_critic_model import Actor, Critic
import acrobot_with_target
# import acrobot
import argparse
import os
from tqdm import tqdm

GAMM = 0.99
LAMB = 0.95 # <--- cannot use "lambda" as a variable name in python code!
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
parser.add_argument('--save_dir', default='test_MPI_save')
parser.add_argument('--mini_batch_size', type=int, default=8000)
parser.add_argument('--alpha', type=float, default=0.0001)
parser.add_argument('--epsilon', type=float, default=0.2)
parser.add_argument('--num_iters', type=int, default=10000)
parser.add_argument('--num_eps_per_iter', type=int, default=100)
parser.add_argument('--max_num_steps_per_ep', type=int, default=100)
parser.add_argument('--actor_epochs_per_iter', type=int, default=20)
parser.add_argument('--critic_epochs_per_iter', type=int, default=20)
parser.add_argument('--task', type=str, default='reach')
parser.add_argument('--target_behavior', type=str, default='fixed')
parser.add_argument('--dump_frequency', type=int, default=1) # dump every `dump_frequency` iteration
parser.add_argument('--run_id', type=int, default=-1)
parser.add_argument('--restart', type=str2bool, default=False)
args = parser.parse_args()

def main():
    """
    Train the agent using PPO
    """

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    SAVE_DIR = args.save_dir if (args.run_id == -1) else os.path.join(args.save_dir, 'run{:03d}'.format(args.run_id))

    # Some training settings prints
    if rank == 0:
        print(f'Acrobot tasked to {args.task} with target behavior {args.target_behavior}\n')

        print('Running with parameters:')
        print(f'Total episode / iteration : {args.num_eps_per_iter}')
        print(f'Total batch size : {args.num_eps_per_iter *args.max_num_steps_per_ep}')
        print(f'Total mini batch size : {args.mini_batch_size}')
        print(f'Total optimizer epoch : {args.actor_epochs_per_iter}')
        print('')
        print('With parallel implementation')
        print(f'Number of processes : {size}')
        print(f'Episode / iteration per process : {args.num_eps_per_iter // size}')
        print(f'Batch size per process: {(args.num_eps_per_iter // size)*args.max_num_steps_per_ep}')
        print(f'Mini batch size per process : {args.mini_batch_size // size}')
        print('')

        # Set up save directory
        if not os.path.isdir(SAVE_DIR):
            # os.system("rm -rf {}".format(SAVE_DIR))
            os.makedirs(SAVE_DIR)

    # Initialize environment, etc.
    env = acrobot_with_target.Acrobot(target_behavior=args.target_behavior, task=args.task)

    # Seed all possible random number generators
    l_seed = secrets.randbits(32)
    seed = comm.bcast(l_seed, root=0)
    # random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    rg = np.random.default_rng(seed + rank)
    env.set_rg(rg)

    # Initialize actor and critic
    actor = Actor(env.num_states, env.num_actions)
    critic = Critic(env.num_states, env.num_actions)

    # Initialize data for storing learning curves
    data = {'step': [], 'reward':[], 'losses':[]}

    restart_iter = 0
    # Take care of possible restarts
    if args.restart:
        # Find the last saved checkpoint
        files = os.listdir(SAVE_DIR)
        # Need to take care of possible misaligned snapshots for actor and critic
        fnames_actor = []
        fnames_critic = []
        for f in files:
            if 'actor' in f and not 'trained' in f:
                fnames_actor.append(int(f.split('_')[-1].replace('.pt', '')))
            if 'critic' in f and not 'trained' in f:
                fnames_critic.append(int(f.split('_')[-1].replace('.pt', '')))
        restart_iter = min( max(fnames_actor), max(fnames_critic) )
        # Only master loads the necessary stuff and broadcasts it later
        if rank == 0:
            print(f'Restarting from iteration : {restart_iter}')
            # Load trained actor
            restart_actor = torch.load(os.path.join(SAVE_DIR, 'actor_{:04d}.pt'.format(restart_iter)))
            actor.load_state_dict(restart_actor)
            # Load trained critic
            restart_critic = torch.load(os.path.join(SAVE_DIR, 'critic_{:04d}.pt'.format(restart_iter)))
            critic.load_state_dict(restart_critic)
            # Load data for reward (actor) and loss (critic) curves
            data = np.load(os.path.join(SAVE_DIR, "training_data.npy"),allow_pickle='TRUE').item()

    # Broadcast data dictionary
    data = comm.bcast(data, root=0)
    # Initialize simulation step counter
    step = 0 if not data['step'] else data['step'][-1]

    # Broadcast actor and critic parameters
    actor.broadcast()
    critic.broadcast()
    # Initalize optimizers
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.alpha)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.alpha)

    # num_iters = args.num_iters // size
    num_eps_per_iter_per_process = args.num_eps_per_iter // size
    mini_batch_size_per_process = args.mini_batch_size // size
    # Only master needs to show progress bar
    if rank == 0:
        pbar = tqdm(total = args.num_iters, initial = restart_iter)
    for it in range(restart_iter, args.num_iters):
        # Get a batch of trajectories
        batch = get_batch(env, num_eps_per_iter_per_process, args.max_num_steps_per_ep, actor, critic)
        # Normalize advantage estimates
        with torch.no_grad():
            adv_mean = allreduce_average(torch.mean(batch['w']))
            adv_std = torch.sqrt(allreduce_average(torch.var(batch['w'])))
            batch['w'] = (batch['w'] - adv_mean) / adv_std

        # Actor optimization
        for epoch in range(args.actor_epochs_per_iter):
            sample_idx = np.random.choice(range(batch['r'].shape[0]), mini_batch_size_per_process, replace=False)
            optimizer_actor.zero_grad()
            # Compute surrogate loss
            (mu, std) = actor(batch['s'][sample_idx])
            dist = torch.distributions.normal.Normal(mu, std)
            log_pi = dist.log_prob(batch['a'][sample_idx]).sum(axis=-1)
            ratio = torch.exp(log_pi - batch['log_pi'][sample_idx].sum(axis=-1))
            loss = ratio * batch['w'][sample_idx]
            ratio_clipped = torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon)
            loss_clipped = ratio_clipped * batch['w'][sample_idx]
            loss = -torch.mean(torch.min(loss, loss_clipped))
            loss.backward()
            # average gradients among processes
            actor.allreduce_average_gradients()
            # optimizer step only on master
            if rank == 0:
                # Optimize using surrogate loss
                optimizer_actor.step()
            # broadcast actor
            actor.broadcast()

        # Critic optimization
        losses = []
        for epoch in range(args.critic_epochs_per_iter):
            optimizer_critic.zero_grad()
            # Compute critic loss
            V = critic(batch['s'])
            # loss = torch.nn.MSELoss()(V.squeeze(1), batch['V_target'])
            loss = torch.nn.MSELoss()(V[:,0], batch['V_target'])
            with torch.no_grad():
                losses.append(allreduce_average(loss).item())
            loss.backward()
            # average gradients among processes
            critic.allreduce_average_gradients()
            # optimizer step only on master
            if rank == 0:
                optimizer_critic.step()
            # broadcast critic
            critic.broadcast()

        data['losses'].append(np.mean(losses))
        reward_mean = allreduce_average(batch['r'].mean())
        data['reward'].append(reward_mean)# * N)

        # sum all the steps taken by all processes
        local_step = batch['r'].shape[0]
        global_step = 0
        global_step = comm.allreduce(local_step, op=MPI.SUM)
        step += global_step
        data['step'].append(step)

        # save actor and critic network every few iterations
        if it % args.dump_frequency == 0 and rank == 0:
            torch.save(actor.state_dict(), os.path.join(SAVE_DIR, 'actor_{:04d}.pt'.format(it)))
            torch.save(critic.state_dict(), os.path.join(SAVE_DIR, 'critic_{:04d}.pt'.format(it)))
            # np.save(os.path.join(SAVE_DIR, 'training_data_{:04d}.npy'.format(it)), data)
            np.save(os.path.join(SAVE_DIR, 'training_data.npy'), data)

        # update tqdm progress bar on master
        if rank == 0:
            pbar.update(1)

    # Save final training stuff
    if rank == 0:
        torch.save(actor.state_dict(), os.path.join(SAVE_DIR, 'actor_trained.pt'))
        torch.save(critic.state_dict(), os.path.join(SAVE_DIR, 'critic_trained.pt'))
        np.save(os.path.join(SAVE_DIR, 'training_data.npy'), data)

        print("Completed run #{} for case of {}".format(args.run_id, args.save_dir))


def get_batch(env, num_eps, num_steps, actor, critic):
    """
    Generate a batch of trajectories
    """
    with torch.no_grad():
        batch = {'s': [], 'a': [], 'r': [], 'w': [], 'V_target': [], 'log_pi': []}
        # for ep in range(num_eps):
        while len(batch['r']) < num_eps * num_steps:
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
                traj['log_pi'].append(dist.log_prob(torch.tensor(a)).numpy())
                s = s_prime
                if done:
                    break

            # Get advantages and value targets
            # - get length of trajectory
            T = len(traj['r'])
            # - create copy of r and V, appended with zero (could append with bootstrap)
            r = np.append(traj['r'], 0.)
            V = np.append(traj['V'], critic(torch.from_numpy(s)).item())
            # - compute deltas
            delta = r[:-1] + (GAMM * V[1:]) - V[:-1]
            # - compute advantages as reversed, discounted, cumulative sum of deltas
            A = delta.copy()
            for t in reversed(range(T - 1)):
                A[t] = A[t] + (GAMM * LAMB * A[t + 1])
            # - get value targets
            for t in reversed(range(T)):
                V[t] = r[t] + (GAMM * V[t + 1])
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

def allreduce_average(local_tensor):
    """
    Average a torch tensor across all processes and send the result to all processes
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    local_array = local_tensor.detach().numpy()
    global_array = np.zeros_like(local_array)
    comm.Allreduce(local_array, global_array, op=MPI.SUM)
    global_array /= size
    global_tensor = torch.tensor(global_array)
    return global_tensor


if __name__ == "__main__":
    main()