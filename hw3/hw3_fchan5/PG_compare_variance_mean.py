import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
import gridworld
from tqdm import tqdm
import os

rg = np.random.default_rng()
env = gridworld.GridWorld(hard_version=False)

def get_batch_torch(env, N, theta):
    with torch.no_grad():
        batch = {'s': [], 'a': [], 'w': [], 'wc': [], 'log_pi': []}

        for i in range(N):
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
                traj_log_pi.append(action_log_probability(a, s, theta.numpy()))
                s = s_prime
                if done:
                    break
            batch['s'].extend(traj_s)
            batch['a'].extend(traj_a)
            batch['w'].extend([np.sum(traj_r)] * len(traj_r))
            batch['wc'].extend([np.sum(traj_r[ri:]) for ri in range(len(traj_r))])
            batch['log_pi'].extend(traj_log_pi)

        return batch

def get_gradient(batch, theta, baseline, causality, importance_sampling):
    if causality:
        b = np.mean(batch['wc'])
        weights = batch['wc']
    else:
        b = np.mean(batch['w'])
        weights = batch['w']
    grad = []
#     for (s, a, w, old_log_pi) in zip(batch['s'], batch['a'], batch['w'], batch['log_pi']):
    for (s, a, w, old_log_pi) in tqdm(
        zip(batch['s'], batch['a'], weights, batch['log_pi']),
        desc=f'Compute gradient #{get_gradient.count}', total=len(batch['s'])):
        log_pi = action_log_probability(a, s, theta)
        ratio = np.exp(log_pi - old_log_pi)
        grad.append(
            (action_log_probability_gradient(a, s, theta) \
            * (w - baseline * b) \
            * (importance_sampling * ratio + (not importance_sampling)*1.)
            ).reshape(-1)
        )
    return np.array(grad).T

def action_log_probability(a, s, theta):
    return theta[s, a] - np.log(np.exp(theta[s]).sum())

def action_log_probability_gradient(a, s, theta):
    theta_t = torch.tensor(theta, requires_grad=True)
    p_t = theta_t[s, a] - torch.exp(theta_t[s, :]).sum().log()
    p_t.backward()
    return theta_t.grad.numpy()

def main():

    SAVE_DIR = 'comparison_mean_variance/'
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    # Number of runs
    M = 10

    # Number of episodes (in a batch) per run
    N = 5_000

    # The policy used for simulation
    # old_theta = np.zeros((env.num_states, env.num_actions))
    old_theta = rg.standard_normal(size=(env.num_states, env.num_actions))
    # The policy used for gradient estimation
    delta = 1.0
    theta = old_theta + delta * rg.standard_normal(size=(env.num_states, env.num_actions))

    # theta = rg.standard_normal(size=(env.num_states, env.num_actions))
    # theta = np.zeros((env.num_states, env.num_actions))

    mu_no = []
    mu_no_IS = []
    mu_baseline = []
    mu_causality = []
    mu_importance_sampling = []

    std_no = []
    std_no_IS = []
    std_baseline = []
    std_causality = []
    std_importance_sampling = []

    # for it in range(M):
    for it in tqdm(range(M), desc='outer loop'):
        batch = get_batch_torch(env, N, torch.tensor(old_theta))
        get_gradient.count = it

        grad_no = get_gradient(batch, old_theta, baseline=False, causality=False, importance_sampling=False)
        mu_no.append(np.mean(grad_no * env.max_num_steps, axis=1))
        std_no.append(np.std(grad_no * env.max_num_steps, axis=1))

        grad_baseline = get_gradient(batch, old_theta, baseline=True, causality=False, importance_sampling=False)
        mu_baseline.append(np.mean(grad_baseline * env.max_num_steps, axis=1))
        std_baseline.append(np.std(grad_baseline * env.max_num_steps, axis=1))

        grad_causality = get_gradient(batch, old_theta, baseline=False, causality=True, importance_sampling=False)
        mu_causality.append(np.mean(grad_causality * env.max_num_steps, axis=1))
        std_causality.append(np.std(grad_causality * env.max_num_steps, axis=1))

        grad_no_IS = get_gradient(batch, theta, baseline=False, causality=False, importance_sampling=False)
        mu_no_IS.append(np.mean(grad_no_IS * env.max_num_steps, axis=1))
        std_no_IS.append(np.std(grad_no_IS * env.max_num_steps, axis=1))

        grad_importance_sampling = get_gradient(batch, theta, baseline=False, causality=False, importance_sampling=True)
        mu_importance_sampling.append(np.mean(grad_importance_sampling * env.max_num_steps, axis=1))
        std_importance_sampling.append(np.std(grad_importance_sampling * env.max_num_steps, axis=1))

        # mu_no.append(np.mean(grad_no * env.max_num_steps, axis=1))
        # mu_yes.append(np.mean(grad_yes * env.max_num_steps, axis=1))
        # std_no.append(np.std(grad_no * env.max_num_steps, axis=1))
        # std_yes.append(np.std(grad_yes * env.max_num_steps, axis=1))

    # mu_no = np.array(mu_no).mean(axis=0)
    # mu_yes = np.array(mu_yes).mean(axis=0)
    # std_no = np.array(std_no).mean(axis=0)
    # std_yes = np.array(std_yes).mean(axis=0)
    mu_no = np.array(mu_no).mean(axis=0)
    mu_no_IS = np.array(mu_no_IS).mean(axis=0)
    mu_baseline = np.array(mu_baseline).mean(axis=0)
    mu_causality = np.array(mu_causality).mean(axis=0)
    mu_importance_sampling = np.array(mu_importance_sampling).mean(axis=0)

    std_no = np.array(std_no).mean(axis=0)
    std_no_IS = np.array(std_no_IS).mean(axis=0)
    std_baseline = np.array(std_baseline).mean(axis=0)
    std_causality = np.array(std_causality).mean(axis=0)
    std_importance_sampling = np.array(std_importance_sampling).mean(axis=0)

    total_var_no = np.sum(std_no[:])
    total_var_no_IS = np.sum(std_no_IS[:])
    total_var_baseline = np.sum(std_baseline[:])
    total_var_causality = np.sum(std_causality[:])
    total_var_importance_sampling = np.sum(std_importance_sampling[:])

    # Print diagnostics
    # print(f'std dev without extensions : {total_var_no}')
    # print(f'std dev with baseline : {total_var_baseline}')
    # print(f'std dev with causality : {total_var_causality}')
    # print(f'std dev with importance sampling : {total_var_importance_sampling}')
    print(f'std dev relative change with baseline: {(total_var_baseline - total_var_no) / abs(total_var_no)}')
    print(f'std dev relative change with causality : {(total_var_causality - total_var_no) / abs(total_var_no)}')
    print(f'std dev relative change with importance sampling : {(total_var_importance_sampling - total_var_no) / abs(total_var_no)}')

    error_baseline = np.linalg.norm((mu_no - mu_baseline)) / np.linalg.norm(mu_no)
    error_causality = np.linalg.norm((mu_no - mu_causality)) / np.linalg.norm(mu_no)
    error_importance_sampling = np.linalg.norm((mu_no_IS - mu_importance_sampling)) / np.linalg.norm(mu_no_IS)
    print(f'mean relative change with baseline : {error_baseline}')
    print(f'mean relative change with causality : {error_causality}')
    print(f'mean relative change with importance sampling : {error_importance_sampling}')

    np.savetxt(os.path.join(SAVE_DIR, 'mean_no.csv'), mu_no)
    np.savetxt(os.path.join(SAVE_DIR, 'mean_no_IS.csv'), mu_no)
    np.savetxt(os.path.join(SAVE_DIR, 'mean_baseline.csv'), mu_baseline)
    np.savetxt(os.path.join(SAVE_DIR, 'mean_causality.csv'), mu_causality)
    np.savetxt(os.path.join(SAVE_DIR, 'mean_importance_sampling.csv'), mu_importance_sampling)

    np.savetxt(os.path.join(SAVE_DIR, 'std_no.csv'), std_no)
    np.savetxt(os.path.join(SAVE_DIR, 'std_no_IS.csv'), std_no)
    np.savetxt(os.path.join(SAVE_DIR, 'std_baseline.csv'), std_baseline)
    np.savetxt(os.path.join(SAVE_DIR, 'std_causality.csv'), std_causality)
    np.savetxt(os.path.join(SAVE_DIR, 'std_importance_sampling.csv'), std_importance_sampling)

    # plt.plot((mu_no - mu_yes) / mu_no) # plot percentage error for mean value of samples of different theta(s,a)
    # plt.show()

if __name__ == '__main__':
    main()