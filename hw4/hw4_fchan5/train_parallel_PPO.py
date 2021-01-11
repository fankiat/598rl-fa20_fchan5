import os
import multiprocessing
import subprocess
from tqdm import tqdm

NUM_PROCESSES = 4
NUM_TRAINING_RUNS = 20

pendulum = {
    'save_dir' : 'PPO_750iteration_20eps_1000mbs_20opstep',
    'mini_batch_size' : 1000,
    'alpha' : 0.001,
    'epsilon': 0.2,
    'num_iters' : 750,
    'num_eps_per_iter': 20,
    'max_num_steps_per_ep': 100,
    'actor_epochs_per_iter': 20,
    'critic_epochs_per_iter': 20,
    'sparse_reward': False,
    'dump_frequency': 10,
}

# generate all the workers
case = []
for irun in range(NUM_TRAINING_RUNS):
    pendulum['run_id'] = irun
    case.extend([
        pendulum.copy(),
        ])


def worker(params):
    python = '/Users/fchan5/Applications/miniconda3/envs/ethereal/bin/python'
    cmd = f'{python} train_PPO.py '
    for key, value in params.items():
        cmd += "--{} {} ".format(key, value)
    # print(cmd)
    os.system(cmd)

p = multiprocessing.Pool(NUM_PROCESSES)
p.map(worker, case, chunksize=1)
