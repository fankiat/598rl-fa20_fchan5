import os
import multiprocessing
import subprocess
from tqdm import tqdm

NUM_PROCESSES = 8
NUM_TRAINING_RUNS = 100

reinforce = {
    'save_dir' : 'reinforce',
    'batch_size' : 100,
    'num_rollouts' : 200,
    'alpha' : 0.002,
    'baseline': False,
    'causality': False,
    'importance_sampling': False,
    'hard_version': False,
    'optimizer': 'SGD',
}

reinforce_b_c = reinforce.copy()
reinforce_b_c['save_dir'] = 'reinforce_baseline_causality'
reinforce_b_c['baseline'] = True
reinforce_b_c['causality'] = True

reinforce_b_is = reinforce.copy()
reinforce_b_is['save_dir'] = 'reinforce_baseline_importance_sampling'
reinforce_b_is['baseline'] = True
reinforce_b_is['importance_sampling'] = True

reinforce_c_is = reinforce.copy()
reinforce_c_is['save_dir'] = 'reinforce_causality_importance_sampling'
reinforce_c_is['causality'] = True
reinforce_c_is['importance_sampling'] = True

reinforce_b_c_is = reinforce.copy()
reinforce_b_c_is['save_dir'] = 'reinforce_baseline_causality_importance_sampling'
reinforce_b_c_is['baseline'] = True
reinforce_b_c_is['causality'] = True
reinforce_b_c_is['importance_sampling'] = True

# generate all the workers
case = []
for irun in range(NUM_TRAINING_RUNS):
    reinforce['run_id'] = irun
    reinforce_b_c['run_id'] = irun
    reinforce_b_is['run_id'] = irun
    reinforce_c_is['run_id'] = irun
    reinforce_b_c_is['run_id'] = irun
    case.extend([
        reinforce.copy(),
        reinforce_b_c.copy(),
        reinforce_b_is.copy(),
        reinforce_c_is.copy(),
        reinforce_b_c_is.copy()])


def worker(params):
    python = '/Users/fchan5/Applications/miniconda3/envs/ethereal/bin/python'
    cmd = f'{python} train_PG_gridworld.py '
    for key, value in params.items():
        cmd += "--{} {} ".format(key, value)
    # print(cmd)
    os.system(cmd)

p = multiprocessing.Pool(NUM_PROCESSES)
p.map(worker, case, chunksize=1)