import os
import multiprocessing
import subprocess
from tqdm import tqdm

NUM_PROCESSES = 4

with_replay_with_target = {
    'save_dir' : 'with_replay_with_target',
    'target_reset_frequency' : 1000,
    'replay_memory_buffer_size' : 10000,
    'mini_batch_size' : 25,
    'num_training_episodes' : 10000,
    'num_training_runs' : 10,
    'process_id' : 0
}

with_replay_without_target = with_replay_with_target.copy()
with_replay_without_target['save_dir'] = 'with_replay_without_target'
with_replay_without_target['target_reset_frequency'] = 1
with_replay_without_target['process_id'] = 1

without_replay_with_target = with_replay_with_target.copy()
without_replay_with_target['save_dir'] = 'without_replay_with_target'
without_replay_with_target['replay_memory_buffer_size'] = \
    without_replay_with_target['mini_batch_size']
without_replay_with_target['process_id'] = 2

without_replay_without_target = with_replay_with_target.copy()
without_replay_without_target['save_dir'] = 'without_replay_without_target'
without_replay_without_target['target_reset_frequency'] = 1
without_replay_without_target['replay_memory_buffer_size'] = \
    without_replay_without_target['mini_batch_size']
without_replay_without_target['process_id'] = 3

case = [
    with_replay_with_target,
    with_replay_without_target,
    without_replay_with_target,
    without_replay_without_target
    ]

def worker(params):
    cmd = "python train_pendulum.py "
    for key, value in params.items():
        cmd += "--{} {} ".format(key, value)
    os.system(cmd)

p = multiprocessing.Pool(NUM_PROCESSES)
p.map(worker, case)