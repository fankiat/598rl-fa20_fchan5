import os
import numpy as np

data = {
    'step': [],
    'reward': [],
    'losses': [],
}

for i in range(40):
    reward = np.genfromtxt('run{:03d}/reward.csv'.format(i), delimiter=',')
    loss = np.genfromtxt('run{:03d}/loss.csv'.format(i), delimiter=',')
    data['step'] = reward[:,0]
    data['reward'] = reward[:,1]
    data['losses'] = loss[:,1]
    os.system("mv run{:03d}/training_data.npy run{:03d}/training_data_old.npy". format(i, i))
    np.save('run{:03d}/training_data.npy'.format(i), data)

