import random
import numpy as np
import matplotlib.pyplot as plt
import pendulum


def main():
    # Create environment
    env = pendulum.Pendulum()

    ######################################
    #
    #   EXAMPLE OF CREATING A VIDEO
    #

    # Define a policy that maps every state to the "zero torque" action
    policy = lambda s: np.array([0])

    # Simulate an episode and save the result as an animated gif
    env.video(policy, filename='results_pendulum.gif')

    #
    ######################################

    ######################################
    #
    #   EXAMPLE OF CREATING A PLOT
    #

    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = np.array([random.gauss(0, 1)])
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    data['a'] = np.array(data['a'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = data['a'][:, 0]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('results_pendulum.png')

    #
    ######################################


if __name__ == '__main__':
    main()
