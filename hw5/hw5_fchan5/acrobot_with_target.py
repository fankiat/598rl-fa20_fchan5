import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class Acrobot():
    def __init__(
        self, rg=None, target_behavior='fixed', gravity=False, task='reach'):
        # Parameters that describe the physical system
        self.params = {
            'm1': 1.0,   # mass of link 1
            'm2': 1.0,   # mass of link 2
            'g': 0.0,   # acceleration of gravity
            'l1': 1.0,   # length of link 1
            'lc1': 0.5,   # length to center of mass of link 1
            'l2': 1.0,   # length of link 2
            'lc2': 0.5,   # length to center of mass of link 2
            'b': 0.1,   # coefficient of viscous friction
        }
        self.params['I1'] = 1. / 3. * self.params['m1'] * self.params['l1']**2
        self.params['I2'] = 1. / 3. * self.params['m2'] * self.params['l2']**2

        if gravity:
            self.params['g'] = 9.8

        # Maximum absolute angular velocity
        self.max_thetadot = 15.0

        # Maximum absolute torque
        self.max_tau = 20.

        # Time step
        self.dt = 0.1

        # Random number generator
        if rg is None:
            self.rg = np.random.default_rng()
        else:
            self.rg = rg

        # Dimension of state space
        self.num_states = 8

        # Dimension of action space
        self.num_actions = 2

        # Task for acrobot (reaching or collecting)
        self.task = task

        # Time horizon
        self.max_num_steps = 100

        # Target stuff
        # Maximum distance to be considered target goal achieved
        self.max_distance_for_goal = 0.1
        # Counter for time step tip spent on target
        self.on_target = 0
        # Counter for time step target spent on destination
        self.on_destination = 0
        # Target behavior - either random, forcing, fixed
        self.target_behavior = target_behavior

        # Reset to initial conditions
        self.reset()

    # s = [theta1, theta2, theta_dot1, theta_dot2, target_pos_x, target_pos_y, target_vel_x, target_vel_y]
    def _x_to_s(self, x):
        return np.array([
            ((x[0] + np.pi) % (2 * np.pi)) - np.pi,
            ((x[1] + np.pi) % (2 * np.pi)) - np.pi,
            x[2],
            x[3],
            self.target_pos[0],
            self.target_pos[1],
            self.target_vel[0],
            self.target_vel[1],])

    def _a_to_u(self, a):
        return np.clip(a, -self.max_tau, self.max_tau)

    def _dxdt(self, x, u):
        theta1 = float(x[0])
        theta2 = float(x[1])
        theta_dot1 = float(x[2])
        theta_dot2 = float(x[3])
        M = np.array([
            [
            self.params['I1'] + self.params['I2'] + self.params['m2'] * self.params['l1']**2 + 2 * self.params['m2'] * self.params['l1'] * self.params['lc2'] * np.cos(theta2),
            self.params['I2'] + self.params['m2'] * self.params['l1'] * self.params['lc2'] * np.cos(theta2)
            ],
            [
            self.params['I2'] + self.params['m2'] * self.params['l1'] * self.params['lc2'] * np.cos(theta2),
            self.params['I2']
            ]
            ])

        C = np.array([
            [
            -2 * self.params['m2'] * self.params['l1'] * self.params['lc2'] * np.sin(theta2) * theta_dot2,
            -self.params['m2'] * self.params['l1'] * self.params['lc2'] * np.sin(theta2) * theta_dot2
            ],
            [
            self.params['m2'] * self.params['l1'] * self.params['lc2'] * np.sin(theta2) * theta_dot1,
            0.0
            ]
            ])

        tau_g = np.array([
            -self.params['m1'] * self.params['g'] * self.params['lc1'] * np.sin(theta1) - self.params['m2'] * self.params['g'] * ( self.params['l1'] * np.sin(theta1) + self.params['lc2'] * np.sin(theta1 + theta2) ),
            -self.params['m2'] * self.params['g'] * self.params['lc2'] * np.sin(theta1 + theta2)
            ])

        B = np.eye(2)

        # q = np.array([x[0], x[1]]).flatten()
        q_dot = np.array([x[2], x[3]]).flatten()
        rhs =  -np.einsum('ij,j', C, q_dot) + tau_g + np.einsum('ij,j', B, u)
        # theta_ddot = scipy.linalg.solve(M, rhs)
        M_inv = 1. / (M[0,0] * M[1,1] - M[0,1] * M[1,0]) * np.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]])
        theta_ddot = np.einsum('ij,j', M_inv, rhs)

        return np.array([float(x[2]), float(x[3]), theta_ddot[0], theta_ddot[1]])

    def set_rg(self, rg):
        self.rg = rg

    def step(self, a):
        if np.ndim(a) != 1:
            raise Exception(' a = {} is not a one-dimensional array-like object'.format(a))

        if len(a) != 2:
            raise Exception(' a = {} has length {} instead of length 1'.format(a, len(a)))

        # Convert a to u
        u = self._a_to_u(a)

        # Solve ODEs to find new x
        sol = scipy.integrate.solve_ivp(fun=lambda t, x: self._dxdt(x, u), t_span=[0, self.dt], y0=self.x[0:4], t_eval=[self.dt])
        self.x[0:4] = sol.y[:, 0]

        # Convert x to s (same but with wrapped theta)
        self.s = self._x_to_s(self.x)

        # Get theta and thetadot
        theta = self.s[0:2]
        thetadot = self.s[2:4]

        # position of joint (p1) and tip (p2)
        p1 = np.array([ self.params['l1'] * np.sin(theta[0]), -self.params['l1'] * np.cos(theta[0]) ])
        p2 = np.array([ p1[0] + self.params['l2'] * np.sin(theta[0] + theta[1]), p1[1] - self.params['l2'] * np.cos(theta[0] + theta[1])])

        # problem length scale
        L = self.params['l1'] + self.params['l2']

        # Compute reward
        if np.any(abs(thetadot) > self.max_thetadot):
            r = -100
        else:
            # Task 1: Reach target within some clearance
            if self.task == 'reach':
                distance = np.linalg.norm(p2-self.target_pos) / L
            # Task 2: Collect target towards origin
            elif self.task == 'collect':
                distance = np.linalg.norm(self.target_pos)
            else:
                raise Exception('Unrecognized task!')
            if distance < self.max_distance_for_goal:
                bonus = 10.0 * L
            else:
                bonus = 0.0
            r_dense = -1.0 * distance**2 + bonus
            r = max(-100, r_dense)

        # Set target velocity and advect target
        # Fixed target
        if self.target_behavior == 'fixed':
            self.target_vel *= 0.0
        # Randomly moving target
        elif self.target_behavior == 'random':
            if self.num_steps % 10 == 0:
                self.target_vel = self.rg.uniform(-0.25, 0.25, 2)
                # make sure the random velocity doesnt advect the target out of acrobot's reach
                if np.linalg.norm(self.target_pos) > 0.9 * L:
                    self.target_vel = -0.1 * abs(self.target_pos)
        # Force on target, where force ~ - relative velocity / exp(distance)
        elif self.target_behavior == 'forcing':
            distance = np.linalg.norm(p2-self.target_pos)
            p2_dot = np.array([
                self.params['l1'] * thetadot[0] * np.cos(theta[0]) + self.params['l2'] * (thetadot[0] + thetadot[1]) * (np.cos(theta[0]) * np.cos(theta[1]) - np.sin(theta[0]) * np.sin(theta[1])),
                self.params['l1'] * thetadot[0] * np.sin(theta[0]) + self.params['l2'] * (thetadot[0] + thetadot[1]) * (np.sin(theta[0]) * np.cos(theta[1]) + np.cos(theta[0]) * np.sin(theta[1]))
                ])
            vel_relative = self.target_vel - p2_dot
            nu = 1e-1
            target_radius = 0.1
            # force = -1.0 * vel_relative / np.exp(distance) - 6.0 * np.pi * target_radius * nu * self.target_vel
            force = -1.0 * (vel_relative / 1.) / np.exp(L * 5 * distance**2) - 6.0 * np.pi * target_radius * nu * self.target_vel
            # update velocity
            m_target = 1.0 # for now
            self.target_vel += force / m_target * self.dt
        else:
            raise Exception('Target velocity mode is not specified!')
        # advect target
        self.target_pos = self.target_pos + self.dt * self.target_vel

        # Increment number of steps and check for end of episode
        self.num_steps += 1
        if self.task == 'collect':
            done = (self.num_steps >= self.max_num_steps) or (np.linalg.norm(self.target_pos) >= 1.5 * L) or (np.linalg.norm(self.target_pos) <= self.max_distance_for_goal / 2.0)
        else:
            done = (self.num_steps >= self.max_num_steps) or (np.linalg.norm(self.target_pos) >= 1.5 * L)

        return (self.s, r, done)

    def reset(self):
        # Acrobot reset
        # Sample theta1, theta2 and set thetadot1 and thetadot2 to zero
        self.x = np.zeros(self.num_states)
        # set random initial acrobot configuration
        self.x[0:2] = self.rg.uniform(
            [-np.pi, -np.pi],
            [np.pi, np.pi]
            )

        # Target reset
        # set position to some fixed position and velocity to zero
        self.target_pos = np.array([1. * np.cos(np.deg2rad(70.)), 1. * np.sin(np.deg2rad(70.))])
        self.target_vel = np.array([0.0, 0.0])

        # Convert x to s
        self.s = self._x_to_s(self.x)

        # Reset current time (expressed as number of simulation steps taken so far) to zero
        self.num_steps = 0

        return self.s

    def video(self, policy, filename='pendulum.gif', writer='imagemagick'):
        s = self.reset()
        s_traj = [s]
        a_traj = [policy(s)]
        done = False
        while not done:
            a = policy(s)
            (s, r, done) = self.step(a)
            s_traj.append(s)
            a_traj.append(a)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
        ax.set_aspect('equal')
        ax.grid()
        line, = ax.plot([], [], 'ko-', lw=2)
        scat = ax.scatter([], [], c=[], s=100, edgecolor="k", cmap='coolwarm', zorder=10, vmin=-self.max_tau, vmax=self.max_tau)
        # colorbar
        cbar = plt.colorbar(scat)
        # cbar.ax.set_yticklabels(['0','1','2','>3'])
        cbar.set_label(r'$\tau$', fontsize=25)
        text = ax.set_title('')

        def animate(i):
            theta1 = s_traj[i][0]
            theta2 = s_traj[i][1]
            a = a_traj[i]
            p1 = [ self.params['l1'] * np.sin(theta1), -self.params['l1'] * np.cos(theta1) ]
            p2 = [ p1[0] + self.params['l2'] * np.sin(theta1 + theta2), p1[1] - self.params['l2'] * np.cos(theta1 + theta2)]
            line.set_data([0, p1[0], p2[0]], [0, p1[1], p2[1]])
            scat.set_offsets(np.array([[0, 0], [p1[0], p1[1]], [p2[0], p2[1]]]))
            col = self._a_to_u(a)
            scat.set_array(np.append(col, 0))
            text.set_text('time = {:3.1f}'.format(i * self.dt))
            return line, text

        anim = animation.FuncAnimation(fig, animate, len(s_traj), interval=(1000 * self.dt), blit=True, repeat=False)
        anim.save(filename, writer=writer, fps=10)

        plt.close()

    def timeit(self, n=10):
        policy = lambda s: self.rg.uniform(-self.max_tau, self.max_tau, self.num_actions)
        tic = time.perf_counter()
        for it in range(n):
            s = self.reset()
            s_traj = [s]
            done = False
            while not done:
                (s, r, done) = self.step(policy(s))
                s_traj.append(s)
        toc = time.perf_counter()
        print("Completed {} episodes with {:0.4f} seconds/episode".format(n, (toc - tic) / n))


if __name__ == '__main__':
    env = Acrobot()
    env.timeit()
