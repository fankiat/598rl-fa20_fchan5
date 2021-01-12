import numpy as np

class SimpleExample():
    def __init__(self, rg=None, beta=0.2, rewards=[10, 0]):
        # Probability that result of action is not what you want
        self.beta = beta

        # Rewards for ending up in a state
        self.rewards = rewards

        # Random number generator
        if rg is None:
            self.rg = np.random.default_rng()
        else:
            self.rg = rg

        # Number of states
        self.num_states = 2

        # Number of actions
        self.num_actions = 2

        # Time horizon
        self.max_num_steps = 2

        # Reset to initial conditions
        self.reset()

    def p0(self, s):
        # Check arguments
        if s not in [0, 1]:
            raise ValueError(f'invalid state {s}')
        return 0.5

    def p(self, s_next, s, a):
        # Check arguments
        if s_next not in [0, 1]:
            raise ValueError(f'invalid next state {s_next}')
        if s not in [0, 1]:
            raise ValueError(f'invalid state {s}')
        if a not in [0, 1]:
            raise ValueError(f'invalid action {a}')
        # Return transition probability
        if a == 0:
            return (1 - self.beta) if s_next == s else self.beta
        else:
            return self.beta if s_next == s else (1 - self.beta)

    def r(self, s, a, s_next):
        # Check arguments
        if s_next not in [0, 1]:
            raise ValueError(f'invalid next state {s_next}')
        if s not in [0, 1]:
            raise ValueError(f'invalid state {s}')
        if a not in [0, 1]:
            raise ValueError(f'invalid action {a}')
        # Return mean reward (happens to be a deterministic function of s_next)
        return self.rewards[s_next]

    def set_rg(self, rg):
        self.rg = rg

    def step(self, a):
        # Compute state
        if a == 0:
            if self.rg.random() < self.beta:
                self.s = 1 - self.s
            else:
                self.s = self.s
        elif a == 1:
            if self.rg.random() < self.beta:
                self.s = self.s
            else:
                self.s = 1 - self.s
        else:
            raise ValueError(f'invalid action {a}')

        # Compute reward
        r = self.rewards[self.s]

        # Increment number of steps and check for end of episode
        self.num_steps += 1
        done = (self.num_steps >= self.max_num_steps)

        return (self.s, r, done)

    def reset(self):
        # Sample state uniformly at random
        self.s = self.rg.integers(self.num_states)

        # Reset current time (expressed as number of simulation steps taken so far) to zero
        self.num_steps = 0

        return self.s
