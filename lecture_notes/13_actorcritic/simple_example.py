import numpy as np

class SimpleExample():
    def __init__(self, rg=None, prob_failed_move=0.1, prob_failed_search=0.2, prob_done=0.8, rewards=[20., 1.]):
        # Probability that result of action is not what you want
        self.prob_failed_move = prob_failed_move
        self.prob_failed_search = prob_failed_search
        self.prob_done = prob_done

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

        # Reset to initial conditions
        self.reset()

    def p0(self, s):
        return 0.5

    def p(self, s_next, s, a, done=False):
        if done:
            p = self.prob_done
        else:
            p = 1 - self.prob_done
        if a == 0:
            return p if s_next == s else 0.
        else:
            return (p * self.prob_failed_move) if s_next == s else (p * (1 - self.prob_failed_move))

    def r(self, s, a, s_next):
        if a == 0:
            return (1 - self.prob_failed_search) * self.rewards[s]
        else:
            return 0.

    def set_rg(self, rg):
        self.rg = rg

    def step(self, a):
        if a == 0:
            self.s = self.s
            if self.rg.random() < self.prob_failed_search:
                r = 0.
            else:
                r = self.rewards[self.s]
        elif a == 1:
            if self.rg.random() < self.prob_failed_move:
                self.s = self.s
            else:
                self.s = 1 - self.s
            r = 0.
        else:
            raise ValueError(f'invalid action {a}')

        # Increment number of steps and check for end of episode
        self.num_steps += 1
        done = (self.rg.random() < self.prob_done)

        return (self.s, r, done)

    def reset(self):
        # Sample state uniformly at random
        self.s = self.rg.integers(self.num_states)

        # Reset current time (expressed as number of simulation steps taken so far) to zero
        self.num_steps = 0

        return self.s
