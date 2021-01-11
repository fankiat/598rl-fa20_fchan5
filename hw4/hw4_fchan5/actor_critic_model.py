import torch

class Actor(torch.nn.Module):
    def __init__(self, num_states, num_actions, args=None):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(num_states, 64).double()
        self.fc2 = torch.nn.Linear(64, 64).double()
        self.fc3 = torch.nn.Linear(64, num_actions).double()
        # You must cast the zero tensor as double() and not the Parameter - if
        # you try to cast the Parameter, it ceases to be a Parameter and simply
        # becomes a zero tensor again.
        self.std = torch.nn.Parameter(-0.5 * torch.ones(1, dtype=torch.double).double())

    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        s = torch.tanh(self.fc2(s))
        mu = self.fc3(s)
        std = torch.exp(self.std)
        return (mu, std)

class Critic(torch.nn.Module):
    def __init__(self, num_states, num_actions, args=None):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(num_states, 64).double()
        self.fc2 = torch.nn.Linear(64, 64).double()
        self.fc3 = torch.nn.Linear(64, 1).double()

    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        s = torch.tanh(self.fc2(s))
        value = self.fc3(s)
        return value