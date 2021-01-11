import torch

class QNet(torch.nn.Module):
    def __init__(self, num_states, num_actions, args=None):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(num_states, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        # return the Q values for each action
        return x