import torch
from mpi4py import MPI
import numpy as np

class Actor(torch.nn.Module):
    def __init__(self, num_states, num_actions, args=None):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(num_states, 64).double()
        self.fc2 = torch.nn.Linear(64, 64).double()
        self.fc3 = torch.nn.Linear(64, num_actions).double()
        # You must cast the zero tensor as double() and not the Parameter - if
        # you try to cast the Parameter, it ceases to be a Parameter and simply
        # becomes a zero tensor again.
        self.std = torch.nn.Parameter(-0.5 * torch.ones(num_actions, dtype=torch.double).double())

    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        s = torch.tanh(self.fc2(s))
        mu = self.fc3(s)
        std = torch.exp(self.std)
        return (mu, std)

    def broadcast(self,):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        d = self.state_dict()
        if rank == 0:
            for k in sorted(d.keys()):
                comm.Bcast(d[k].numpy(), root=0)
        else:
            for k in sorted(d.keys()):
                param = np.empty_like(d[k].numpy())
                comm.Bcast(param, root=0)
                d[k] = torch.tensor(param).double()
            self.load_state_dict(d, strict=True)

    """ Gradient averaging. """
    def allreduce_average_gradients(self,):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        for param in self.parameters():
            local_grad = param.grad.detach().numpy()
            global_grad = np.zeros_like(local_grad)
            comm.Allreduce(local_grad, global_grad, op=MPI.SUM)
            global_grad /= size
            param.grad = torch.tensor(global_grad)

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
        # value = torch.sinh(self.fc3(s))
        return value

    def broadcast(self,):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        d = self.state_dict()
        if rank == 0:
            for k in sorted(d.keys()):
                comm.Bcast(d[k].numpy(), root=0)
        else:
            for k in sorted(d.keys()):
                param = np.empty_like(d[k].numpy())
                comm.Bcast(param, root=0)
                d[k] = torch.tensor(param).double()
            self.load_state_dict(d, strict=True)

    """ Gradient averaging. """
    def allreduce_average_gradients(self,):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        for param in self.parameters():
            local_grad = param.grad.detach().numpy()
            global_grad = np.zeros_like(local_grad)
            comm.Allreduce(local_grad, global_grad, op=MPI.SUM)
            global_grad /= size
            param.grad = torch.tensor(global_grad)