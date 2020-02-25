import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from mlp import MLP

class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, radius):
        super(MessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.radius = radius
        self.fc1 = nn.ModuleList([MLP(2, input_dim, output_dim, output_dim, dropout) for i in range(radius+2)])
        self.fc2 = nn.ModuleList([MLP(2, output_dim, output_dim, output_dim, dropout) for i in range(3*radius-2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, features):
        l = list()
        for i in range(self.radius+1):
            l.append(features[i])

        for i in range(2*self.radius-1, -1, -1):
            if i == 2*self.radius-1:
                if adj[i].shape != (1,1):
                    x = self.fc1[(i+1)//2](l[i//2+1]) + torch.spmm(adj[i], self.fc1[(i+1)//2+1](l[i//2+1]))
                else:
                    x = self.fc1[(i+1)//2](l[i//2+1])
            elif i%2 == 0:
                x = self.fc1[i//2](l[i//2]) + torch.spmm(adj[i], self.fc2[i+i//2](x))
            else:
                if adj[i].shape != (1,1):
                    x = self.fc2[i+(i-1)//2](x) + torch.spmm(adj[i], self.fc2[i+(i-1)//2+1](x))
            
            x = self.dropout(x)
            
        return x
