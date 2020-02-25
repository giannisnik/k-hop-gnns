import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MessagePassing

class k_hop_GraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout, radius, device):
        super(k_hop_GraphNN, self).__init__()
        self.mp1 = MessagePassing(input_dim, hidden_dim, dropout, radius)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.relu = nn.ReLU()
        
    def forward(self, adj, features, idx):
        x = self.mp1(adj, features)
        x = self.bn1(x)
        idx = torch.transpose(idx.repeat(x.size()[1],1), 0, 1)
        out = torch.zeros(torch.max(idx)+1, x.size()[1]).to(self.device)
        out = out.scatter_add_(0, idx, x)
        out = self.bn2(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

    def clip_grad(self, max_norm):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        total_norm = total_norm ** (0.5)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        return total_norm