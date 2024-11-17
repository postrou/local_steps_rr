import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def gradient_norm(self):
        gn = 0
        for p in self.parameters():
            gn += p.grad.square().sum()
        return gn.sqrt().item()
