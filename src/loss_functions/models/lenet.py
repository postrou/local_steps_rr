import torch
import torch.nn as nn


class LeNet5(nn.Module):
    
    def __init__(self, in_features=3):
        super().__init__()
        
        self.conv_block = nn.Sequential( nn.Conv2d(in_channels=in_features,
                                                   out_channels=6,
                                                   kernel_size=5,
                                                   stride=1),
                                         nn.Tanh(),
                                         nn.MaxPool2d(2,2),
                                         
                                         nn.Conv2d(in_channels=6,
                                                   out_channels=16,
                                                   kernel_size=5,
                                                   stride=1),
                                         nn.Tanh(),
                                         nn.MaxPool2d(2,2)
                                        )
        
        self.linear_block = nn.Sequential( nn.Linear(16*5*5, 120),
                                           nn.Tanh(),
                                           nn.Linear(120,84),
                                           nn.Tanh(),
                                           nn.Linear(84,10)
                                         )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x,1)
        x = self.linear_block(x)
        return x

    def gradient_norm(self):
        gn = 0
        for p in self.parameters():
            gn += p.grad.square().sum()
        return gn.sqrt().item()


def build_lenet_model(device):
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion