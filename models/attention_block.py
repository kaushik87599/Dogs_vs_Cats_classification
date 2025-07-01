from torch import nn
import torch

class AttentionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels,ratio = 4):
        super(AttentionBlock, self).__init__()
        self.ratio = ratio
        self.sequeeze = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=(out_channels//self.ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=(out_channels//self.ratio), out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        original = x
        x = self.sequeeze(x)
        x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        self.attention_weights = x      # Store the attention weights
        x = original*x.view(x.size(0), x.size(1), 1, 1)
        return x
    




