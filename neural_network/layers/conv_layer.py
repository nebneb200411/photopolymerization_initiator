import torch.nn as nn
import torch

from .base_layer import BaseLayer

class Conv2dLayer(BaseLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, alpha=0.2):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
        )
        self.activation = nn.LeakyReLU(negative_slope=alpha)
    
    def forward(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x