import torch
import torch.nn as nn
from .base_layer import BaseLayer

class LinearLayer(BaseLayer):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        #self.activation = nn.ReLU(negative_slope=alpha)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x