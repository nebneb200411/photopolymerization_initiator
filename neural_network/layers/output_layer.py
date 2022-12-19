import torch
import torch.nn as nn
import numpy as np

from .base_layer import BaseLayer


class DenseLayer(BaseLayer):
    def __init__(self, in_features=None, out_features=None, use_batchnorm=True, activation='ReLU'):
        super().__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.use_batchnorm = use_batchnorm

        self.linear = nn.Linear(self.in_feature, self.out_feature)
        if self.use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(self.out_feature)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x
