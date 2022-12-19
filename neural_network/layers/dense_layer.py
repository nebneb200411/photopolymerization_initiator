import torch
import torch.nn as nn
import numpy as np

from .base_layer import BaseLayer


class DenseLayer(BaseLayer):
    def __init__(self, in_features=None, out_features=None, use_batchnorm=True, use_dropout=True, activation='ReLU'):
        super().__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.linear = nn.Linear(self.in_feature, self.out_feature, bias=True)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.25)
        if self.use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(self.out_feature)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = None

    def forward(self, x):
        x = self.linear(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x
