import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

from layers.base_layer import BaseLayer
from layers.dense_layer import DenseLayer

class DenseNetwork(BaseLayer):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict()

        ### defining model ###
        self.num_of_layers = 4
        self.layers[str(1)] = DenseLayer(in_features=84, out_features=42, activation=None, use_dropout=False, use_batchnorm=False) 
        self.layers[str(2)] = DenseLayer(in_features=42, out_features=16, activation=None, use_dropout=False, use_batchnorm=False)
        self.layers[str(3)] = DenseLayer(in_features=16, out_features=8, activation=None, use_dropout=False, use_batchnorm=False)
        #self.layers[str(4)] = nn.Linear(in_features=256, out_features=1)
        self.layers[str(4)] = DenseLayer(in_features=8, out_features=1, activation='tanh', use_dropout=False, use_batchnorm=False)

    def forward(self, x):
        for i in range(self.num_of_layers):
            x = self.layers[str(i+1)](x)
        return x

    def set_channel(self, pre_ch, mag):
        """
        pre_ch -> 一個前の層のチャンネルの出力
        next_ch -> 次の層のチャンネルの入力
        基本的に1/magしていく
        """
        next_ch = pre_ch / mag
        return next_ch