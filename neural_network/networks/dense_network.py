import torch.nn as nn
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append('../')
try:
    from layers.base_layer import BaseLayer
    from layers.dense_layer import DenseLayer
except:
    from neural_network.layers.base_layer import BaseLayer
    from neural_network.layers.dense_layer import DenseLayer

class BDE_estimator(BaseLayer):
    def __init__(self, first_node):
        super().__init__()
        self.layers = nn.ModuleDict()

        ### defining model ###
        self.num_of_layers = 13
        use_dropout = False
        use_batch_norm1 = False
        use_batch_norm2 = False
        activation1 = 'LeakyReLU'
        final_activation = 'ReLU'
        self.layers[str(1)] = DenseLayer(first_node, 4096, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(2)] = DenseLayer(4096, 2048, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(3)] = DenseLayer(2048, 1024, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(4)] = DenseLayer(1024, 512, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(5)] = DenseLayer(512, 256, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(6)] = DenseLayer(256, 128, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(7)] = DenseLayer(128, 64, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(8)] = DenseLayer(64, 32, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(9)] = DenseLayer(32, 16, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(10)] = DenseLayer(16, 8, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(11)] = DenseLayer(8, 4, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(12)] = DenseLayer(4, 2, activation=activation1, use_dropout=use_dropout, use_batchnorm=use_batch_norm1)
        self.layers[str(13)] = DenseLayer(2, 1, activation=final_activation, use_dropout=use_dropout, use_batchnorm=use_batch_norm2)

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

class ABS_estimator(BaseLayer):
    def __init__(self, first_node, use_dropout=False, use_batchnorm=True):
        super().__init__()
        self.layers = nn.ModuleDict()

        ### defining model ###
        self.num_of_layers = 12
        self.layers[str(1)] = DenseLayer(in_features=first_node, out_features=2048, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm) 
        self.layers[str(2)] = DenseLayer(in_features=2048, out_features=1024, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm) 
        self.layers[str(3)] = DenseLayer(in_features=1024, out_features=512, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(4)] = DenseLayer(in_features=512, out_features=256, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(5)] = DenseLayer(in_features=256, out_features=128, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(6)] = DenseLayer(in_features=128, out_features=64, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(7)] = DenseLayer(in_features=64, out_features=32, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(8)] = DenseLayer(in_features=32, out_features=16, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(9)] = DenseLayer(in_features=16, out_features=8, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(10)] = DenseLayer(in_features=8, out_features=4, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(11)] = DenseLayer(in_features=4, out_features=2, activation='LeakyReLU', use_dropout=use_dropout, use_batchnorm=use_batchnorm)
        self.layers[str(12)] = DenseLayer(in_features=2, out_features=1, activation='sigmoid', use_dropout=use_dropout, use_batchnorm=use_batchnorm)

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

class ABS_estimator_fingerprint(BaseLayer):
    def __init__(self, first_node):
        super().__init__()
        self.layers = nn.ModuleDict()

        ### defining model ###
        self.num_of_layers = 12
        self.layers[str(1)] = DenseLayer(in_features=first_node, out_features=2048, activation='LeakyReLU', use_dropout=False, use_batchnorm=True) 
        self.layers[str(2)] = DenseLayer(in_features=2048, out_features=1024, activation='LeakyReLU', use_dropout=False, use_batchnorm=True) 
        self.layers[str(3)] = DenseLayer(in_features=1024, out_features=512, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(4)] = DenseLayer(in_features=512, out_features=256, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(5)] = DenseLayer(in_features=256, out_features=128, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(6)] = DenseLayer(in_features=128, out_features=64, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(7)] = DenseLayer(in_features=64, out_features=32, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(8)] = DenseLayer(in_features=32, out_features=16, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(9)] = DenseLayer(in_features=16, out_features=8, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(10)] = DenseLayer(in_features=8, out_features=4, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(11)] = DenseLayer(in_features=4, out_features=2, activation='LeakyReLU', use_dropout=False, use_batchnorm=True)
        self.layers[str(12)] = DenseLayer(in_features=2, out_features=1, activation='sigmoid', use_dropout=False, use_batchnorm=True)

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