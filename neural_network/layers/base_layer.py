import torch
import torch.nn as nn

class BaseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    @classmethod
    def print_model_parameters(cls, model, model_name):
        print('---- {} ----'.format(model_name))
        for index, key in enumerate(model.state_dict().keys()):
            print('index: {}, name: {}, shape: {}'.format(index, key, model.state_dict()[key].shape))