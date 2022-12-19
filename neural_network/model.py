import pandas as pd
from descriptor.fingeprint_from_smiles import smiles_list_to_fingerprint_dataframe
import torch
from torch import Tensor
from networks.dense_network import DenseNetwork
import os
from dotenv import load_dotenv
load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def NeuralNetwork_estimator(smiles_list, mode):
    assert type(smiles_list) == list, 'Please input smiles as for list type'
    nBits = int(os.environ['NBITS'])
    radius = int(os.environ['RADIUS'])
    df = smiles_list_to_fingerprint_dataframe(smiles_list, nBits=nBits, radius=radius)

    x = df.iloc[:, 1:nBits]
    x = Tensor(x).to(device)

    if mode == 'BDE':
        model = DenseNetwork(data_length=nBits)
        model.load_state_dict(torch.load('./model/model_BDE_2048_3.pth', map_location=device))
    else:
        model = DenseNetwork(data_length=nBits)
        model.load_state_dict(torch.load('./model/model_abs_2048_3.pth', map_location=device))

    pred = model(x)

    df = pd.DataFrame()
    df['Smiles'] = smiles_list
    df[mode] = pred.detach().numpy

    return df