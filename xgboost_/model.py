import os, sys
import pandas as pd
sys.path.append('../')
from fingerprint.fingeprint_from_smiles import smiles_list_to_fingerprint_dataframe

def XGBoost_estimator(smiles_list, mode):
    assert type(smiles_list) == list, 'Please input smiles as for list type'
    nBits = 2048
    radius = 3
    df = smiles_list_to_fingerprint_dataframe(smiles_list, nBits=nBits, radius=radius)

    x = df.iloc[:, 1:nBits]

    if mode == 'BDE':
        model = 'input model in here'
    else:
        model = 'input model in here'

    pred_BDE = model.predict(x)

    df_gen = pd.DataFrame()
    df_gen['Smiles'] = smiles_list
    df_gen[mode] = pred_BDE

    return df_gen