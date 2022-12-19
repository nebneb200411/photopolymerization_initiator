from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm
import numpy as np
import pandas as pd

def fingerprint(smiles, radius, nBits):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return fp

def smiles_list_to_fingerprint_dataframe(smiles_list, radius, nBits, delete_cols=None):
    """
    【目的】
    Smilesをリストで入力し、fingerprintを含んだデータフレームを返す
    radius -> 半径
    nBits -> 記述子の数
    """
    fingerprints = []
    smiles = []
    for s in tqdm(smiles_list):
        fgp = fingerprint(s, radius=radius, nBits=nBits)
        smiles.append(s)
        fingerprints.append(fgp)
    
    fingerprints = np.array(fingerprints)
    df_fin = pd.DataFrame(fingerprints)
    df_fin.insert(0, 'Smiles', smiles)
    if delete_cols:
        df_fin = df_fin.drop(delete_cols, axis=1)
    return df_fin

def dataframe_to_fingerprint(df, radius, nBits, y_col):
    fingerprints = []
    smiles = df['Smiles'].values
    for s in tqdm(smiles):
        fgp = fingerprint(s, radius=radius, nBits=nBits)
        fingerprints.append(fgp)
    fingerprints = np.array(fingerprints)
    df_fin = pd.DataFrame(data=fingerprints)
    df_fin.insert(0, 'Smiles', smiles)
    df_fin[y_col] = df[y_col].values
    return df_fin
