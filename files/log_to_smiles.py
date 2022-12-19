import os, sys
sys.path.append('../')
from smiles.change_format import alt_smiles_to_smiles

def log_to_smiles(log_path):
    basename = os.path.basename(log_path)
    alt_smiles = basename.spilt('_')[0]
    smiles = alt_smiles_to_smiles(alt_smiles)
    return smiles