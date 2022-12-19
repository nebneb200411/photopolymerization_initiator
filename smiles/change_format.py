from rdkit import Chem
from openbabel import pybel

def gen_alt_smiles(smiles):
    smiles = smiles.replace('\\', 'W')
    smiles = smiles.replace('/', 'Q')
    return smiles

def alt_smiles_to_smiles(alt_smiles):
    smiles = alt_smiles.replace('W', '\\')
    smiles = smiles.replace('Q', '/')
    return smiles

def smiles_to_mol_to_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def log_to_smiles(log_path):
    mol = next(pybel.readfile('log', log_path))
    smiles = mol.write(format="smi")
    smiles = smiles.split()[0].strip()
    return smiles

def sdf_to_smiles(sdf_path):
    """ChemDrawで作成した化合物をSmiles文字列に変換する

    Args
        sdf_path: ChemDrawで作成したsdfファイルのパス
    
    Returns
        Smiles文字列のリスト

    """
    smiles_list = []
    mols = Chem.SDMolSupplier(sdf_path)
    for mol in mols:
        if mol:
            smiles_list.append(Chem.MolToSmiles(mol))
    return smiles_list
