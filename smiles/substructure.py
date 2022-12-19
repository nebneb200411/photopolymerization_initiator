from rdkit import Chem

def check_substructure(check_smiles, sub_smiles):
    check_mol = Chem.MolFromSmiles(check_smiles)
    substructure = Chem.MolFromSmiles(sub_smiles)
    if check_mol:
        return check_mol.GetSubstructMatch(substructure)
    else:
        return None