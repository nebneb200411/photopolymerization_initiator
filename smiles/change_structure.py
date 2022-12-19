from rdkit import Chem

def iminyl_to_cap(mol, acitive_radical):
    assert mol.HasSubstructMatch(Chem.MolFromSmiles('[N]=C')), 'input mol has to be iminiy radical'
    