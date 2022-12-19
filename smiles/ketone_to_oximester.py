from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import os
from .change_format import gen_alt_smiles

class KetoneToOximester:
    """
    【目的】
    ケトンのSmilesを受け取りそれをオキシムエステルに変換
    【注意】
    あらかじめsearch_ketone等でketoneのスクリーニングを行うこと!!
    Smilesの名前が長すぎるとエラーになるのでファイル名は最大20文字列分までに指定している
    """
    def __init__(self, ketone_smiles, img_save_dir=None):
        self.ketone_smiles = ketone_smiles
        self.replaced_smiles = []
        self.img_save_dir = img_save_dir
    
    def replace(self):
        ketone_mol = Chem.MolFromSmiles(self.ketone_smiles)
        replace_from = Chem.MolFromSmiles("C=O")
        replace_to = Chem.MolFromSmiles("C=NOC(=O)C")

        replaced_mols = AllChem.ReplaceSubstructs(ketone_mol, replace_from, replace_to)

        for replaced_mol in replaced_mols:
            if self.is_oximester(replaced_mol) and not self.is_inring(replaced_mol):
                replaced_s = Chem.MolToSmiles(replaced_mol)
                if '.' in replaced_s:
                    pass
                else:
                    if self.validation_2nd_step(replaced_mol):
                        self.replaced_smiles.append(replaced_s)
                        img = Draw.MolToImage(replaced_mol)
                        if len(replaced_s) > 40:
                            replaced_s = replaced_s[:40]
                        if self.img_save_dir:
                            img.save(os.path.join(self.img_save_dir, gen_alt_smiles(replaced_s) + '.png'))
            """
            else:
                replaced_s = Chem.MolToSmiles(replaced_mol)
                print('Failed To Convert!!! {}'.format(replaced_s))
            """
        
        return self.replaced_smiles
    
    def is_oximester(self, mol):
        sub = Chem.MolFromSmiles('NOC(=O)C')
        return mol.HasSubstructMatch(sub)
    
    def validation_2nd_step(self, mol):
        oxim = Chem.MolFromSmiles('NOC(=O)C')
        atom_indexs = sorted(mol.GetSubstructMatches(oxim)[0])
        N_index = [x for x in atom_indexs if mol.GetAtomWithIdx(x).GetAtomicNum() == 7][0]
        N_atom = mol.GetAtomWithIdx(N_index)
        ketone_base = N_atom.GetNeighbors()[0]
        ketone_base_neibors = ketone_base.GetNeighbors()
        ketone_base_neibors = [x for x in ketone_base_neibors if x.GetIdx() != N_index]
        """
        if len(ketone_base_neibors) != 2:
            return False
        """
        if ketone_base_neibors[0].GetAtomicNum() == 6 and ketone_base_neibors[1].GetAtomicNum() == 6:
            return True
        else:
            return False
    
    def is_inring(self, mol):
        oxim = Chem.MolFromSmiles("C=NOC(=O)C")
        atom_indexs = sorted(mol.GetSubstructMatches(oxim)[0])
        N_index = [x for x in atom_indexs if mol.GetAtomWithIdx(x).GetAtomicNum() == 7][0]
        N_atom = mol.GetAtomWithIdx(N_index)
        ketone_base = N_atom.GetNeighbors()[0]
        return ketone_base.IsInRing()

def search_ketone(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ketone = Chem.MolFromSmiles('CC(=O)C')
    if mol is None:
        return False
    return mol.HasSubstructMatch(ketone)

