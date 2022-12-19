from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from dotenv import load_dotenv
load_dotenv()

def is_radical(smiles):
    smiles = Chem.MolFromSmiles(smiles)
    if Descriptors.NumRadicalElectrons(smiles) > 0:
        return True
    else:
        return False


def get_bondic_distance(smiles):
    """発色団とオキシムエステル開裂部位の間に何個原子があるかを図る

    args
        smiles: 調査したい化合物のSmiles
    
    Note
        オキシムステルが前提
        ２つの開裂部位でフェニル基を挟んでいる場合はどうする？
        発色団の定義が曖昧なのでそこをはっきりさせたほうがいい
        発色団判定のルール
        もしも発色団を含んでいそうな系内で芳香環が複数去る場合は
    """
    mol = Chem.MolFromSmiles(smiles)
    oxim_mol = Chem.MolFromSmiles(os.environ['OXIMESTER_SMILES'])

    assert mol.GetSubstructMatch(oxim_mol), 'input smiles must be oximester!!'

    subs = mol.GetSubstructMatches(oxim_mol)
    distances = []
    for sub in subs:
        N_atom_idx = [x for x in sub if mol.GetAtomWithIdx(x).GetSymbol() == 'N'][0]
        C_atom_idx = [x.GetIdx() for x in mol.GetAtomWithIdx(N_atom_idx).GetNeighbors() if mol.GetAtomWithIdx(x.GetIdx()).GetSymbol() == 'C'][0]
        C_atom = mol.GetAtomWithIdx(C_atom_idx)

        if not C_atom.IsInRing():
            """
            オキシムエステルのC末端が環内にない場合の処理
            """
            C_neighbors = [x.GetIdx() for x in C_atom.GetNeighbors() if x.GetIdx() not in sub]
            cut_bonds = [mol.GetBondBetweenAtoms(x, C_atom_idx).GetIdx() for x in C_neighbors]
            frags = Chem.FragmentOnBonds(mol, cut_bonds, addDummies=False)
            # 生成したフラグメントから注目しているオキシムエステルをラベル付け
            for s in sub:
                frags.GetAtomWithIdx(s).SetAtomicNum(0)
            #frags = Chem.DeleteSubstructs(frags, Chem.MolFromSmarts('[#0]'))
            frags = Chem.GetMolFrags(frags, asMols=True)
            # frags <- [MolObject1, MolObject2, ...]
            frags = [x for x in frags if not x.HasSubstructMatch(Chem.MolFromSmarts('[#0]'))]
            aromatic_nums = [Descriptors.NumAromaticRings(frag) for frag in frags]

            # 芳香族数が違う場合の処理
            if aromatic_nums[0] != aromatic_nums[1]:
                # chlomopher <- MolObject
                chromophore = frags[aromatic_nums.index(max(aromatic_nums))]

            elif aromatic_nums[0] == 0 and aromatic_atoms[1] == 0:
                """
                特殊条件
                芳香環がどちらにもない場合 
                """
                molwts = [Chem.rdMolDescriptors._CalcMolWt(frag) for frag in frags]
                if molwts[0] == molwts[1]:
                    chromophore = frags[0]
                else:
                    chromophore = frags[molwts.index(max(molwts))]
            
            # 芳香族数が同じ場合の処理
            else:
                molwts = [Chem.rdMolDescriptors._CalcMolWt(frag) for frag in frags]
                if molwts[0] == molwts[1]:
                    chromophore = frags[0]
                else:
                    chromophore = frags[molwts.index(max(molwts))]
            # 発色団の抽出終わり
            # chromophore <- Mol
            sub_chr = mol.GetSubstructMatches(chromophore)[0]
            atoms_info = [mol.GetAtomWithIdx(x) for x in sub_chr]
            aromatic_atoms = []
            # 芳香族の原子をaromatic_atomsに入れる
            for atom in atoms_info:
                if atom.GetIsAromatic():
                    aromatic_atoms.append(atom.GetIdx())
            # Nと発色団の距離を測る。すべて計算して最小のものを抽出する
            
        else:
            """
            オキシムエステルのC末端が芳香環内にある場合の処理
            """
            distances.append(0)