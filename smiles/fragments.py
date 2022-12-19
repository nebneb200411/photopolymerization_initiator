from rdkit import Chem
from rdkit.Chem import Recap
import os
from dotenv import load_dotenv
load_dotenv()


class FragementGenerator:
    def __init__(self, smiles):
        """主にオキシムエステルに関するフラグメントを作成

        Note
            cut_NO_bond: N-O結合を切る
            decarboxylation: アセトキシラジカルのCO2を切る
        """
        self.smiles = smiles
        
    def recap(self):
        mol = Chem.MolFromSmiles(self.smiles)
        fragments = Recap.RecapDecompose(mol)
        fragments = fragments.children.keys()
        fragments = list(fragments)
        return fragments
    
    def cut_NO_bond(self):
        mol = Chem.MolFromSmiles(self.smiles)
        sub = Chem.MolFromSmiles(os.environ['OXIMESTER_SMILES'])
        subs = mol.GetSubstructMatches(sub)
        
        radicals = []
        carb_rad = Chem.MolFromSmiles('C(=O)[O]')
        iminyl_rad = Chem.MolFromSmiles('[N]=C')
        for sub in subs:
            N_bond_idx = [x for x in sub if mol.GetAtomWithIdx(x).GetSymbol() == 'N'][0]
            O_bond_idx = [x.GetIdx() for x in mol.GetAtomWithIdx(N_bond_idx).GetNeighbors() if mol.GetAtomWithIdx(x.GetIdx()).GetSymbol() == 'O'][0]
            bond = mol.GetBondBetweenAtoms(N_bond_idx, O_bond_idx)
            frags = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(0, 0)])
            frags = Chem.MolToSmiles(frags)
            if frags.count('.') == 1:
                rad1, rad2 = frags.split('.')
                def checker(rad):
                    rad = rad.replace('*', '')
                    if rad[0] == '/' or rad[0] == '\\':
                        rad = rad[1:]
                    return rad
                rad1 = checker(rad1)
                rad2 = checker(rad2)
                def to_radical(rad):
                    rad = '[' + rad[0] + ']' + rad[1:]
                    return rad
                rad1 = to_radical(rad1)
                rad2 = to_radical(rad2)

                rad_dict = {}
                try:
                    rad1_mol = Chem.MolFromSmiles(rad1)
                    rad2_mol = Chem.MolFromSmiles(rad2)
                    rad1 = Chem.MolToSmiles(rad1_mol)
                    rad2 = Chem.MolToSmiles(rad2_mol)
                    rad_dict['acetoxy'] = rad1 if rad1_mol.GetSubstructMatch(carb_rad) else rad2
                    rad_dict['iminyl'] = rad2 if rad2_mol.GetSubstructMatch(iminyl_rad) else rad1
                    
                except:
                    rad_dict['acetoxy'] = None
                    rad_dict['iminyl'] = None
                radicals.append(rad_dict)
            else:
                radicals.append(None)

        return radicals

    """
    def decarboxylation(self):
        mol = Chem.MolFromSmiles(self.smiles)
        fragments = Recap.RecapDecompose(mol)
        fragments = fragments.children.keys()
        fragments = list(fragments)
        substructure = Chem.MolFromSmiles('C([O])=O')
        substructure_smile = Chem.MolToSmiles(substructure)
        
        substructure_idx = None
        for i, fragment in enumerate(fragments):
            fragment = fragment.strip('*')
            fragment = Chem.MolFromSmiles(fragment)
            fragment = Chem.MolToSmiles(fragment)
            if fragment==substructure_smile:
                substructure_idx = i
                break
        
        def to_radical(smiles):
            smiles = "[" + smiles[1] + "]" + smiles[2:]
            return smiles

        if (substructure_idx is not None) and (substructure % 2 != 0):
            radical_idx = substructure_idx + 1
            radical = fragments[radical_idx]
            radical = to_radical(radical)

        else:
            radical = None
        return radical
    """
    
    def decarboxylation(self):
        mol = Chem.MolFromSmiles(self.smiles)
        substructure = Chem.MolFromSmiles('C(=O)[O]')

        if mol.GetSubstructMatch(substructure):
            sub_poses = mol.GetSubstructMatches(substructure)[0]

            # determine position of C
            cPos = [x for x in sub_poses if mol.GetAtomWithIdx(x).GetSymbol() == 'C'][0]
            cAtom = mol.GetAtomWithIdx(cPos)
            neighbors = cAtom.GetNeighbors()
            c_neighbor_pos = [x.GetIdx() for x in neighbors if x.GetIdx() not in sub_poses][0]
            cut_bond = mol.GetBondBetweenAtoms(cPos, c_neighbor_pos)
            frags = Chem.FragmentOnBonds(mol, [cut_bond.GetIdx()], dummyLabels=[(0, 0)])
            frags = Chem.MolToSmiles(frags)
            
            frag1, frag2 = frags.split('.')

            def checker(frag):
                frag = frag.replace('*', '')
                if frag[0] == '/' or frag[0] == '\\':
                    frag = frag[1:]
                return frag
            frag1 = checker(frag1)
            frag2 = checker(frag2)
            
            rad = [x for x in [frag1, frag2] if not Chem.MolFromSmiles(x).GetSubstructMatch(substructure)][0]

            def to_radical(rad):
                rad = '[' + rad[0] + ']' + rad[1:]
                return rad

            rad = to_radical(rad)

            try:
                rad = Chem.MolFromSmiles(rad)
                rad = Chem.MolToSmiles(rad)
            except:
                rad = None
        
        else:
            rad = None
        return rad
        """
        # get all atoms index and symbols
        pares = []
        atoms = mol.GetAtoms()
        for atom in atoms:
            pare = [atom.GetIdx(), atom.GetSymbol()]
            pares.append(pare)
        
        # detect where C is in substructure
        o_pos = []
        for pare in pares:
            pos = pare[0]
            atom = pare[1]
            if (pos in sub_pos) and (atom == 'O'):
                o_pos.append(pos)
        
        c_pos = [x for x in sub_pos if x not in o_pos][0]
        
        cut_bond = None
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if ((begin not in o_pos) and (end not in o_pos)) and ((begin == c_pos) or (end == c_pos)):
                cut_bond = bond.GetIdx()
        
        fragments_mol = Chem.FragmentOnBonds(mol, [cut_bond], addDummies=True, dummyLabels=[(0, 0)])
        fragments_smiles = Chem.MolToSmiles(fragments_mol)
        
        if fragments_smiles.count('.') == 0:
            radical = None
        else:
            dot_pos = fragments_smiles.find('.')
            fragment1 = fragments_smiles[:dot_pos]
            fragment2 = fragments_smiles[dot_pos+1:]
            fragment1 = fragment1.strip('[*]')
            fragment2 = fragment2.strip('[*]')
            fragments = [fragment1, fragment2]
            
            new_fragments = []
            for fragment in fragments:
                try:
                    mol = Chem.MolFromSmiles(fragment)
                    if mol.HasSubstructMatch(substructure):
                        fragment = "C(=O)=O"
                    else:
                        pass
                except:
                    if fragment[0] == "\\":
                        fragment = fragment[1:]
                        mol = Chem.MolFromSmiles(fragment)
                        if mol.HasSubstructMatch(substructure):
                            fragment = "C(=O)=O"
                        else:
                            pass
                    else:
                        fragment = "WRONG"
                new_fragments.append(fragment)
            
            radical = [x for x in new_fragments if 'C(=O)=O' not in x][0]
            radical = '[' + radical[0] + ']' + radical[1:]
            
        return radical
        """