import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from smiles.change_format import gen_alt_smiles
from dotenv import load_dotenv
load_dotenv()

class DatsGenerator:
    def __init__(self, smiles, base_dir, calc_condition, mode='MMFF'):
        self.smiles = smiles
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.base_dir = base_dir
        self.calc_condition = calc_condition
        self.mode = mode
    
    def write_dat(self, keyword='PM7'):
        mol = Chem.MolFromSmiles(self.smiles)
        mol = self.lfc.choose(mol)
        energy = []
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=1000, randomSeed=1234,
                                    pruneRmsThresh=2, numThreads=0)
        if self.mode == 'MMFF':
            for cid in cids:
                prop = AllChem.MMFFGetMoleculeProperties(mol)
                mmff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=cid)
                mmff.Minimize()
                energy.append((mmff.CalcEnergy(), cid))
        else:
            for cid in cids:
                uff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                uff.Minimize()
                energy.append((uff.CalcEnergy(), cid))
        energy.sort()
        conf = mol.GetConformer(energy[0][1])

        alt_smiles = gen_alt_smiles(self.smiles)
        dat_path = os.path.join(self.base_dir, '{}_{}_{}.dat'.format(alt_smiles, keyword, self.mode))
        with open(dat_path, 'w') as f:
            print(self.calc_condition, file=f)
            print(alt_smiles, file=f)
            print(os.environ['MOPAC_TITLE'], file=f)
            for n, (x, y, z) in enumerate(conf.GetPositions()):
                atom = mol.GetAtomWithIdx(n)
                print('{}  {:.6f} {:.6f} {:.6f}'.format(atom.GetSymbol(), x, y, z), file=f)
            f.close()
        return dat_path