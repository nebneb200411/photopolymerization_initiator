from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
import os


class Smiles_to_gjf:
    def __init__(self):
        """
        【目的】
        Smilesをgjfファイルに変換する
        【仕様】
        gjfs_save_dirに指定したディレクトリにgjfを保存する
        neutralがTrueだと通常の分子としてgjfファイルを作成, neutralがFalseだとラジカルとしてgjfファイルを保存
        chargeには電荷と多重度を指定 Exp)charge='0 1'
        calc_conditionに計算条件を指定
        【注意点】
        これを使うぐらいならGaussian/calc.py/GaussianSequentialCalculationFromSmilesを使った方がいい
        """
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
    
    def to_gjf(self, smiles, gjfs_save_dir, neutral, charge, calc_condition, optionals):
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol) # smilesの名前取得
        mol = self.lfc.choose(mol)
        alt_smiles = self.gen_alt_smiles(smiles)

        if neutral:
            gjf_extension = '_n.gjf'
        else:
            gjf_extension = '_rc.gjf'

        try:
            energy = []
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol) # 3次元構造化
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=1000, randomSeed=1234,
                                        pruneRmsThresh=2, numThreads=0)

            for cid in cids:
                prop = AllChem.MMFFGetMoleculeProperties(mol)
                mmff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=cid)
                mmff.Minimize()
                energy.append((mmff.CalcEnergy(), cid))
            energy.sort()
            conf = mol.GetConformer(energy[0][1])
            gjf_path = os.path.join(gjfs_save_dir, alt_smiles + gjf_extension)

            self.writer(mol, conf, gjf_path, alt_smiles, neutral, charge, calc_condition, optionals)
            
            smiles_path = alt_smiles + gjf_extension

        except:
            smiles_path = 0
        return smiles_path

    def gen_alt_smiles(self, smiles):
        smiles = smiles.replace('\\', 'W')
        smiles = smiles.replace('/', 'Q')
        return smiles
    
    def writer(self, mol, conf, file_path, alt_smiles, neutral, charge, calc_condition, optionals):
        with open(file_path, 'w') as f:
            if neutral:
                chk_extension = '_n.chk'
            else:
                chk_extension = '_rc.chk'
            print('%chk=' + alt_smiles + chk_extension, file=f)
            for optional in optionals.values():
                if not optional=='NO':
                    print(optional, file=f)
                else:
                    pass
            print(calc_condition, file=f)
            print('', file=f)
            print('good luck!', file=f)
            print('', file=f)
            print(charge, file=f)
            for n, (x, y, z) in enumerate(conf.GetPositions()):
                atom = mol.GetAtomWithIdx(n)
                print('{}  {:.6f} {:.6f} {:.6f}'.format(atom.GetSymbol(), x, y, z), file=f)
            print('', file=f)