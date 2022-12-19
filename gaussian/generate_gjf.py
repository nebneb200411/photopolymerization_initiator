import os, sys
sys.path.append('../')
from smiles.change_format import gen_alt_smiles
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from files.extension import ExtensionChanger
from dotenv import load_dotenv
load_dotenv()

class GjfGenerator:
    def __init__(self, smiles):
        """ Gaussianの連続計算時のgjfファイルの作成
        Args
            smiles: Smiles文字列
        
        Note
            base_dirに指定したディレクトリにgjfファイルを作成
            calc_levelに計算の条件を指定 ex) # opt freq b3lyp/6-31G(d)
            fc_chkはフランクコドン解析を行うときに記入する励起状態のchkファイルの名前
            stateは計算する分子の状態、書き方はなんでもいい ex) 基底状態の分子を計算するなら->state=ground、励起状態ならstate=excited
            charge_and_mult->電荷と多重度
            **kwargs 計算のオプションを記入 ex) Mem='6GB' Nprocshared='8' ※指定する因数はstrである必要がある 
        """
        self.smiles = smiles
        self.filename = gen_alt_smiles(smiles)
    
    def write_gjf(self, base_dir, calc_level, fc_chk=None, charge_and_mult='0 1', connectivity=True, oldchk=None, **kwargs):
        """ oldchkから計算を行う用のchkファイルを作成

        Args
            base_dir: gjfファイルを出力するディレクトリ
            calc_level: 計算条件
            charge_and_mult: 電荷と多重度
            oldchk: 情報を継承するchkファイル
        
        Returns
            generated_chk: 作られたchkファイルの名前
            gjf_path: gjfファイルのパス
        
        Note

        """
        filename = change_calc_level_format(calc_level)
        generated_chk = '{}_{}.chk'.format(self.filename, filename)
        gjf_path = os.path.join(base_dir, '{}_{}.gjf'.format(self.filename, filename))
        with open(gjf_path, 'w') as f:
            for k, v in kwargs.items():
                print('%{}={}'.format(k, v), file=f)
            if oldchk:
                print('%oldchk={}'.format(oldchk), file=f)
            print('%chk={}'.format(generated_chk), file=f)
            print(calc_level, file=f)
            print('', file=f)
            print('Good Luck!!', file=f)
            print('', file=f)
            if fc_chk:
                print(fc_chk, file=f)
                print('', file=f)
            print(charge_and_mult, file=f)
            print('', file=f)
            if connectivity:
                lines = self.write_connectivity()
                f.write('\n'.join(lines))
            print('', file=f)
            f.close()
        return generated_chk, gjf_path
    
    def write_connectivity(self):
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol)

        atoms = mol.GetNumAtoms()
        atoms_idx = [x for x in range(atoms)]

        lines = []
        for idx in atoms_idx:
            atom = mol.GetAtomWithIdx(idx)
            print_idx = idx + 1
            bonds = atom.GetBonds()
            neibors = atom.GetNeighbors()
            neibors_dict = {}
            for atom, bond in zip(neibors, bonds):
                neibor_atom_idx = atom.GetIdx() + 1
                bond_type = bond.GetBondTypeAsDouble()
                if neibor_atom_idx > print_idx:
                    neibors_dict[str(neibor_atom_idx)] = bond_type

                if len(neibors) == 1:
                    text_to_print = ' ' + str(print_idx)
                else:
                    text_to_print = ' ' + str(print_idx)
                    for atom_idx, bond in neibors_dict.items():
                        text_to_print = text_to_print + ' ' + str(atom_idx) + ' ' + str(bond)
            lines.append(text_to_print)
        return lines
    
    def write_gjf_from_smiles(self, smiles, base_dir, calc_condition, charge_and_mult, oldchk=False, connectivity=True, mode='MMFF', **link0_command):
        """ Smiles文字列からgjfファイルを作成

        Args
            smiles: Smiles文字列
            base_dir: gjfファイルを出力するディレクトリ
            calc_condition: 計算条件が出力されるディレクトリ
            charge_and_mult: 電荷と多重度
            link0_command: GaussianのLink0 Command 参考: https://gaussian.com/link0/
        
        Return
            gjf_path: 作ったgjfのパス
        
        Note
        """
        mol = Chem.MolFromSmiles(smiles)
        assert mol != None, 'Failed to convert Smiles to mol by RDkit!! Check the Smiles!!'
        alt_smiles = gen_alt_smiles(smiles)
        gjf_path = os.path.join(base_dir, '{}_{}_{}.gjf'.format(alt_smiles, change_calc_level_format(calc_condition), mode))
        chk_path = '{}_{}_{}.chk'.format(alt_smiles, change_calc_level_format(calc_condition), mode)
        lfc = MolStandardize.fragment.LargestFragmentChooser()
        mol = lfc.choose(mol)
        energy = []
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol) # 3次元構造化
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=1000, randomSeed=1234,
                                    pruneRmsThresh=2, numThreads=0)
        if mode == 'MMFF':
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

        with open(gjf_path, 'w') as f:
            for k, v in link0_command.items():
                print('%{}={}'.format(k, v), file=f)
            if oldchk:
                print('%oldchk={}'.format(oldchk), file=f)
            print('%chk={}'.format(chk_path), file=f)
            print(calc_condition, file=f)
            print('', file=f)
            print('good luck!', file=f)
            print('', file=f)
            print(charge_and_mult, file=f)
            for n, (x, y, z) in enumerate(conf.GetPositions()):
                atom = mol.GetAtomWithIdx(n)
                print(' {}  {:.6f} {:.6f} {:.6f}'.format(atom.GetSymbol(), x, y, z), file=f)
            print('', file=f)
            if connectivity:
                lines = self.write_connectivity()
                f.write('\n'.join(lines))
            print(' ', file=f)
            f.close()
        return gjf_path

    def arc_to_gjf(self, base_dir, arc_path, calc_condition, connectivity=True, **kwargs):
        assert os.path.exists(arc_path), 'dat file not exists!!'
        with open(arc_path, 'r') as f:
            lines = f.readlines()
            title_index = [idx for idx, x in enumerate(lines) if os.environ['MOPAC_TITLE'] in x][-1]
            xyz = lines[title_index+1:-1]
            xyz = [x.split() for x in xyz]
            # ['atom', x_position, charge, y_position, charge, z_position]
            f.close()
        
        
        gjf_path = os.path.join(base_dir, '{}_{}.gjf'.format(gen_alt_smiles(self.smiles), change_calc_level_format(calc_condition)))
        chk_path = '{}_{}.chk'.format(gen_alt_smiles(self.smiles), change_calc_level_format(calc_condition))

        with open(gjf_path, 'w') as f:
            for k, v in kwargs.items():
                print('%{}={}'.format(k, v), file=f)
            print('%chk={}'.format(chk_path), file=f)
            print(calc_condition, file=f)
            print('', file=f)
            print('Good Luck!!', file=f)
            print('', file=f)
            for p in xyz:
                print(' {}  {} {} {}'.format(p[0], p[1], p[3], p[5]), file=f)
            print('', file=f)
            if connectivity:
                lines = self.write_connectivity()
                f.write('\n'.join(lines))
            print(' ', file=f)
            f.close()

        return gjf_path

def change_calc_level_format(calc_level):
    calc_level = calc_level.replace(' ', '_')
    calc_level = calc_level.replace('/', '_')
    calc_level = calc_level.replace(',', '&')
    return calc_level

def path_generator(smiles, calc_level, extension='gjf'):
    extension = '.' + extension if '.' not in extension else extension
    return '{}_{}{}'.format(gen_alt_smiles(smiles), calc_level, extension)
