import os, sys
sys.path.append('../')
from .run import run_gaussian, chk_to_fchk
from .generate_gjf import GjfGenerator, change_calc_level_format
from mopac.generate_dats import DatsGenerator
from mopac.run import run_mopac
from rdkit import Chem
from rdkit.Chem import AllChem
from smiles.change_format import gen_alt_smiles
from rdkit.Chem import MolStandardize
from .load_calc_result import LoadCalcResultFromLogFile
from files.extension import ExtensionChanger
from files.log_to_smiles import log_to_smiles
from .recorder import CalcualationRecorder


def record_current_num(BASE_DIR, current_num, total_num):
    path = os.path.join(BASE_DIR, 'current_num.txt')
    with open(path, 'w') as f:
        print('now at : {}/{}'.format(current_num, total_num), file=f)
        f.close()

def record_calculated(BASE_DIR, obj):
    path = os.path.join(BASE_DIR, "job.txt")
    obj = os.path.basename(obj)
    with open(path, 'a') as f:
        print(obj, file=f)
        f.close()

def write_gjf_from_smiles(base_dir, smiles, calc_condition, charge_and_mult='0 1', connectivity=True, oldchk=None, **optionals):
    mol = Chem.MolFromSmiles(smiles)
    assert mol != None, 'Failed to convert Smiles to mol by RDkit!! Check the Smiles!!'
    alt_smiles = gen_alt_smiles(smiles)
    gjf_path = os.path.join(base_dir, alt_smiles + '_' + change_calc_level_format(calc_condition) +'.gjf')
    lfc = MolStandardize.fragment.LargestFragmentChooser()
    mol = lfc.choose(mol)
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

        with open(gjf_path, 'w') as f:
            for k, v in optionals.items():
                print('%{}={}'.format(k, v), file=f)
            if oldchk:
                print('%oldchk={}'.format(oldchk), file=f)
            print('%chk=' + alt_smiles + '.chk', file=f)
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
                lines = write_connectivity(smiles)
                f.write('\n'.join(lines))
            print(' ', file=f)
            f.close()
        return gjf_path
    except:
        assert True, 'Failed to Generate GJF file!! Something is wrong with Mol Object...'

def write_connectivity(smiles):
    """
    GaussianでGJFファイルを作るときに使う
    Smilesを入力する必要あり
    Gaussian上での結合がRDkit上で作成した分子と違うときに使うと良い
    ただし、Inputファイルにはgeom=connectivityを指定する必要あり
    計算結果が大きく変わる場合がある（スペクトル計算など）ので使用した方がいい
    """
    mol = Chem.MolFromSmiles(smiles)
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

            if len(neibors) == 1 and idx != 0:
                text_to_print = ' ' + str(print_idx)
            else:
                text_to_print = ' ' + str(print_idx)
                for atom_idx, bond in neibors_dict.items():
                    text_to_print = text_to_print + ' ' + str(atom_idx) + ' ' + str(bond)
        lines.append(text_to_print)
    return lines


class GaussianSequentialCalculation:
    def __init__(self, base_dir, smiles, calc_conditions, initial_gjf_path, charge_and_mult, fchk=True, **kwargs):
        """ gjfファイルからの連続計算

        Args
            base_dir: 計算結果を出力したいディレクトリ
            smiles: Smiles文字列
            calc_conditions: 計算条件を辞書型で指定
            initial_gjf_path: 最初に計算するgjfファイル
            charge_and_mult: 電荷と多重度
        
        Note
            1分子に対し、連続的に計算条件を指定して計算する
            ex)
            1段階目: # opt pm6
            2段階目: # opt b3lyo/6-31G(d)
            3段階目: # opt freq b3lyp/6-31G(d)
            ・・・
            n段階目: # opt td=(singlets Nstate=10) b3lyp/6-311G++(2d,p)
            - 仕様
            base_dir -> 計算を行うchkファイルが格納されているフォルダーを指定.計算はこのフォルダーで実行されるので計算結果の出力もここに出力される
            smiles -> 計算する化合物のSmilesを指定する.ファイル名に使用される
            calc_conditions -> 計算する分子の計算条件を辞書型で入力する.
            initial_gjf_path -> 最初に計算するgjfのファイルのパス

        Requirements
            base_dirで指定したファイル内で計算が行われる
            最初に計算するgjfファイルはあらかじめ作っている必要があり
            出力されるファイル名は計算条件で決まる
        
        Memo
            計算条件の指定がNprocsared, Mem, Oldchkぐらいしかできない
            10/8改定
            Gauusianの計算プロパティの項を可変長に変更
            これに伴い、Gaussianの計算プロパティはクラス呼び出しの際に引数として指定しないといけない
            Exp: GaussianSequentialCalculation(base_dir=..., Mem='6GB' Nprocshared='8')
        """
        self.base_dir = base_dir
        self.calc_conditions = calc_conditions.copy()
        self.fc_cache = []
        assert type(self.calc_conditions) == dict, 'you have to input calc_conditions as for dict type!!'
        self.gjf_generator = GjfGenerator(smiles=smiles)
        self.initial_gjf_path = initial_gjf_path
        self.charge_and_mult = charge_and_mult
        os.chdir(base_dir)
        self.run_gaussian(fchk, **kwargs)
    
    def run_gaussian(self, fchk, **kwargs):
        start_time = self.print_working_start_process(self.initial_gjf_path)
        run_gaussian(self.initial_gjf_path)
        self.print_end_time(start_time=start_time, gjf_path=self.initial_gjf_path)
        generated_chk = self.find_chkfile_from_gjffile(self.initial_gjf_path)
        if '.chk' not in generated_chk:
            generated_chk = generated_chk + '.chk'
        for mode, calc_condition in self.calc_conditions.items():
            if 'FrankCodonStep' in mode:
                fc_excited_chk = generated_chk
            elif 'FCProcess' in mode:
                self.fc_cache.append(generated_chk)
            else:
                fc_excited_chk = None
            if 'singlets' in calc_condition:
                state = 'excited'
            else:
                state = 'ground'
            generated_chk, generated_gjf = self.gjf_generator.write_gjf(self.base_dir, calc_condition, fc_excited_chk, state, self.charge_and_mult, oldchk=generated_chk, **kwargs)
            start_time = self.print_working_start_process(generated_gjf)
            run_gaussian(generated_gjf)
            self.print_end_time(start_time=start_time, gjf_path=generated_gjf)
            log_path = ExtensionChanger(generated_gjf).to_log()
            if fchk:
                chk_to_fchk(chk_filepath=generated_chk)
    
    def find_chkfile_from_gjffile(self, gjf_filepath):
        with open(gjf_filepath, 'r') as f:
            lines = f.readlines()
            chk_line = [line for line in lines if '%chk=' in line]
            assert len(chk_line) > 0 
            chk_filepath = chk_line[0]
            chk_filepath = chk_filepath[5:-1]
            f.close()
        return chk_filepath
    
    def print_working_start_process(self, gjf_path):
        import datetime
        start_time = datetime.datetime.now()
        print('---------------------------------------------------------------------------------------')
        print('Now calculating: ', os.path.basename(gjf_path))
        print('Start at {}'.format(start_time.strftime("%Y年%m月%d日 %H時%M分%S秒")))
        print('---------------------------------------------------------------------------------------')
        return start_time
    
    def print_end_time(self, start_time, gjf_path):
        import datetime
        elapsed = datetime.datetime.now() - start_time
        print('---------------------------------------------------------------------------------------')
        print('Calculation Finished!!', gjf_path)
        print('Elapsed Time: ', elapsed)
        print('---------------------------------------------------------------------------------------')

class GaussianSequentialCalculationFromSmiles(GaussianSequentialCalculation):
    def __init__(self, base_dir:str, smiles:str, calc_conditions:dict, charge_and_mult:str, connectivity=False, fchk=True, initial_conformation="MMFF", **link0_command):
        """Smilesを入力して連続計算を行う

        Args
            base_dir: 計算結果を出力するディレクトリ
            smiles: Smiles文字列
            calc_conditions: 辞書型で計算条件を記入
            charge_and_mult: 電荷と多重度
            connectivity: 結合を書くかどうか。Falseでいい
            initial_conformation: 初期配座の精製方法 UFF or MMFF
            link0_command: Gaussianのリンク0コマンド
        
        Returns 
            なし
        
        Note
            Smiles->molの変換が失敗するとエラーを吐く
            gjfファイルの作成に失敗するとエラーを吐く
            その他注意事項は継承元のGaussianSequentialCalculationを参考に
            - 注意事項
            使い方の注意
            !!! 計算条件は辞書型で入力 !!!
            MOPACの計算も入れたい場合はキーにMOPACを入れる
            MOPACを計算してGaussianに切り替える最初の計算ははMtoGをキーに入れる

        Example
            MOPACでPM7->Gaussian計算1->Gaussian計算2にした場合の計算記入例
            ------以下連続計算の入力例----------------------------------------------------------------------------------------------------
            calc_condition = {
                'MOPAC': 'PM7 ~~~計算条件~~~',
                'MtoG': '# opt b3lyp/6-31G(d) ~~~計算条件~~~',
                '1': '# freq b3lyp/6-31G(d) ~~~計算条件~~~',
                ・・・
                'n': '# td(Singlets, Nstates=10) b3lyp/6-31G(d) ~~~計算条件~~~',
            }
            GaussianSequentialCalculationFromSmiles(
                                                    base_dir="./計算結果を出力したいDirectory", 
                                                    smiles="c1cccc1",
                                                    calc_condition=calc_condition,
                                                    charge_and_mult='0 1' # 中性分子なら'0 1' ラジカルなら '0 2'
                                                    connecticity=True # ここは何でもいい
                                                    fchk=True, # chkがfchk化される
                                                    initial_conformation="UFF", # rdkitで最初に初期配座を生成するときのメソッド(MMFF or UFF)
                                                    Mem='6GB',          | ここから先はGaussianのlink0コマンド
                                                    Nprocshared='8',    | リンクコマンドとその値を入力
                                                                        | 今回の例だと gjfの先頭には%Mem=6GBと%Nproshared=8が刻まれる
                                                )
            -------連続計算の入力例終わり--------------------------------------------------------------------------------------------------

        Memo
            フランクコドン解析を行うファイルに何かしらのキーワードを入れて、キャッシュリストに保存してフランクコドン解析の時に使うとかいいかも...
            9/30追記:connectivityを追加、詳しくはwrite_connectivityを参照
            10/8改定
            Gauusianの計算プロパティの項を可変長に変更
            これに伴い、Gaussianの計算プロパティはクラス呼び出しの際に引数として指定しないといけない
            Exp: GaussianSequentialCalculation(base_dir=..., Mem='6GB' Nprocshared='8')
        """
        ### params ###
        self.base_dir = base_dir
        os.chdir(self.base_dir)
        self.smiles = smiles
        self.calc_conditions = calc_conditions
        self.charge_and_mult = charge_and_mult
        self.connectivity = connectivity
        self.initial_conformation = initial_conformation

        ### running ###
        self.main(**link0_command)
    
    def main(self, **link0_command):
        step = 1
        for k, v in self.calc_conditions.items():
            if step == 1:
                ### run mopac or gaussian in first step ###
                if 'MOPAC' in k:
                    dat_path = self.write_dat(calc_condition=v)
                    start_at = self.print_working_start_process(dat_path)
                    run_mopac(dat_path)
                    self.print_end_time(start_at, dat_path)
                else:
                    gjf_path = self.write_gjf(smiles=self.smiles, calc_condition=v, charge_and_mult=self.charge_and_mult, connectivity=self.connectivity, **link0_command)
                    generated_chk = ExtensionChanger(os.path.basename(gjf_path)).replacer('.chk')
                    start_at = self.print_working_start_process(gjf_path)
                    self.run_ganussian(gjf_path)
                    self.print_end_time(start_at, gjf_path)
            else:
                if 'MtoG' in k:
                    arc_path = ExtensionChanger(dat_path).replacer('.arc')
                    gjf_path = GjfGenerator(smiles=self.smiles).arc_to_gjf(base_dir=self.base_dir, arc_path=arc_path, calc_condition=v, connectivity=self.connectivity, **link0_command)
                    generated_chk = ExtensionChanger(os.path.basename(gjf_path)).replacer('.chk')
                    start_at = self.print_working_start_process(gjf_path)
                    self.run_ganussian(gjf_path)
                    self.print_end_time(start_at, gjf_path)
                else:
                    generated_chk, gjf_path = GjfGenerator(smiles=self.smiles).write_gjf(base_dir=self.base_dir, calc_level=v, charge_and_mult=self.charge_and_mult, connectivity=False, oldchk=generated_chk, **link0_command)
                    start_at = self.print_working_start_process(gjf_path)
                    self.run_ganussian(gjf_path)
                    self.print_end_time(start_at, gjf_path)
            step += 1
    
    def write_gjf(self, smiles, calc_condition, charge_and_mult, connectivity=True, **link0_command):
        generator = GjfGenerator(smiles)
        gjf_path = generator.write_gjf_from_smiles(base_dir=self.base_dir, smiles=self.smiles, calc_condition=calc_condition, charge_and_mult=charge_and_mult, connectivity=connectivity, mode=self.initial_conformation, **link0_command)
        return gjf_path
    
    def write_dat(self, calc_condition):
        generator = DatsGenerator(smiles=self.smiles, base_dir=self.base_dir, calc_condition=calc_condition, mode=self.initial_conformation)
        dat_path = generator.write_dat()
        return dat_path
    
    def run_ganussian(self, gjf_path):
        run_gaussian(gjf_path)
    
    def run_mopac(self, dat_path):
        run_mopac(dat_path)


def calculate_from_oldchk(smiles:str, base_dir:str, oldchk:str, calc_condition:str, charge_and_mult:str, **link0_command):
    """既存のchkファイルから計算を行う

    Args
        smiles: Smiles文字列。実在するSmilesである必要はないが、この入力がファイルの名前の先頭になる
        base_dir: chkファイルを出力したいディレクトリ
        oldchk: 既存のchkの名前. base_dirと同じ場所に設定
        calc_condition: 計算条件を入力
        charge_and_mult: 電荷と多重度
    
    Returns
        なし
    
    Note
        GaussianSequentialCalculationFromSmiles等を参考
    """
    alt_smiles = gen_alt_smiles(smiles)
    calc_condition_file_format = change_calc_level_format(calc_condition)
    gjf_path = os.path.join(base_dir, '{}_{}.gjf'.format(alt_smiles, calc_condition_file_format))
    chk_name = '{}_{}.chk'.format(alt_smiles, calc_condition_file_format)
    with open(gjf_path, 'w') as f:
        for k, v in link0_command.items():
            print('%{}={}'.format(k, v), file=f)
        print('%oldchk={}'.format(oldchk), file=f)
        print('%chk={}'.format(chk_name), file=f)
        print(calc_condition, file=f)
        print(' ', file=f)
        print('Good Luck!!', file=f)
        print(' ', file=f)
        print(' ' + charge_and_mult, file=f)
        print(' ', file=f)
        f.close()
    run_gaussian(gjf_path)

if __name__ == '__main__':
    GaussianSequentialCalculation(base_dir="./result", smiles="C1=CC=C(C=C1)O")