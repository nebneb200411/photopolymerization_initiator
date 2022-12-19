import numpy as np
import pandas as pd
from scipy import integrate
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import Descriptors
import math
import os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append('../utils')

class LoadCalcResultFromLogFile:
    def __init__(self, log_file_path):
        """
        log_file_path -> log_file_path
        """
        self.log_file_path = log_file_path
    
    def get_lines(self):
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
            f.close()
        return lines
    
    def is_ended_normally(self):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
                final_line = lines[-1]
                if 'Normal termination of Gaussian' in final_line:
                    return True
                else:
                    return False
        except:
            return False
    
    def get_thermal_energy(self):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
                l_TE = [line.split() for line in lines if "Sum of electronic and thermal Energies=" in line][0]
                if l_TE:
                    TE = float(l_TE[-1])
                else:
                    TE = None
                f.close()
        except:
            TE = None
        
        return TE

    def get_GibbsFreeEnergy(self):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
                l_E = [line.split() for line in lines if "Sum of electronic and thermal Free Energies=" in line][0]
                if l_E:
                    Gibbs = float(l_E[-1])
                else:
                    Gibbs = None
                f.close()
        except:
            Gibbs = None
        
        return Gibbs
    
    def get_Enthalpy(self):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
                l_E = [line.split() for line in lines if "Sum of electronic and thermal Enthalpies=" in line][0]
                if l_E:
                    Enthalpies = float(l_E[-1])
                else:
                    Enthalpies = None
                f.close()
        except:
            Enthalpies = None
        
        return Enthalpies


    def get_TE(self, keyword):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
                l_PE = [line.strip() for line in lines if "E({})".format(keyword) in line][-1]
                start_pos = l_PE.find('=')
                start_pos += 1
                end_pos = l_PE.find('A')
                end_pos -= 1
                PE = l_PE[start_pos:end_pos]
                PE = float(PE)
        except:
            PE = None
            
        return PE
    
    def excited_prop(self):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
                excited_state_lines = [line for line in lines if 'Excited State' in line]
                excited_state_1 = [idx for idx, line in enumerate(excited_state_lines) if 'Excited State   1:' in line]
                last_excited_state_1_idx = excited_state_1[-1]
                target_lines = excited_state_lines[last_excited_state_1_idx:]

                fs = []
                lambdas = []

                for line in target_lines:
                    line = line.split()
                    lambda_idx = line.index('nm') - 1
                    lam = float(line[lambda_idx])
                    f = [i for i in line if 'f=' in i]
                    f = f[0]
                    f = f.strip('f=')
                    f = float(f)
                    if f == 0:
                        f = 1e-14
                    fs.append(f)
                    lambdas.append(lam)
        except:
            fs, lambdas = None
        
        return fs, lambdas

    def spectra(self):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()

                excited_state_lines = [line for line in lines if 'Excited State' in line]
                excited_state_1 = [idx for idx, line in enumerate(excited_state_lines) if 'Excited State   1:' in line]
                last_excited_state_1_idx = excited_state_1[-1]
                target_lines = excited_state_lines[last_excited_state_1_idx:]

                fs = []
                lambdas = []

                for line in target_lines:
                    line = line.split()
                    lambda_idx = line.index('nm') - 1
                    lam = float(line[lambda_idx])
                    f = [i for i in line if 'f=' in i]
                    f = f[0]
                    f = f.strip('f=')
                    f = float(f)
                    if f == 0:
                        f = 1e-14
                    fs.append(f)
                    lambdas.append(lam)
                
                try:
                    std = 0.00032262
                    wavelengths = np.arange(100, 800, 1)

                    epsilons = []

                    for i in range(len(fs)):
                        c = 1.3062974*(10**8) * (fs[i] / (std*(10**7)))
                        square = np.square((1/wavelengths - 1/lambdas[i])/std)
                        epsilon = c*np.exp(-square)
                        epsilons.append(epsilon)
                    epsilons_array = np.array(epsilons)
                    epsilons = np.sum(epsilons_array, axis=0)
                
                except:
                    wavelengths, epsilons = None, None

        except:
            wavelengths, epsilons = None, None
        
        return wavelengths, epsilons
    
    def is_normal_frequencies(self):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()

                frequencies_lines = [line for line in lines if 'Frequencies' in line]

                for l in frequencies_lines:
                    splited = l.split()
                    values = splited[2:]
                    values = [float(v) for v in values]
                    for v in values:
                        if v < 0:
                            f.close()
                            return False
                f.close()
            return True
        
        except:
            return False
    
    def get_calc_condition(self):
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
            calc_level = None
            for line in lines:
                try:
                    if line[1] == '#':
                        calc_level = line
                except:
                    pass
        return calc_level
    
    def spectra_integration(self, x_range=(350, 420), step=0.01):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()

                excited_state_lines = [line for line in lines if 'Excited State' in line]
                excited_state_1 = [idx for idx, line in enumerate(excited_state_lines) if 'Excited State   1:' in line]
                last_excited_state_1_idx = excited_state_1[-1]
                target_lines = excited_state_lines[last_excited_state_1_idx:]

                fs = []
                lambdas = []

                for line in target_lines:
                    line = line.split()
                    lambda_idx = line.index('nm') - 1
                    lam = float(line[lambda_idx])
                    f = [i for i in line if 'f=' in i]
                    f = f[0]
                    f = f.strip('f=')
                    f = float(f)
                    if f == 0:
                        f = 1e-14
                    fs.append(f)
                    lambdas.append(lam)
            integration = simpson_integration_spectra(x_range=x_range, lambdas=lambdas, fs=fs, step=step)

        except:
            integration = None
        
        return integration
        
    def UV_spectra(self, spectra_range=(100, 800), step=1):
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()

                excited_state_lines = [line for line in lines if 'Excited State' in line]
                excited_state_1 = [idx for idx, line in enumerate(excited_state_lines) if 'Excited State   1:' in line]
                last_excited_state_1_idx = excited_state_1[-1]
                target_lines = excited_state_lines[last_excited_state_1_idx:]

                fs = []
                lambdas = []

                for line in target_lines:
                    line = line.split()
                    lambda_idx = line.index('nm') - 1
                    lam = float(line[lambda_idx])
                    f = [i for i in line if 'f=' in i]
                    f = f[0]
                    f = f.strip('f=')
                    f = float(f)
                    if f == 0:
                        f = 1e-14
                    fs.append(f)
                    lambdas.append(lam)
                
                wavelengths = np.arange(min(spectra_range), max(spectra_range), step)
                epsilons = [UV_function(wav, lambdas, fs) for wav in wavelengths]

                df = pd.DataFrame()
                df['wavelength'] = wavelengths
                df['epsilon'] = epsilons

            return df

        except:
            return None

    def is_not_breaking(self):
        """openbabelを使って構造が壊れていないか判定する

        Returns
            壊れていたらFalse
            壊れていなかったらTrue
        """
        try:
            mol = next(pybel.readfile('log', self.log_file_path))
            smiles = mol.write(format="smi")
            smiles = smiles.split()[0].strip()
            if '.' in smiles:
                return False
            else:
                return True
        except:
            print('Failed to load {}'.format(self.log_file_path))
            return False
    
    def smiles(self):
        try:
            mol = next(pybel.readfile('log', self.log_file_path))
            smiles = mol.write(format="smi")
            smiles = smiles.split()[0].strip()
            return smiles
        except:
            print('Failed to load {}'.format(self.log_file_path))
            return None
    
    def osc_strength_range(self, search_range=(300, 450)):
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()

            excited_state_lines = [line for line in lines if 'Excited State' in line]
            excited_state_1 = [idx for idx, line in enumerate(excited_state_lines) if 'Excited State   1:' in line]
            last_excited_state_1_idx = excited_state_1[-1]
            target_lines = excited_state_lines[last_excited_state_1_idx:]

            oscs = []
            lambdas = []

            for line in target_lines:
                line = line.split()
                lambda_idx = line.index('nm') - 1
                lam = float(line[lambda_idx])
                osc = [i for i in line if 'f=' in i]
                osc = osc[0]
                osc = osc.strip('f=')
                osc = float(osc)
                oscs.append(osc)
                lambdas.append(lam)
            oscs = np.array(oscs)
            oscs = oscs[:, np.newaxis]
            lambdas = np.array(lambdas)
            lambdas = lambdas[:, np.newaxis]
            specs = np.concatenate([lambdas, oscs], 1)
            specs = specs[(specs[:, 0] > min(search_range)) & (specs[:, 0] < max(search_range))]
            specs = specs[specs[:, 1] > 0]

            if len(specs) > 0:
                return True
            else: 
                return False
        
    def get_spacial_distance(self):
        """openbabelとrdkitを使って発色団と開裂部位の距離を測る
        
        Note
            - 発色団の定義
            以下のようなルールで分ける
            ※以下のCはオキシムエステル部位のOC(=O)C-N=CのNを指す
            1.Cに結合している2つの部分構造の中に芳香環を含んでいるもの
            2.Cに結合している2つの部分構造においてどちらにも芳香環がある場合、芳香環の数が大きい方を採用
            3.Cに結合している2つの部分構造においてどちらにも芳香環があり、芳香環の分子量がどちらも同じ場合は、原子数の大きい方を採用
            4.Cに結合している2つの部分構造においてどちらにも芳香環がない場合は、分子量の大きい方を採用
            5.Cが環構造に含まれる場合は、
            1 ~ 3についてはCとの距離が最も近い芳香環内のCとの距離を測る
            4 ~ 5については目視で検討
        """
        mol = next(pybel.readfile('log', self.log_file_path))
        sdf = mol.write(format="sdf")
        mol = Chem.MolFromMolBlock(sdf)
        sub = Chem.MolFromSmiles(os.environ['OXIMESTER_SMILES'])
        subs = mol.GetSubstructMatches(sub)
        mol_xyz_list = '\n'.join(Chem.MolToXYZBlock(mol).splitlines()[2:])

        distances = []

        if not subs:
            sub = Chem.MolFromSmiles("C[N]OC(=O)")
            subs = mol.GetSubstructMatches(sub)
        print(subs)
        for sub in subs:
            # sub <- (a, b, c, d, e, f)
            N_atom_idx = [x for x in sub if mol.GetAtomWithIdx(x).GetSymbol() == 'N'][0]
            N_cordinate = mol_xyz_list.split('\n')[N_atom_idx]
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
                sub_chr = mol.GetSubstructMatches(chromophore)[0]
                atoms_info = [mol.GetAtomWithIdx(x) for x in sub_chr]
                aromatic_atoms = []
                for atom in atoms_info:
                    if atom.GetIsAromatic():
                        aromatic_atoms.append(atom.GetIdx())
                x1 = float(N_cordinate.split()[1])
                y1 = float(N_cordinate.split()[2])
                z1 = float(N_cordinate.split()[3])
                dis = []
                for number in aromatic_atoms:
                    xyz = mol_xyz_list.split('\n')[number]
                    x2 = float(xyz.split()[1])
                    y2 = float(xyz.split()[2])
                    z2 = float(xyz.split()[3])
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
                    dis.append(distance)
                distance = min(dis)
                distances.append(distance)
            else:
                """
                オキシムエステルのC末端が芳香環内にある場合の処理
                """
                distances.append(0)
        return distances
    
    def singlets_and_triplets_energies(self):
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
            lines = [line for line in lines if 'Excited State' in line]

            energies = {}
            energies['Singlets'] = []
            energies['Triplets'] = []

            for line in lines:
                if 'Singlet' in line:
                    line = line.split()
                    e = float(line[4])
                    energies['Singlets'].append(e)

                elif 'Triplet' in line:
                    line = line.split()
                    e = float(line[4])
                    energies['Triplets'].append(e)
                else:
                    pass
            f.close()
        return energies
    
    def singlets(self):
        """一重項の計算結果を辞書形式で取得

        Returns
            singlets_prp: energy->エネルギー wavelength->波長 oscilator: 振動子強度
        """
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
            lines = [line for line in lines if 'Excited State' in line]
            singlets_prop = {}
            singlets_prop['energy'] = [float(line.split()[4]) for line in lines if 'Singlet' in line]
            singlets_prop['wavelength'] = [float(line.split()[6]) for line in lines if 'Singlet' in line]
            singlets_prop['oscilator'] = [float(line.split()[8][2:]) for line in lines if 'Singlet' in line]
            f.close()
        return singlets_prop
    
    def triplets(self):
        """三重項の計算結果を辞書形式で取得

        Returns
            triplets_prp: energy->エネルギー wavelength->波長
        """
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
            lines = [line for line in lines if 'Excited State' in line]
            triplets_prop = {}
            triplets_prop['energy'] = [float(line.split()[4]) for line in lines if 'Triplet' in line]
            triplets_prop['wavelength'] = [float(line.split()[6]) for line in lines if 'Triplet' in line]
            f.close()
        return triplets_prop
    
    def get_energy_state(self, search_state):
        assert len(search_state) == 2, 'S1, S2, ..., Sn, T1, T2, ..., Tn can available'
        s_and_p = self.singlets_and_triplets()
        state = 'Singlets' if search_state[0] == 'S' else 'Triplets'
        index = int(search_state[1]) - 1
        energy = s_and_p[state][index]
        return energy

    def calc_condition(self):
        """計算条件の取得
        """
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
            target = [line for line in lines if '#' in line]
            target = [x for x in target if x[1] == '#']
            if target:
                target = target[0]
            else:
                target = ' Not Found '
            f.close()
        return target[1:-1]


def BDE_calculation(n_path, r1_path, r2_path):
    loader_n = LoadCalcResultFromLogFile(n_path)
    loader_r1 = LoadCalcResultFromLogFile(r1_path)
    loader_r2 = LoadCalcResultFromLogFile(r2_path)
    return -(loader_n.get_Enthalpy() - (loader_r1.get_Enthalpy() + loader_r2.get_Enthalpy())) * 627.51


def integrate_spectra_from_df(df, wavelength_column_name="wavelength", epsilon_column_name="epsilon", integral_range=(350, 420)):
    wavelengths = df[wavelength_column_name]
    wavelengths = np.array(wavelengths).astype('float32')
    epsilons = df[epsilon_column_name]
    epsilons = np.array(epsilons).astype('float32')

    array = np.stack([wavelengths, epsilons], 1)

    array = array[(integral_range[0] <= array[:, 0])&(array[:, 0] <= integral_range[1])]

    integrals = []
    condition = len(array) - 1
    for i in range(condition):
        # S = 1/2 * h * (a+b)
        h = array[i+1, 0] - array[i, 0]
        a = array[i, 1]
        b = array[i+1, 1]
        s = 0.5 * h * (a+b)
        integrals.append(s)
    integrals = np.array(integrals)
    result = np.sum(integrals)

    return result

def check_float(obj):
    dot_pos = obj.find('.')
    check = obj[dot_pos+1:]
    add = obj[:dot_pos+1]
    if len(check) < 8:
        while len(check) < 8:
            check = check + '0'
    elif len(check) > 8:
        check = check[:8]
    else:
        pass
    result = add + check
    return result

def UV_function(x, lambdas, fs):
    fs = np.array(fs)
    lambdas = np.array(lambdas)
    return np.sum(1.3062974*(10**8) * fs / 10**7 * 3099.6 * np.exp(-(((1/x)-1/lambdas)*3099.6)**2))

def simpson_integration_spectra(x_range, lambdas, fs, step=0.01):
    a, b = min(x_range), max(x_range)
    x_array = np.arange(a, b, step=step)
    y = []
    for x in x_array:
        delta_s = UV_function(x, lambdas, fs)
        y.append(delta_s)
    y = np.array(y)

    integral = integrate.simps(y, x_array)
    
    return integral


atomic_nums_and_symbols = [
    (1, 'H'),
    (2, 'He'),
    (3, 'Li'),
    (4, 'Be'),
    (5, 'B'),
    (6, 'C'),
    (7, 'N'),
    (8, 'O'),
    (9, 'F'),
    (10, 'Ne'),
    (11, 'Na'),
    (12, 'Mg'),
    (13, 'Al'),
    (14, 'Si'),
    (15, 'P'),
    (16, 'S'),
    (17, 'Cl'),
    (18, 'Ar'),
    (19, 'K'),
    (20, 'Ca'),
    (21, 'Sc'),
    (22, 'Ti'),
    (23, 'V'),
    (24, 'Cr'),
    (25, 'Mn'),
    (26, 'Fe'),
    (27, 'Co'),
    (28, 'Ni'),
    (29, 'Cu'),
    (30, 'Zn'),
    (31, 'Ga'),
    (32, 'Ge'),
    (33, 'As'),
    (34, 'Se'),
    (35, 'Br'),
    (36, 'Kr'),
    (37, 'Rb'),
    (38, 'Sr'),
    (39, 'Y'),
    (40, 'Zr'),
    (41, 'Nb'),
    (42, 'Mo'),
    (43, 'Tc'),
    (44, 'Ru'),
    (45, 'Rh'),
    (46, 'Pd'),
    (47, 'Ag'),
    (48, 'Cd'),
    (49, 'In'),
    (50, 'Sn'),
    (51, 'Sb'),
    (52, 'Te'),
    (53, 'I'),
    (54, 'Xe'),
    (55, 'Cs'),
    (56, 'Ba'),
    (57, 'La'),
    (58, 'Ce'),
    (59, 'Pr'),
    (60, 'Nd'),
    (61, 'Pm'),
    (62, 'Sm'),
    (63, 'Eu'),
    (64, 'Gd'),
    (65, 'Tb'),
    (66, 'Dy'),
    (67, 'Ho'),
    (68, 'Er'),
    (69, 'Tm'),
    (70, 'Yb'),
    (71, 'Lu'),
    (72, 'Hf'),
    (73, 'Ta'),
    (74, 'W'),
    (75, 'Re'),
    (76, 'Os'),
    (77, 'Ir'),
    (78, 'Pt'),
    (79, 'Au'),
    (80, 'Hg'),
    (81, 'Tl'),
    (82, 'Pb'),
    (83, 'Bi'),
    (84, 'Po'),
    (85, 'At'),
    (86, 'Rn'),
    (87, 'Fr'),
    (88, 'Ra'),
    (89, 'Ac'),
    (90, 'Th'),
    (91, 'Pa'),
    (92, 'U'),
    (93, 'Np'),
    (94, 'Pu'),
    (95, 'Am'),
    (96, 'Cm'),
    (97, 'Bk'),
    (98, 'Cf'),
    (99, 'Es'),
    (100, 'Fm'),
    (101, 'Md'),
    (102, 'No'),
    (103, 'Lr'),
    (104, 'Rf'),
    (105, 'Db'),
    (106, 'Sg'),
    (107, 'Bh'),
    (108, 'Hs'),
    (109, 'Mt'),
    (110, 'Ds'),
    (111, 'Rg'),
    (112, 'Cn'),
    (113, 'Nh'),
    (114, 'Fl'),
    (115, 'Mc'),
    (116, 'Lv'),
    (117, 'Ts'),
    (118, 'Og')
]