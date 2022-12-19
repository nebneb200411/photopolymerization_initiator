from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
import sys, os, glob
import shutil
sys.path.append('../')
from mopac.calc import calc_from_smiles
from mopac.load_calc_result import LoadCalcResultFromOutFile
from files.extension import ExtensionChanger
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

class DescriptorGenerator:
    def __init__(self, smiles):
        """記述子作成
        Args
            smiles: smilesのリスト
        
        Returns
            df: dataframe 
        """
        assert type(smiles) == list, 'input smiles must be list type'
        self.smiles_list = smiles
    
    def generator(self, PM7_calc_condition, y, y_column, rads_smiles=None, use_col=None):
        if y:
            assert len(self.smiles_list) == len(y), 'Smiles length and target length must be same'
        mols = [Chem.MolFromSmiles(x) for x in self.smiles_list]
        descriptor_names = [descriptor_name[0] for descriptor_name in Descriptors._descList]
        descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        rdkit_descriptors_results = [descriptor_calculator.CalcDescriptors(mol) for mol in mols]
        df = pd.DataFrame(rdkit_descriptors_results, columns=descriptor_names)
        columns = df.columns
        if use_col:
            drop_col = [x for x in columns if x not in use_col]
            df = df.drop(drop_col, axis=1)
        
        if PM7_calc_condition:
            gaps = []
            base_dir = "./cache_mopac"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
            else:
                shutil.rmtree(base_dir)
                os.makedirs(base_dir, exist_ok=True)
                
            for s in tqdm(self.smiles_list):
                dats_path = calc_from_smiles(smiles=s, base_dir=base_dir, calc_condition=PM7_calc_condition)
                out_path = ExtensionChanger(dats_path).replacer('.out')
                loader = LoadCalcResultFromOutFile(out_path)
                if loader.is_normally_ended():
                    gaps.append(loader.HOMO_LUMO_gap())
                else:
                    gaps.append(None)
                for path in glob.glob(os.path.join(base_dir, '*')):
                    os.remove(path)
            shutil.rmtree(base_dir)
            df['HOMO-LUMO-Gap'] = gaps

        if rads_smiles and PM7_calc_condition:
            alpha_somo_lumo_gaps = []
            beta_somo_lumo_gaps = []
            somos = []
            for s in tqdm(rads_smiles):
                dats_path = calc_from_smiles(smiles=s, base_dir=base_dir, calc_condition=os.environ['MOPAC_RADICAL_CALC_CONDITION'])
                out_path = ExtensionChanger(dats_path).replacer('.out')
                loader = LoadCalcResultFromOutFile(out_path)
                if loader.is_normally_ended():
                    alpha_somo_lumo_gaps.append(loader.alpha_SOMO_LUMO_gap())
                    beta_somo_lumo_gaps.append(loader.beta_SOMO_LUMO_gap())
                    somos.append(loader.somo_level())
                else:
                    alpha_somo_lumo_gaps.append(None)
                    beta_somo_lumo_gaps.append(None)
                    somos.append(None)
                for path in glob.glob(os.path.join(base_dir, '*')):
                    os.remove(path)
            df['alpha-SOMO-LUMO-Gap'] = alpha_somo_lumo_gaps
            df['beta-SOMO-LUMO-Gap'] = beta_somo_lumo_gaps
            df['SOMO'] = somos
        df.insert(0, 'Smiles', self.smiles_list)
        if y:
            df[y_column] = y
            df.to_csv('./data/descriptors_{}.csv'.format(y_column))
        print('Descriptors generation has done!!')
        return df
    
    def mordred_descriptors(self, y, y_column):
        df = pd.DataFrame()
        df['Smiles'] = self.smiles_list
        df['mol'] = df['Smiles'].apply(Chem.MolFromSmiles)
        calc = Calculator(descriptors, ignore_3D=True)
        df_descriptors_mordred = calc.pandas(df['mol'])
        df_descriptors = df_descriptors_mordred.astype(str)
        masks = df_descriptors.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
        df_descriptors = df_descriptors[~masks]
        df_descriptors = df_descriptors.astype(float)
        df_descriptors.insert(0, 'Smiles', self.smiles_list)
        if y:
            df_descriptors[y_column] = y
        return df_descriptors

