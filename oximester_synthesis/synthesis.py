from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
import torch
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('../')
from dataframe.load import load_csv
from smiles.ketone_to_oximester import search_ketone, KetoneToOximester
from descriptor.generator import DescriptorGenerator
from neural_network.networks.dense_network import ABS_estimator, BDE_estimator

substituents = ['[O-][N+]=O', 'F', 'Cl', 'Br', 'I', '*OH', 'NH2']
use_col_bde = list(load_csv('./data/descriptors_absorbance_label.csv').columns)
use_col_abs = list(load_csv('./data/descriptors_bde.csv').columns)

class OximesterSynthesizer:
    def __init__(self, bde_model_path, abs_model_path):
        self.bde_model = self.open_xgboost_model(bde_model_path) if 'xgb' in bde_model_path else BDE_estimator().load_state_dict(torch.load(bde_model_path, map_location='cpu'))
        self.abs_model = self.open_xgboost_model(abs_model_path) if 'xgb' in abs_model_path else ABS_estimator().load_state_dict(torch.load(abs_model_path, map_location='cpu'))
        print(type(self.bde_model))
        print(type(self.abs_model))
    
    def pred_props(self, smiles, absorbance=None, bde=None, bde_range=(40, 45), use_mopac=False):
        """オキシムエステルの選定仮定

        Args
            bde_range: どのレンジのbdeを抽出するか
        
        Return
            df_fin -> 最終予測のdataframe
        """
        assert type(smiles) == list, 'input smiles must be list type!!'
        # absorbance descriptors
        g_abs = DescriptorGenerator(smiles)
        df_abs = g_abs.generator(use_mopac, absorbance, None, use_col=use_col_abs)
        # predict
        x_abs = df_abs.iloc[:, 1:]
        abs_pred = self.bde_model.predict(x_abs)
        abs_pred = np.where(abs_pred > 0.5, 1, 0)
        abs_pred = abs_pred.astype(np.int8)
        # choose pred=1
        df_abs['abs_pred'] = abs_pred
        df_selected = df_abs[df_abs['abs_pred'] == 1]
        # bde descriptors
        smiles = df_selected['Smiles'].values.tolist()
        g_bde = DescriptorGenerator(smiles)
        df_bde = g_bde.generator(use_mopac, absorbance, None, use_col=use_col_bde)
        # predict
        x_bde = df_bde.iloc[:, 1:]
        bde_pred = self.bde_model.predict(x_bde)
        df_bde['BDE_pred'] = bde_pred
        # choose smiles which predicted bde values in range
        df_fin = df_bde[(min(bde_range) < df_bde['BDE_pred'])&(df_bde['BDE_pred'] < max(bde_range))]
        df_ = pd.DataFrame()
        df_['Smiles'] = df_fin['Smiles']
        # saving
        dfs = {
            'abs': df_abs,
            'bde': df_bde,
            'pred': df_
        }
        for key, df in dfs.items():
            df.to_csv('./data/oximester_pred_{}.csv'.format(key))
        
        return df_
    
    def open_xgboost_model(self, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def extracter(self, smiles, molweight=100):
        """ 置換基を省いた化合物を選定

        Returns
            extracted: 抽出されたMolオブジェクトのリスト
        """
        extracted = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if search_ketone(s) and rdMolDescriptors._CalcMolWt(mol) > molweight and Descriptors.NumAromaticRings(mol) > 0:
                checker = []
                for i, sub in enumerate(substituents):
                    sub_mol = Chem.MolFromSmiles(sub)
                    if mol.HasSubstructMatch(sub_mol):
                        break
                    if i == len(substituents) - 1:
                        extracted.append(mol)
        return extracted
    
    def synthesis(self, smiles):
        df = self.pred_props(smiles)
        #smiles = df['Smiles'].values.tolist()
        #extracted_mols = self.extracter(smiles)
        return df
