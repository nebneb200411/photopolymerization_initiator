import os, sys
sys.path.append('../')
import pickle
import matplotlib.pyplot as plt
import numpy as np
from plot.graph import horizontal_bar

class XGBoostAnalyzer:
    def __init__(self, model_path):
        """ XGBoostモデルの解析

        Args
            model_path: モデルのパス

        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def get_FI(self, features):
        """ feature importancesの出力

        Args
            features: 特徴量のリスト
        
        Returns
            特徴量と特徴量のスコアの値
        """
        FI = self.model.feature_importances_
        feature_importances = {}
        assert len(features) == len(
            FI), 'feature importance score and features must be same length!! length of score: {} length of features: {}'.format(len(FI), len(features))
        for f, score in zip(features, FI):
            feature_importances[f] = score

        return feature_importances
    
    def plot_FI(self, features, figsize=(12, 4), bar_height=0.25, color='turquoise', save_path=None):
        fis = self.get_FI(features)
        labels = list(fis.keys())
        vals = list(fis.values())
        horizontal_bar(labels, vals, xlabel='Feature importances', bar_height=bar_height, figisize=figsize, color=color, save_path=save_path)
