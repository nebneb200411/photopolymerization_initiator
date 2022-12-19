import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from sklearn.metrics import r2_score
from networks.dense_network_rdkit_descriptor import DenseNetwork
from dataloader.dataloader import get_dataloader
from dotenv import load_dotenv
load_dotenv()
import os


class Trainer:
    def __init__(self, opt):
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device(
            str("cuda:0") if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.opt = opt

        ### data processing ###
        df = pd.read_csv(self.opt.df_path, index_col=0)
        self.x = df.iloc[:, 1:-1].values
        self.y = df[self.opt.y_column].values
        smiles = df['Smiles'].values

        if opt.normalize:
            y_mean = np.mean(self.y, axis=0, keepdims=True)
            y_std = np.std(self.y, axis=0, keepdims=True)
            self.y = (self.y - y_mean) / y_std
            print('平均:{}\n標準偏差:{}'.format(y_mean, y_std))
            if not os.path.exists('./neural_network/cache'):
                os.makedirs('./neural_network/cache', exist_ok=True)
            np.savez('./neural_network/cache/stats_{}.npz'.format(self.opt.y_column), y_mean, y_std)

        self.dataloder = get_dataloader(opt.batch_size, self.x, self.y, smiles, self.opt.y_column)
        
        self.network = DenseNetwork()

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=opt.learning_rate)

    def train(self):
        data_train, data_test = self.dataloder.__iter__()
        data_train = iter(data_train).next()

        x = data_train[0]
        y_true = data_train[1]
        
        self.network.train()
        self.optimizer.zero_grad()
        y_pred = self.network(x)
        y_pred = torch.squeeze(y_pred) 
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        #print(y_pred)
        #print(x)

        r2_train = r2_score(np.squeeze(y_true.to('cpu').detach().numpy().copy()), np.squeeze(y_pred.to('cpu').detach().numpy().copy()))

        loss = torch.mean(loss)

        self.network.eval()
        data_test = iter(data_test).next()

        x_test = data_test[0]
        y_test = data_test[1]
        y_pred = self.network(x_test)

        y_test = y_test.to('cpu').detach().numpy().copy()
        y_pred = y_pred.to('cpu').detach().numpy().copy()

        y_test = np.squeeze(y_test)
        y_pred = np.squeeze(y_pred)

        #print('Prediction:{} True:{}'.format(y_test[:5], y_true[:5]))

        r2_test = r2_score(y_test, y_pred)

        return loss, r2_train, r2_test