import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from networks.dense_network import BDE_estimator, ABS_estimator
from dataloader.dataloader import get_dataloader
from dotenv import load_dotenv
load_dotenv()
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix

class Trainer:
    def __init__(self, opt):
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device(
            str("cuda:0") if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.opt = opt

        ### data processing ###
        df = pd.read_csv(self.opt.df_path, index_col=0)
        """
        if self.opt.normalize:
            columns = df.columns[1:]
            for col in columns:
                if len(df[col].unique()) != 2 and len(df[col].unique()) != 1:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
        """
        self.x = df.iloc[:, 1:-1].values
        self.y = df[self.opt.y_column].values
        smiles = df['Smiles'].values

        self.dataloder = get_dataloader(opt.batch_size, self.x, self.y, smiles, self.opt.y_column, normalize=self.opt.normalize)
        
        self.network = BDE_estimator(len(self.x[0])).to(self.device)

        self.criterion = nn.MSELoss()
        #self.criterion = nn.L1Loss()

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

        random_index = np.random.choice(range(len(y_true)))
        print(y_true[random_index], y_pred[random_index])

        r2_train = r2_score(np.squeeze(y_true.to('cpu').detach().numpy().copy()), np.squeeze(y_pred.to('cpu').detach().numpy().copy()))

        loss = torch.mean(loss)

        data_test = iter(data_test).next()

        x_test = data_test[0]
        y_test = data_test[1]
        y_pred = self.network(x_test)

        y_test = y_test.to('cpu').detach().numpy().copy()
        y_pred = y_pred.to('cpu').detach().numpy().copy()

        y_test = np.squeeze(y_test)
        y_pred = np.squeeze(y_pred)

        random_index = np.random.choice(range(len(y_test)))
        print(y_test[random_index], y_pred[random_index])

        r2_test = r2_score(y_test, y_pred)

        return loss, r2_train, r2_test


class ABSTrainer:
    def __init__(self, opt):
        """分類タスク
        """
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device(
            str("cuda:0") if torch.cuda.is_available() else "cpu")
        self.opt = opt
        ### data processing ###
        df = pd.read_csv(self.opt.df_path, index_col=0)
        self.x = df.iloc[:, 1:-1].values
        self.y = df[self.opt.y_column].values
        smiles = df['Smiles'].values

        self.dataloder = get_dataloader(opt.batch_size, self.x, self.y, smiles, self.opt.y_column)

        first_node = len(self.x[0])
        self.network = ABS_estimator(first_node, self.opt.use_dropout, self.opt.use_batchnorm).to(self.device)

        self.criterion = nn.BCELoss()

        self.optimizer = self.optimizer = optim.Adam(
            self.network.parameters(), lr=opt.learning_rate)

    def train(self, logger, test_interval=100):
        for epoch in tqdm(range(self.opt.epoch)):
            data_train, data_test = self.dataloder.__iter__()
            data_train = iter(data_train).next()

            x = data_train[0]
            label = data_train[1] 

            self.network.train()
            self.optimizer.zero_grad()
            y_pred = self.network(x)
            y_pred = torch.squeeze(y_pred)
            loss = self.criterion(y_pred, label)
            loss.backward()
            self.optimizer.step()

            #print(y_pred.size(), y_pred, label.size(), label)

            y_pred = torch.round(y_pred)
            accuracy_train = torch.sum(y_pred == label.data) / len(label) * 100

            if epoch%test_interval == 0 and epoch != 0:
                data_test = iter(data_test).next()
                x_test = data_test[0]
                label_test = data_test[1]
                y_pred_test = self.network(x_test)
                y_pred_test = torch.squeeze(y_pred_test)
                y_pred_test = torch.round(y_pred_test)
                accuracy_test = torch.sum(y_pred_test.data == label_test.data) / len(label_test) * 100
                y_pred_test = y_pred_test.to('cpu').detach().numpy().copy()
                y_pred_test = y_pred_test.astype(np.int8)
                label_test = label_test.to('cpu').detach().numpy().copy()
                label_test = label_test.astype(np.int8)
                logger.add_scalar('accuracy_test/', accuracy_test, epoch)
            logger.add_scalar('accuracy/', accuracy_train, epoch)
            logger.add_scalar('loss/', loss.item(), epoch)
        
        self.test()
    
    def test(self):
        _, data_test = self.dataloder.__iter__()
        data_test = iter(data_test).next()
        x_test = data_test[0]
        label_test = data_test[1]
        y_pred_test = self.network(x_test)
        y_pred_test = torch.squeeze(y_pred_test)
        y_pred_test = torch.round(y_pred_test)
        accuracy_test = torch.sum(y_pred_test.data == label_test.data) / len(label_test) * 100
        y_pred_test = y_pred_test.to('cpu').detach().numpy().copy()
        y_pred_test = y_pred_test.astype(np.int8)
        label_test = label_test.to('cpu').detach().numpy().copy()
        label_test = label_test.astype(np.int8)
        cm = confusion_matrix(y_pred=y_pred_test, y_true=label_test)
        print(y_pred_test, 'predicted')
        print(label_test, 'label test')
        print('-----test score-----\n', cm, '\n', '-------------', accuracy_test.data)