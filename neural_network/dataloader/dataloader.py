from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
load_dotenv()


def get_dataloader(batch_size, x, y, smiles, y_column, normalize=False):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=int(os.environ['TRAINING_SEED']))
    smiles_train, smiles_test = train_test_split(smiles, test_size=0.2, random_state=int(os.environ['TRAINING_SEED']))

    if normalize:
        y_mean, y_std = np.mean(y_train), np.std(y_train)
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

    df_train = pd.DataFrame(data=x_train)
    df_train.insert(0, 'Smiles', smiles_train)
    df_train[y_column] = y_train
    df_test = pd.DataFrame(data=x_test)
    df_test.insert(0, 'Smiles', smiles_test)
    df_test[y_column] = y_test

    df_train.to_csv('./neural_network/cache/train_{}.csv'.format(y_column))
    df_test.to_csv('./neural_network/cache/test_{}.csv'.format(y_column))

    transform = transforms.Compose([])

    dataset_train = FingerPrintDataSet(
        x_train,
        y_train,
        transforms=transform,
    )

    dataset_test = FingerPrintDataSet(
        x_test,
        y_test,
        transforms=transform,
    )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=len(x_test),
        shuffle=True
    )

    return dataloader_train, dataloader_test


class FingerPrintDataSet(Dataset):
    def __init__(self, x, y, transforms):
        super().__init__()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.x = transforms(self.Tensor(x))
        self.y = transforms(self.Tensor(y))

        self.len = len(x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return self.len


def get_dataloader_beta(df_path, batch_size):
    df = pd.read_csv(df_path, index_col=0)
    x = df.iloc[:, 1:4097].values
    y = df.iloc[:, [4097, 4099]].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    dataset_train = FingerPrintDataSetBeta(
        x_train,
        y_train[0],
        y_train[1],
    )

    dataset_test = FingerPrintDataSetBeta(
        x_test,
        y_test[0],
        y_test[1],
    )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=10,
        shuffle=True
    )

    return dataloader_train, dataloader_test


class FingerPrintDataSetBeta(Dataset):
    def __init__(self, x, BDE, abs, transforms):
        super().__init__()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.x = self.Tensor(x)
        self.BDE = self.Tensor(BDE)
        self.abs = self.Tensor(abs)

        self.len = len(x)

        #self.x = torch.Tensor(x, dtype=torch.float32)
        #self.y = torch.Tensor(y, dtype=torch.float32)

        self.transforms = transforms

    def __getitem__(self, index):
        x = self.x[index]
        #x = self.Tensor(x)
        if self.transforms:
            x = self.transforms(x)

        BDE = self.BDE[index]
        abs = self.abs[index]
        #y = self.Tensor(y)
        return x, BDE, abs

    def __len__(self):
        return self.len