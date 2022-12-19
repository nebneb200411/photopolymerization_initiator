from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
load_dotenv()


def get_dataloader_rdkit(batch_size, x, y, smiles, y_column):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=int(os.environ['TRAINING_SEED']))
    smiles_train, smiles_test = train_test_split(smiles, test_size=0.2, random_state=int(os.environ['TRAINING_SEED']))

    df_train = pd.DataFrame(data=x_train)
    df_train.insert(0, 'Smiles', smiles_train)
    df_train[y_column] = y_train
    df_test = pd.DataFrame(data=x_test)
    df_test.insert(0, 'Smiles', smiles_test)
    df_test[y_column] = y_test

    df_train.to_csv('./neural_network/cache/train_{}.csv'.format(y_column))
    df_test.to_csv('./neural_network/cache/test_{}.csv'.format(y_column))

    transform = transforms.Compose([])

    dataset_train = RdkitDataSet(
        x_train,
        y_train,
        transforms=transform,
    )

    dataset_test = RdkitDataSet(
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

class RdkitDataSet(Dataset):
    def __init__(self, x, y, transforms):
        super().__init__()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.x = self.Tensor(x)
        self.y = transforms(self.Tensor(y))

        self.len = len(x)

        self.transforms = transforms

    def __getitem__(self, index):
        x = self.x[index]
        if self.transforms:
            x = self.transforms(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return self.len