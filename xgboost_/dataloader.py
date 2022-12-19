from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append('../')
from xgboost_ import params as P
import os
from dotenv import load_dotenv
load_dotenv()

def dataloader(df_path, y_column):
    df = pd.read_csv(df_path, index_col=0)
    x = df.iloc[:, 1:-1].values
    y = df[y_column].values
    smiles = df['Smiles'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=P.TEST_SIZE, random_state=int(os.environ['TRAINING_SEED']))
    smiles_train, smiles_test = train_test_split(smiles, test_size=P.TEST_SIZE, random_state=int(os.environ['TRAINING_SEED']))

    df_train = pd.DataFrame(data=x_train)
    df_train.insert(0, 'Smiles', smiles_train)
    df_train[y_column] = y_train
    df_test = pd.DataFrame(data=x_test)
    df_test.insert(0, 'Smiles', smiles_test)
    df_test[y_column] = y_test

    if not os.path.exists('./xgboost_/cache'):
        os.makedirs('./xgboost_/cache', exist_ok=True)
    df_train.to_csv('./xgboost_/cache/train_{}.csv'.format(y_column))
    df_test.to_csv('./xgboost_/cache/test_{}.csv'.format(y_column))

    print('shape of x_train: {}\n shape of x_test: {}\n shape of y_train: {}\n shape of y_test: {}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))
    
    return x_train, x_test, y_train, y_test

def dataloader_rdkit(df_path, y_column):
    df = pd.read_csv(df_path, index_col=0)
    x = df.iloc[:, 1:-1].values
    y = df[y_column].values
    smiles = df['Smiles'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=P.TEST_SIZE, random_state=int(os.environ['TRAINING_SEED']), shuffle=True)
    smiles_train, smiles_test = train_test_split(smiles, test_size=P.TEST_SIZE, random_state=int(os.environ['TRAINING_SEED']), shuffle=True)

    df_train = pd.DataFrame(data=x_train)
    df_train.insert(0, 'Smiles', smiles_train)
    df_train[y_column] = y_train
    df_test = pd.DataFrame(data=x_test)
    df_test.insert(0, 'Smiles', smiles_test)
    df_test[y_column] = y_test

    print('shape of x_train: {}\n shape of x_test: {}\n shape of y_train: {}\n shape of y_test: {}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    if not os.path.exists('./xgboost_/cache'):
        os.makedirs('./xgboost_/cache', exist_ok=True)
    df_train.to_csv('./xgboost_/cache/train_{}.csv'.format(y_column))
    df_test.to_csv('./xgboost_/cache/test_{}.csv'.format(y_column))
    
    return x_train, x_test, y_train, y_test

def dataloader_set_seed(seed):
    df = pd.read_csv(P.DF_PATH, index_col=0)
    x = df.iloc[:, 1:data_length-1].values
    y = df.iloc[:, data_length-1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=P.TEST_SIZE, random_state=seed)

    return x_train, x_test, y_train, y_test

