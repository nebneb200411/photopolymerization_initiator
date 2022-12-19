import datetime
from matplotlib import pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold, cross_val_score, validation_curve
import numpy as np
from tqdm.notebook import tqdm
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append('../')
from dataloader import dataloader, dataloader_rdkit

import params as P
from params import eval_method

def pretuning_abs(opt):
    SEED = int(os.environ['TRAINING_SEED'])
    model = XGBClassifier(booster='gbtree', objective='binary:logistic',
                     random_state=SEED, n_estimators=10000)

    ### first cross validation ###
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=SEED)
    x_train, _, y_train, _ = dataloader_rdkit(df_path=opt.df_path, y_column=opt.y_column)
    FIT_PARAMS = {
        'verbose': 0,
        'early_stopping_rounds': 10,
        'eval_metric': 'logloss',
        'eval_set': [(x_train, y_train)],
    }
    scoring = 'neg_log_loss'
    scores = cross_val_score(model, X=x_train, y=y_train, cv=cross_validation,
                            scoring=scoring, n_jobs=-1, fit_params=FIT_PARAMS)

    print('----first validation\'s score----')
    print('scores', scores)
    print('mean:{}'.format(np.mean(scores)))
    print('---------------------------------')

    ### tuning step ###
    #print(P.cv_params)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.subplots(4, 3)
    i, j = 0, 0
    for key, val in tqdm(P.cv_params.items()):
        train_scores, val_scores = validation_curve(estimator=model, X=x_train, y=y_train, param_name=key,
                                                    param_range=val, fit_params=FIT_PARAMS, cv=cross_validation, scoring=scoring, n_jobs=-1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        train_plus_sigma = train_mean + train_std
        train_minus_sigma = train_mean - train_std

        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        val_plus_sigma = val_mean + val_std
        val_minus_sigma = val_mean - val_std

        ax[i,j].plot(val, train_mean, color='blue', marker='o',
                markersize=5, label='training score')
        ax[i,j].fill_between(val, train_plus_sigma, train_minus_sigma,
                        alpha=0.15, color='blue')

        ax[i,j].plot(val, val_mean, color='green', linestyle='--',
                marker='o', markersize=5, label='validation score')
        ax[i,j].fill_between(val, val_plus_sigma, val_minus_sigma,
                        alpha=0.15, color='green')

        ax[i,j].set_xlabel(key)
        ax[i,j].set_ylabel(scoring)

        j += 1
        if j > 2:
            i += 1
            j = 0

    plt.legend(loc='lower right')
    plt.show()
    if not os.path.exists(('./xgboost_/result')):
        os.makedirs('./xgboost_/result', exist_ok=True)
    fig.savefig('./xgboost_/result/validation_at_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    plt.close()

def main(opt):
    objective, eval_metrics, scoring = eval_method(opt.eval_method)
    SEED = int(12)
    model = XGBRegressor(booster='gbtree', objective=objective,
                     random_state=SEED, n_estimators=10000)

    ### first cross validation ###
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=SEED)
    x_train, _, y_train, _ = dataloader_rdkit(df_path=opt.df_path, y_column=opt.y_column)
    FIT_PARAMS = {
        'verbose': 0,
        'early_stopping_rounds': 10,
        'eval_metric': eval_metrics,
        'eval_set': [(x_train, y_train)],
    }
    scoring = scoring
    scores = cross_val_score(model, X=x_train, y=y_train, cv=cross_validation,
                            scoring=scoring, n_jobs=-1, fit_params=FIT_PARAMS)

    print('----first validation\'s score----')
    print('scores', scores)
    print('mean:{}'.format(np.mean(scores)))
    print('---------------------------------')

    ### tuning step ###
    #print(P.cv_params)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.subplots(4, 3)
    i, j = 0, 0
    for key, val in tqdm(P.cv_params.items()):
        train_scores, val_scores = validation_curve(estimator=model, X=x_train, y=y_train, param_name=key,
                                                    param_range=val, fit_params=FIT_PARAMS, cv=cross_validation, scoring=scoring, n_jobs=-1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        train_plus_sigma = train_mean + train_std
        train_minus_sigma = train_mean - train_std

        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        val_plus_sigma = val_mean + val_std
        val_minus_sigma = val_mean - val_std

        ax[i,j].plot(val, train_mean, color='blue', marker='o',
                markersize=5, label='training score')
        ax[i,j].fill_between(val, train_plus_sigma, train_minus_sigma,
                        alpha=0.15, color='blue')

        ax[i,j].plot(val, val_mean, color='green', linestyle='--',
                marker='o', markersize=5, label='validation score')
        ax[i,j].fill_between(val, val_plus_sigma, val_minus_sigma,
                        alpha=0.15, color='green')

        ax[i,j].set_xlabel(key)
        ax[i,j].set_ylabel(scoring)

        j += 1
        if j > 2:
            i += 1
            j = 0

    plt.legend(loc='lower right')
    plt.show()
    if not os.path.exists(('./xgboost_/result')):
        os.makedirs('./xgboost_/result', exist_ok=True)
    fig.savefig('./xgboost_/result/validation_at_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_column", type=str, help="column")
    parser.add_argument("--df_path", type=str, help="データフレームのパス")
    parser.add_argument("--mode", type=str, help="choose absorbance or BDE")
    parser.add_argument("--eval_method", type=str, help="evaluate method")

    opt = parser.parse_args()

    if opt.mode == "BDE":
        main(opt)
    elif opt.mode == "abs":
        pretuning_abs(opt)
    else:
        sys.stderr, 'BDEもしくはabsを選んでください'



"""
model = XGBRegressor(booster='gbtree', objective='reg:squarederror',
                     random_state=P.SEED, n_estimators=10000)

### first cross validation ###
cross_validation = KFold(n_splits=5, shuffle=True, random_state=P.SEED)
x_train, x_test, y_train, y_test = dataloader()
FIT_PARAMS = {
    'verbose': 0,
    'early_stopping_rounds': 10,
    'eval_metric': 'rmse',
    'eval_set': [(x_train, y_train)],
}
scoring = 'neg_mean_squared_error'
scores = cross_val_score(model, X=x_train, y=y_train, cv=cross_validation,
                         scoring=scoring, n_jobs=-1, fit_params=FIT_PARAMS)

print('----first validation\'s score----')
print('scores', scores)
print('mean:{}'.format(np.mean(scores)))
print('---------------------------------')

### tuning step ###
#print(P.cv_params)
for i, (key, val) in tqdm(enumerate(P.cv_params.items())):
    train_scores, val_scores = validation_curve(estimator=model, X=x_train, y=y_train, param_name=key,
                                                param_range=val, fit_params=FIT_PARAMS, cv=cross_validation, scoring=scoring, n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    train_plus_sigma = train_mean + train_std
    train_minus_sigma = train_mean - train_std

    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    val_plus_sigma = val_mean + val_std
    val_minus_sigma = val_mean - val_std

    fig = plt.figure(figsize=(15, 15))
    ax = fig.subplots(3, 3)

    plt.plot(val, train_mean, color='blue', marker='o',
             markersize=5, label='training score')
    plt.fill_between(val, train_plus_sigma, train_minus_sigma,
                     alpha=0.15, color='blue')

    plt.plot(val, val_mean, color='green', linestyle='--',
             marker='o', markersize=5, label='validation score')
    plt.fill_between(val, val_plus_sigma, val_minus_sigma,
                     alpha=0.15, color='green')

    plt.xlabel(key)
    plt.ylabel(scoring)

    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('./results/{}_validation_at_{}.png'.format(key,
                                                           datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

plt.close()
"""