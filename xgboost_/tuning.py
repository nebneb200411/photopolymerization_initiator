from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
from .dataloader import dataloader, dataloader_rdkit
from sklearn.model_selection import KFold, cross_val_score, validation_curve
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from .params import eval_method
import optuna
import pickle
import time
import os
import sys
sys.path.append('../')
from xgboost_ import params as P
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
load_dotenv()

SEED = int(os.environ['TRAINING_SEED'])

def xgb_train_abs(cv_params, df_path, y_column):
    scoring = 'neg_log_loss'
    SEED = int(os.environ['TRAINING_SEED'])
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    start = time.time()
    model = XGBClassifier(booster='gbtree', objective='binary:logistic', random_state=SEED, n_estimators=10000)
    gridcv = GridSearchCV(model, cv_params, cv=cv, scoring=scoring, n_jobs=-1)

    x_train, x_test, y_train, y_test = dataloader(df_path=df_path, y_column=y_column)

    FIT_PARAMS = {
        'verbose': 0,
        'early_stopping_rounds': 10,
        'eval_metric': 'logloss',
        'eval_set': [(x_train, y_train)],
    }

    gridcv.fit(x_train, y_train, **FIT_PARAMS)

    best_params = gridcv.best_params_
    best_score = gridcv.best_score_
    print('best parameters{}\nscore {}'.format(best_params, best_score))
    print('time eplapsed{}sec'.format(time.time() - start))

    model = XGBClassifier(**best_params)
    model.fit(x_train, y_train)
    if not os.path.isdir('./xgboost_/model'):
        os.makedirs('./xgboost_/model', exist_ok=True)
    pickle.dump(model, open("./xgboost_/model/xgb_model_{}.pickle".format(y_column), "wb"))

    ### validation step ###
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    print('pred', y_pred)
    print('test', y_test.astype(np.int16))
    print(accuracy_score(y_pred=y_pred, y_true=y_test))

    df = pd.read_csv('./xgboost_/cache/test_{}.csv'.format(y_column))
    df['pred_{}'.format(y_column)] = y_pred
    df.to_csv('./xgboost_/cache/test_{}.csv'.format(y_column))

    print('========xgboost training done========')

def xgb_train(cv_params, df_path, y_column, method):
    objective, eval_metrics, scoring = eval_method(method)
    scoring = scoring
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    start = time.time()
    model = XGBRegressor(booster='gbtree', objective=objective, random_state=SEED, n_estimators=10000)
    gridcv = GridSearchCV(model, cv_params, cv=cv, scoring=scoring, n_jobs=-1)

    x_train, x_test, y_train, y_test = dataloader(df_path=df_path, y_column=y_column)

    FIT_PARAMS = {
        'verbose': 0,
        'early_stopping_rounds': 10,
        'eval_metric': eval_metrics,
        'eval_set': [(x_train, y_train)],
    }

    gridcv.fit(x_train, y_train, **FIT_PARAMS)

    best_params = gridcv.best_params_
    best_score = gridcv.best_score_
    print('best parameters{}\nscore {}'.format(best_params, best_score))
    print('time eplapsed{}sec'.format(time.time() - start))

    model = XGBRegressor(**best_params)
    model.fit(x_train, y_train)
    if not os.path.isdir('./xgboost_/model'):
        os.makedirs('./xgboost_/model', exist_ok=True)
    pickle.dump(model, open("./xgboost_/model/xgb_model_{}.pickle".format(y_column), "wb"))

    ### validation step ###
    y_pred = model.predict(x_test)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.subplots(1)

    r2 = r2_score(y_test, y_pred)
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('test data')
    ax.set_ylabel('predicted data')
    ax.text(1, 0.5, str(r2), fontsize="xx-large")
    ax.grid()
    if not os.path.isdir('./xgboost_/result'):
        os.makedirs('./xgboost_/result', exist_ok=True)
    fig.savefig('./xgboost_/result/validation.png')

    df = pd.read_csv('./xgboost_/cache/test_{}.csv'.format(y_column))
    df['pred_{}'.format(y_column)] = y_pred
    df.to_csv('./xgboost_/cache/test_{}.csv'.format(y_column))

    print('========xgboost training done========')

def optuna_tuning(df_path, y_column, method, task='regression'):
    start = time.time()
    # ベイズ最適化時の評価指標算出メソッド
    SEED = int(os.environ['TRAINING_SEED'])
    objective, eval_metrics, scoring = eval_method(method)
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    if task == 'regression':
        model = XGBRegressor(booster='gbtree', objective=objective, random_state=SEED, n_estimators=10000)
    elif task == 'classification':
        model = XGBClassifier(booster='gbtree', objective=objective, random_state=SEED, n_estimators=10000)
    else:
        sys.stderr, 'Choose regression or classification'
    x_train, x_test, y_train, y_test = dataloader_rdkit(df_path=df_path, y_column=y_column)
    fit_params = {'verbose': 0,  # 学習中のコマンドライン出力
                'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
                'eval_metric': eval_metrics,  # early_stopping_roundsの評価指標
                'eval_set': [(x_train, y_train)]  # early_stopping_roundsの評価指標算出用データ
                }
    
    def bayes_objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1.0),
            'subsample': trial.suggest_float('subsample', 0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
        }
        # モデルにパラメータ適用
        model.set_params(**params)
        # cross_val_scoreでクロスバリデーション
        scores = cross_val_score(model, x_train, y_train, cv=cv,
                                scoring=scoring, fit_params=fit_params, n_jobs=-1)
        val = scores.mean()
        return val

    # ベイズ最適化を実行
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(bayes_objective, n_trials=600)

    # 最適パラメータの表示と保持
    best_params = study.best_trial.params
    best_score = study.best_trial.value
    print(f'最適パラメータ {best_params}\nスコア {best_score}')
    print(f'所要時間{time.time() - start}秒')

    model = XGBRegressor(**best_params)
    model.fit(x_train, y_train)
    if not os.path.isdir('./xgboost_/model'):
        os.makedirs('./xgboost_/model', exist_ok=True)
    pickle.dump(model, open("./xgboost_/model/xgb_model_{}.pickle".format(y_column), "wb"))

    ### validation step ###
    y_pred = model.predict(x_test)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.subplots(1)

    r2 = r2_score(y_test, y_pred)
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('test data')
    ax.set_ylabel('predicted data')
    ax.text(0, 0, str(r2), fontsize="xx-large")
    ax.grid()
    if not os.path.isdir('./xgboost_/result'):
        os.makedirs('./xgboost_/result', exist_ok=True)
    fig.savefig('./xgboost_/result/validation.png')

    df = pd.read_csv('./xgboost_/cache/test_{}.csv'.format(y_column))
    df['pred_{}'.format(y_column)] = y_pred
    df.to_csv('./xgboost_/cache/test_{}.csv'.format(y_column))