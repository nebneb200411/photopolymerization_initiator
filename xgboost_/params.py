TEST_SIZE = 0.2

# tuning params
cv_params = {
    # sampling data ratio xgboost is going to select
    'subsample': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    'colsample_bytree': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # L1 regularization term on weights. Increasing this value will make model more conservative
    'reg_alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    # L2 regularization term on weights. Increasing this value will make model more conservative
    'reg_lambda': [0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    # learning rate
    'learning_rate': [0.001, 0.01, 0.1, 0.3, 1.0],
    # Minimum sum of instance weight (hessian) needed in a child
    'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 , 14, 16, 18, 20],
    # max depth
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20],
    # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'gamma': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
}

cv_params_for_test = {
    # sampling data ratio xgboost is going to select
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
    'colsample_bytree': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # L1 regularization term on weights. Increasing this value will make model more conservative
    'reg_alpha': [0, 0.0001, 0.001, 0.01, 0.03, 0.1],
    # L2 regularization term on weights. Increasing this value will make model more conservative
    'reg_lambda': [0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0],
    # learning rate
    'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 1.0],
    # Minimum sum of instance weight (hessian) needed in a child
    'min_child_weight': [1, 3, 5, 7, 9],
    # max depth
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'gamma': [0, 0.0001, 0.001, 0.01, 0.03],
}

params_scale = {
    'subsample': 'linear',
    'colsample_bytree': 'linear',
    'reg_alpha': 'log',
    'reg_lambda': 'log',
    'learning_rate': 'log',
    'min_child_weight': 'linear',
    'max_depth': 'linear',
    'gamma': 'log'
}

SEEDS = [1, 28, 121, 290, 818, 1908, 2104, 4654, 5908, 10324, 11000, 14210, 189011, 20000, 29870, 29990, 30122, 32910, 330989, 
         333333, 34568, 34980, 35098, 36609, 37908, 38523, 38546, 38670, 39121, 40001, 41320, 41321, 42345, 46666, 50987, 50999, 
         60121, 61231, 62555, 63333, 63498, 63500, 64125, 65987, 66000, 67777, 67982, 67999, 68528, 68531, 69000, 69120, 70000, 70651]

import sys

def eval_method(method):
    if method == "RMSE":
        objective = 'reg:squarederror'
        eval_metrics = 'rmse'
        scoring = 'neg_mean_squared_error'
    elif method == "MAE":
        objective = 'reg:pseudohubererror'
        eval_metrics = 'mae'
        scoring = 'neg_mean_absolute_error'
    elif method == "RMSLE":
        objective = 'reg:squaredlogerror'
        eval_metrics = 'rmsle'
        scoring = 'neg_mean_absolute_error'
    elif method == 'classification':
        objective = 'binary:logistic'
        eval_metrics = 'logloss'
        scoring = 'neg_log_loss'
    else:
        sys.stderr, 'Choose RMSE, MAE or RMSLE'
    return objective, eval_metrics, scoring
