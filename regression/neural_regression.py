import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

def kfold_regression(regression, X, y, n_splits=6, score_by_fold=True, use_tqdm=True):
    pearsonr_vec = np.vectorize(pearsonr, signature='(n),(n)->(),()')
    scores, kfold = [], KFold(n_splits, random_state = 1)
    kfolds = kfold.split(np.arange(y.shape[0]))
    pred_y = np.zeros((y.shape[0],y.shape[1]))
    kfolds_tqdm = tqdm(kfolds, total = n_splits, leave=False)
    for train_indices, test_indices in kfolds_tqdm if use_tqdm else kfolds:
        train_X, test_X = X[train_indices, :], X[test_indices, :]
        train_y, test_y = y[train_indices], y[test_indices]
        regression = regression.fit(train_X, train_y)
        pred_y[test_indices] = regression.predict(test_X)
        
        if score_by_fold:
            test_y_ = test_y.transpose()
            pred_y_ = pred_y[test_indices].transpose()
            scores.append(pearsonr_vec(pred_y_, test_y_)[0]**2)
            
    if score_by_fold:
        return(np.mean(np.array(scores), axis = 0))
    if not score_by_fold:
        return(pearsonr_vec(pred_y.transpose(), y.transpose())[0]**2)
    

### PLS Regression

def kfold_pls_regression(X, y, n_components = 25, n_splits = 6, score_by_fold=True, use_tqdm=True):
    regression = PLSRegression(n_components = n_components)
    return kfold_regression(regression, X, y, n_splits, score_by_fold, use_tqdm)

# Ridge Regression

def kfold_ridge_regression(X, y, alpha = 1.0, n_splits = 6, score_by_fold=False, use_tqdm=True):
    regression = Ridge(alpha = alpha)
    return kfold_regression(regression, X, y, n_splits, score_by_fold, use_tqdm)