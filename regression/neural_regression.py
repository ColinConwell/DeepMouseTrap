import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

### PLS Regression

def kfold_pls_regression(X, y, n_splits = 6, n_components = 25, score_by_fold=True):
    kfolds, scores, pred_y = KFold(n_splits, shuffle=False), [], np.zeros(y.shape[0])
    for train_indices, test_indices in kfolds.split(np.arange(y.shape[0])):
        train_X, test_X = X[train_indices, :], X[test_indices, :]
        train_y, test_y = y[train_indices], y[test_indices]
        regression = PLSRegression(n_components).fit(train_X, train_y)
        pred_test_y = regression.predict(test_X).squeeze()
        scores.append(pearsonr(pred_test_y, test_y)[0]**2)
        pred_y[test_indices] = pred_test_y
    if score_by_fold:
        return(np.mean(np.array(scores)))
    if not score_by_fold:
        return(pearsonr(pred_y, y)[0]**2)

# Ridge Regression

def kfold_ridge_regression(X, y, n_splits = 6, score_by_fold=True):
    kfolds, scores, pred_y = KFold(n_splits, shuffle=False), [], np.zeros(y.shape[0])
    for train_indices, test_indices in kfolds.split(np.arange(y.shape[0])):
        train_X, test_X = X[train_indices, :], X[test_indices, :]
        train_y, test_y = y[train_indices], y[test_indices]
        regression = Ridge(alpha=1.0).fit(train_X, train_y)
        pred_test_y = regression.predict(test_X).squeeze()
        scores.append(pearsonr(pred_test_y, test_y)[0]**2)
        pred_y[test_indices] = pred_test_y
    if score_by_fold:
        return(np.mean(np.array(scores)))
    if not score_by_fold:
        return(pearsonr(pred_y, y)[0]**2)
