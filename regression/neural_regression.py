import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

### PLS Regression

def kfold_pls_regression(X, y, n_splits = 6, n_components = 25):
    scores, kfolds = [], KFold(n_splits, shuffle=False)
    for train_indices, test_indices in kfolds.split(np.arange(119)):
        train_X, test_X = X[train_indices, :], X[test_indices, :]
        train_y, test_y = y[train_indices], y[test_indices]
        regression = PLSRegression(n_components).fit(train_X, train_y)
        pred_y = regression.predict(test_X)
        scores.append(pearsonr(pred_y.squeeze(), test_y)[0]**2)
    return(np.mean(np.array(scores)))

def brain_data_pls_regression(feature_map, brain_data, n_components):
    score_sheet = []
    X = feature_map
    y_array = brain_data.transpose()
    for y_i, y in enumerate(tqdm(y_array, leave=0)):
        score = kfold_pls_regression(X,y,n_components=n_components)
        score_sheet.append({'neuron': y_i, 'score': score})
    return(pd.DataFrame(score_sheet))

# Ridge Regression

def kfold_ridge_regression(X, y, n_splits = 6):
    scores, kfolds = [], KFold(n_splits, random_state = 1)
    for train_indices, test_indices in kfolds.split(np.arange(119)):
        train_X, test_X = X[train_indices, :], X[test_indices, :]
        train_y, test_y = y[train_indices], y[test_indices]
        regression = Ridge(alpha=1.0).fit(train_X, train_y)
        pred_y = regression.predict(test_X)
        scores.append(pearsonr(pred_y.squeeze(), test_y)[0]**2)
    return(np.mean(np.array(scores)))

def brain_data_ridge_regression(feature_map, brain_data):
    score_sheet = []
    X = feature_map
    y_array = brain_data.transpose()
    for y_i, y in enumerate(tqdm(y_array, leave=0)):
        score = kfold_ridge_regression(X,y)
        score_sheet.append({'neuron': y_i, 'score': score})
    return(pd.DataFrame(score_sheet))