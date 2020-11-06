import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

from sklearn.metrics import r2_score, explained_variance_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

pearsonr_vec = np.vectorize(pearsonr, signature='(n),(n)->(),()')

def pearson_r2_score(y_true, y_pred, multioutput=None):
    y_true_ = y_true.transpose()
    y_pred_ = y_pred.transpose()
    return(pearsonr_vec(y_true_, y_pred_)[0]**2)

def get_predicted_values(y_true, y_pred, transform = None):
    if transform == None:
        return(y_pred)

scoring_options = {'r2': r2_score, 'pearson_r2': pearson_r2_score, 
                   'explained_variance': explained_variance_score, 
                   'predicted_values': get_predicted_values}

def get_scoring_options():
    return scoring_options

def score_func(y_true, y_pred, score_type='pearson_r2'):
    if not isinstance(score_type, list):
        return(scoring_options[score_type](y_true, y_pred, multioutput='raw_values'))
    
    if isinstance(score_type, list):
        scoring_dict = {}
        for score_type_i in score_type:
            scoring_dict[score_type_i] = scoring_options[score_type_i](y_true, y_pred, multioutput='raw_values')
        
    return(scoring_dict)
        
        
def kfold_regression(regression, X, y, n_splits, score_type = 'pearson_r2', return_preds = False, use_tqdm = True):
    kfolds = KFold(n_splits, shuffle=False).split(np.arange(y.shape[0]))
    kfolds_tqdm = tqdm(kfolds, total = n_splits, leave=False)
    
    y_pred = np.zeros((y.shape[0],y.shape[1]))
    for train_indices, test_indices in kfolds_tqdm if use_tqdm else kfolds:
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        regression = regression.fit(X_train, y_train)
        y_pred[test_indices] = regression.predict(X_test)
            
    scores = score_func(y, y_pred, score_type)
    
    if not return_preds:
        return(scores)
    if return_preds:
        return(scores, y_pred)
    

### PLS Regression

def kfold_pls_regression(X, y, n_components = 10, n_splits = 6, score_type = 'pearson_r2', return_preds = False, use_tqdm = True):
    return kfold_regression(PLSRegression(n_components = n_components), X, y, n_splits, score_type, return_preds, use_tqdm)

#### Ridge Regression

def kfold_ridge_regression(X, y, alpha = 1.0, n_splits = 6, score_type = 'pearson_r2', return_preds = False, use_tqdm = True):
    return kfold_regression(Ridge(alpha = alpha), X, y, n_splits, score_type, return_preds, use_tqdm)

def gcv_ridge_regression(X,y, alpha = 1.0, score_type = 'pearson_r2', return_preds = False):
    regression = RidgeCV(alphas=[alpha], store_cv_values = True, scoring = 'r2').fit(X,y)
    y_pred = regression.cv_values_.squeeze()
    scores = score_func(y, y_pred, score_type)
    if return_preds:
        return(scores, y_pred)
    if not return_preds:
        return(scores)