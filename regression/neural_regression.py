import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

from sklearn.metrics import r2_score, explained_variance_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from scipy.stats import pearsonr

def anscombe_transform(x):
    return 2.0*np.sqrt(x + 3.0/8.0)

### Scoring Options ---------------------------------------------------------

pearsonr_vec = np.vectorize(pearsonr, signature='(n),(n)->(),()')

def pearson_r_score(y_true, y_pred, multioutput=None):
    y_true_ = y_true.transpose()
    y_pred_ = y_pred.transpose()
    return(pearsonr_vec(y_true_, y_pred_)[0])

def pearson_r2_score(y_true, y_pred, multioutput=None):
    return(pearson_r_score(y_true, y_pred)**2)

def get_predicted_values(y_true, y_pred, transform = None, multioutput = None):
    if transform == None:
        return(y_pred)

scoring_options = {'r2': r2_score, 'pearson_r': pearson_r_score, 'pearson_r2': pearson_r2_score,
                   'explained_variance': explained_variance_score, 'predicted_values': get_predicted_values}

def get_scoring_options():
    return scoring_options

def score_func(y_true, y_pred, score_type='pearson_r'):
    if not isinstance(score_type, list):
        return(scoring_options[score_type](y_true, y_pred, multioutput='raw_values'))
    
    if isinstance(score_type, list):
        scoring_dict = {}
        for score_type_i in score_type:
            scoring_dict[score_type_i] = scoring_options[score_type_i](y_true, y_pred, multioutput='raw_values')
        
    return(scoring_dict)

### Regression Methods ---------------------------------------------------------
        
def kfold_regression(X, y, regression, n_splits, score_type, use_tqdm):
    if regression == 'ridge':
        regression = Ridge(alpha = 1.0)
    if regression == 'pls':
        regression = PLSRegression(n_components = 10)
    if isinstance(regression, str) and regression not in ('ridge','pls'):
        assert "Unknown regression string. Please use one of ('ridge', 'pls') or an sklearn regression object."
    
    kfolds = KFold(n_splits, shuffle=False).split(np.arange(y.shape[0]))
    kfolds = tqdm(kfolds, total = n_splits, leave=False) if use_tqdm else kfolds
    
    y_pred = np.zeros((y.shape[0],y.shape[1]))
    for train_indices, test_indices in kfolds:
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        regression = regression.fit(X_train, y_train)
        y_pred[test_indices] = regression.predict(X_test)
            
    return score_func(y, y_pred, score_type)

def gcv_ridge_regression(X,y, score_type, alphas = [1.0], return_best_alpha = False):
    regression = RidgeCV(alphas=alphas, store_cv_values = True, 
                         scoring = 'explained_variance').fit(X,y)
    
    y_pred = regression.cv_values_.squeeze()
    
    if len(alphas) > 1: 
        best_alpha_index = 0
        current_best_score = 0
        for alpha, alpha_index in enumerate(alphas):
            y_pred = regression.cv_values_[:,:,alpha_index]
            score = score_func(y, y_pred, score_type).mean()
            if score > current_best_score:
                current_best_score = score
                best_alpha_index = alpha_index
        
        y_pred = regression.cv_values_[:,:,best_alpha_index]
            
    scores = score_func(y, y_pred, score_type)
    
    if not return_best_alpha:
        return scores
    if return_best_alpha:
        return (scores, regression.alpha_)
    
def neural_regression(feature_map, neural_response, regression = Ridge(alpha=1.0), cv_splits = 5,
                      score_type = 'pearson_r', use_tqdm = False, **kwargs):
    
    if cv_splits == 'gcv' and regression != 'ridge' and not isinstance(regression, Ridge):
        raise Warning("gcv mode selected, but regression is not ridge.")
        
    X,y = feature_map, neural_response
    
    if cv_splits is None:
        warnings.warn('No cv_splits selected. Returning fitted regression object...')
        return regression.fit(X,y)
    
    if cv_splits == 'gcv':
        return gcv_ridge_regression(X, y, score_type, **kwargs)
    
    if isinstance(cv_splits, int):
        return kfold_regression(X, y, regression, cv_splits, score_type, use_tqdm)
    
### Data Transforms ---------------------------------------------------------

def max_transform(df, group_vars, measure_var = 'score', deduplicate=True):
    if not isinstance(group_vars, list):
        group_vars = list(group_vars)
    
    max_df = (df[df.groupby(group_vars)[measure_var]
                 .transform(max) == df[measure_var]]).reset_index(drop=True)
                 
    if deduplicate:
        max_df = max_df[~max_df.duplicated(group_vars + [measure_var])]
        
    return max_df

def min_transform(df, group_vars, measure_var = 'score', deduplicate=True):
    if not isinstance(group_vars, list):
        group_vars = list(group_vars)
    
    min_df = (df[df.groupby(group_vars)[measure_var]
                 .transform(min) == df[measure_var]]).reset_index(drop=True)
                 
    if deduplicate:
        min_df = min_df[~min_df.duplicated(group_vars + [measure_var])]
        
    return min_df
