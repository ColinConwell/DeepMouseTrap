import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import KFold, RepeatedKFold
from scipy.stats import pearsonr, spearmanr

#### Standard Representational Similarity --------------------------------------------------

def compare_rdms(rdm1, rdm2, dist_type = 'pearson'):
    rdm1_triu = rdm1[np.triu_indices(rdm1.shape[0], k=1)]
    rdm2_triu = rdm2[np.triu_indices(rdm2.shape[0], k=1)]
    
    if dist_type == 'pearson':
        return pearsonr(rdm1_triu, rdm2_triu)[0]
    if dist_type == 'spearman':
        return spearmanr(rdm1_triu, rdm2_triu)[0]

#### Representational Similarity Regression --------------------------------------------------

def kfold_nonnegative_regression(target_rdm, model_rdms, regression_type='linear', 
                                 n_splits=10, n_repeats=None, random_state=None):
    '''Non-negative least squares linear regression on RDMs with k-fold cross-validation.
    Parameters
    ----------
    target_rdm: your brain data RDM (n_samples x n_samples)
    model_rdms: your model layer RDMs (n_samples x n_samples x n_layers)
    n_splits: how many cross_validated folds
    n_repeats: how many times to perform k-fold splits
    random_state: used if you want to use a particular set of random splits
    Attributes
    ----------
    r : correlation between predicted and actual RDM
    coefficients : the coefficients across k-fold splits
    intercepts : the intercepts across k-fold splits
    '''
    n_items = target_rdm.shape[0]
    
    predicted_rdm = np.zeros(target_rdm.shape)
    predicted_sum = np.zeros(target_rdm.shape)
    predicted_count = np.zeros(target_rdm.shape)

    coefficients = []
    intercepts = []
    i,j = np.triu_indices(target_rdm.shape[0],k=1)
    if n_repeats == None:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if n_repeats != None:
        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    for train_indices, test_indices in kf.split(list(range(n_items))):
        
        # indices for training and test cells of matrix
        test_idx = (np.isin(i, test_indices) | np.isin(j, test_indices))
        train_idx = ~test_idx       

        # target data (excluding test_indices)
        y_train = target_rdm[i[train_idx], j[train_idx]]

        # model data (excluding test_indices)
        X_train = model_rdms[i[train_idx], j[train_idx], :]

        # test data (test_indices)
        X_test = model_rdms[i[test_idx], j[test_idx], :]

        # fit the regression model
        if regression_type == 'linear':
            regression = LinearRegression(fit_intercept=True, positive=True)
            regression.fit(X_train, y_train)
        if regression_type == 'elastic_net':
            regression = ElasticNet(alpha = 1.0, l1_ratio = 0, positive = True)
            regression.fit(X_train, y_train)

        # predict the held out cells
        # note that for a k-fold procedure, some cells are predicted more than once
        # so we keep a sum and count, and later will average (sum/count) these predictions
        predicted_sum[i[test_idx],j[test_idx]] += regression.predict(X_test)        
        predicted_count[i[test_idx],j[test_idx]] += 1
        
        # save the regression coefficients
        coefficients.append(regression.coef_)
        intercepts.append(regression.intercept_)
    
    predicted_rdm = predicted_sum / predicted_count
    coefficients = np.stack(coefficients)
    intercepts = np.stack(intercepts)
    
    # make sure each cell received one value
    cell_counts = predicted_count[np.triu_indices(target_rdm.shape[0], k=1)]
    assert cell_counts.min()>=1, "A cell of the predicted matrix contains less than one value."
    
    # compute correlation between target and predicted upper triangle
    target = target_rdm[np.triu_indices(target_rdm.shape[0], k=1)]
    predicted = predicted_rdm[np.triu_indices(predicted_rdm.shape[0], k=1)]

    r = pearsonr(target, predicted)[0]
    
    return r, coefficients, intercepts

#### Data Transforms --------------------------------------------------

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