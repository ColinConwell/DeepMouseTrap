import numpy as np

from scipy import sparse
from scipy.optimize import OptimizeResult, nnls
from sklearn.utils import check_X_y
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model.base import LinearModel

class NonNegativeLinearRegression(LinearModel, RegressorMixin):
    """Non-negative least squares linear regression.
    Parameters
    ----------
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        estimated coefficients more robust and almost independent of the
        number of samples. The same property is not valid for standardized
        data. However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.
    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.
    Attributes
    ----------
    coef_ : array, shape (n_features, )
        Estimated coefficients for the linear regression problem.
    intercept_ : array
        Independent term in the linear model.
    opt_result_ : OptimizeResult
        Result of non-negative least squares optimization
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        
    def _rescale_data(X, y, sample_weight):
        """Rescale data so as to support sample_weight"""
        n_samples = X.shape[0]
        sample_weight = sample_weight * np.ones(n_samples)
        sample_weight = np.sqrt(sample_weight)
        sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                      shape=(n_samples, n_samples))
        X = safe_sparse_dot(sw_matrix, X)
        y = safe_sparse_dot(sw_matrix, y)
        return X, y

    def fit(self, X, y, sample_weight=None, zero_coef_alert=True):
        """Fit non-negative linear model.
        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : numpy array of shape [n_samples,]
            Target values
        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample
        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)

        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.coef_, result = nnls(X, y.squeeze())

        if np.all(self.coef_ == 0) and zero_coef_alert == True:
            raise ConvergenceWarning("All coefficients estimated to be zero in"
                                     " the non-negative least squares fit.")

        self._set_intercept(X_offset, y_offset, X_scale)
        self.opt_result_ = OptimizeResult(success=True, status=0, x=self.coef_,
                                          fun=result)
        return self

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, RepeatedKFold
from scipy.stats import pearsonr, spearmanr

def kfold_nonnegative_regression(target_rdm, model_rdms, regression_type='linear', n_splits=6, n_repeats=None, 
                              random_state=None, zero_coef_alert=False):
    '''
        target_rdm: your brain data RDM (n_samples x n_samples)
        model_rdms: your model layer RDMs (n_samples x n_samples x n_layers)
        standardize: whether to standardize features for regression (see NNLSRegression)
        n_splits: how many cross_validated folds
        random_state: used if you want to use a particular set of random splits
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
            regression = NonNegativeLinearRegression(fit_intercept=True, normalize=False)
            regression.fit(X_train, y_train, zero_coef_alert=zero_coef_alert)
        if regression_type == 'elastic_net':
            regression = ElasticNet(l1_ratio = 0, positive = True)
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