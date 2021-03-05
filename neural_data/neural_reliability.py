import numpy
from scipy.stats import pearsonr

#modified from: https://github.com/esdalmaijer/reliability
def splithalf_r(x, n_splits=100, mode='spearman-brown'):
    
    """Computes the split-half reliability, which speaks to the internal
    consistency of the measurement.
    
    Arguments
    
    x           -   A NumPy array with shape (M,N), where M is the number of
                    observations (trials) and N is the number of stimuli.
                    M will be split in half to compute the reliability, not N!
    
    Keyword Arguments
    
    n_splits    -   An integer that indicates the number of times you would
                    like to split the data in X. Default value is 100.
    
    mode        -   A string that indicates the type of split-half reliability.
                    You can choose from: 'correlate' or 'spearman-brown'.
                    Default value is 'spearman-brown'.
    
    Returns
    (r, sem)    -   r is the average split-half reliability over n_splits.
                    sem standard error of the mean split-half reliability.
    """
    
    # Check the input.
    if n_splits < 1:
        raise Exception("Expected n_splits to be 1 or more, not '%s'." % \
            (n_splits))
    allowed_modes = ['correlation', 'spearman-brown']
    if mode not in allowed_modes:
        raise Exception("Mode '%s' not supported! Please use a mode from %s" \
            % (mode, allowed_modes))
    
    # Get the number of trials per neuron, and the number of neurons.
    n_trials, n_neurons = x.shape
    
    # Compute the size of each group.
    n_half_1 = n_trials//2
    n_half_2 = n_trials - n_half_1
    # Generate a split-half-able vector. Assign the first half 1 and the
    # second half 2.
    halves = numpy.ones((n_trials, n_neurons), dtype=int)
    halves[n_half_1:, :] = 2
    
    # Run through all runs.
    r_ = numpy.zeros(n_splits, dtype=float)
    for i in range(n_splits):

        # Shuffle the split-half vector along the first axis.
        numpy.random.shuffle(halves)

        # Split the data into two groups.
        x_1 = numpy.reshape(x[halves==1], (n_half_1, n_neurons))
        x_2 = numpy.reshape(x[halves==2], (n_half_2, n_neurons))
        
        # Compute the averages for each group.
        m_1 = numpy.mean(x_1, axis=0)
        m_2 = numpy.mean(x_2, axis=0)
        
        # Compute the correlation between the two averages.
        pearson_r, p = pearsonr(m_1, m_2)

        # Store the correlation coefficient.
        if mode == 'correlation':
            r_[i] = pearson_r
        elif mode == 'spearman-brown':
            r_[i] = 2.0 * pearson_r / (1.0 + pearson_r)
    
    # Compute the average R value.
    r = numpy.mean(r_, axis=0)
    # Compute the standard error of the mean of R.
    sem = numpy.std(r_, axis=0) / numpy.sqrt(n_splits)
    
    return r, sem

def oracle_r(x, statistic = 'mean'):
    
    """Computes the oracle, or leave-one-out reliability, which measures how
    well one trial predicts the average of the rest.
    
    Arguments
    
    x           -   A NumPy array with shape (M,N), where M is the number of
                    trials and N is the number of neurons or neural sites.
                    M will be split in half to compute the reliability, not N!
    
    Keyword Arguments
    
    statistic   -   Statistic of central tendency to calculate across trials.
                    Only the 'mean' is implemented at the moment.
    
    Returns
    (r, sem)    -   r is the mean of oracle correlations across trials.
                    sem standard error of the mean oracle correlation.
    """
    
    # Get the number of trials per neuron, and the number of neurons.
    n_trials, n_neurons = x.shape
    
    # Iterate over all trials.
    r_ = numpy.zeros(n_trials, dtype=float)
    for i in range(n_trials):
        
        # Hold out one trial, and average the rest.
        x_out = x[i]
        x_avg = x[numpy.arange(len(x))!=i].mean(axis=0)

        # Correlate the held out trial with the average of the rest,
        # and append it to the running list of r values (r_).
        r_[i], p = pearsonr(x_out, x_avg)
    
    # Compute the average R value.
    r = numpy.nanmean(r_, axis=0)
    # Compute the standard error of the mean of R.
    sem = numpy.nanstd(r_, axis=0) / numpy.sqrt(n_trials)
    
    return r, sem
