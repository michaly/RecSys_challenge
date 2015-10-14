'''
Created on Dec 10, 2014

@author: nancy
'''


#scaling and transformation
import math
import numpy as np
from scipy.sparse import issparse
from scipy.stats import percentileofscore


def minMaxScaling(X, indices):
    
    """Standardizes features by scaling each feature to a [0,1] range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set.

    The standardization is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (maxRange - minRange) + minRange

    """
    # Desired range of transformed data. Default [0,1]
    minRange = 0.0
    maxRange = 1.0
    
    for index in indices:
     
        Y = X[:,index]
    
        # Compute the minimum and maximum to be used for later scaling.
        data_min = np.min(Y, axis=0)
        data_range = np.max(Y, axis=0) - data_min
        if(data_range == 0.0): data_range = 1.0 # Do not scale constant features
        scale_ = (maxRange - minRange) / data_range
        min_ = minRange - data_min * scale_
        
        # Scaling features of X according to feature_range.
        X[:,index] *= scale_
        X[:,index] += min_
        
    return X

def binarizationScaling(X, indices):
    
    """Boolean thresholding of array-like """
    
    isSparse = issparse(X)
    
    for index,threshold  in indices.iteritems():
        
        Y = X[:,index]
        
        if isSparse:
            cond = Y.data > threshold
            not_cond = np.logical_not(cond)
            Y.data[cond] = 1
            Y.data[not_cond] = 0
            Y.eliminate_zeros()
        else:
            cond = Y > threshold
            not_cond = np.logical_not(cond)
            Y[cond] = 1
            Y[not_cond] = 0
    
    return X
   
def percentileScaling(X, indices):
    
    """ 
    The percentile rank of a score relative to a list of scores.
    A percentileofscore of, for example, 80% means that 80% of the scores in a are below the given score
    """
    for index in indices:
        X[:,index] = [percentileofscore(X[:,index], i, kind='strict')/100 for i in X[:,index]]
    
    return X

def logTranformating(X, indices,alpha,beta):
    
    C = (-math.log(1-beta))/alpha # Natural basis
    
    for index in indices:
        X[:,index] = 1-np.exp(-C*X[:,index])
    
    return X

