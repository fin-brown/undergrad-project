#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 17:31:54 2018

@author: finbrown1
"""

import numpy as np
import random

from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import _num_samples


class InnerCVSetup(BaseCrossValidator):
    
    """ Cross validation object for the inner CV phase.
        
        Leave-one-out validation for a specified number of iterations.
        
        Parameters
        ----------
        iters : number of iterations to run
        
        random_state : if specified will set the random seed before sampling.
    
        Returns
        -------
        CV object
    
    """
    
    def __init__(self, iters=3, random_state=None):

        self.iters = iters
        self.random_state = random_state
        
    def _iter_test_indices(self, X, y=None, groups=None):

        if self.random_state is not None:
            random.seed(self.random_state)
            
        n_samples = _num_samples(X)
        
        for i in random.sample(range(n_samples), self.iters):
            
            yield np.array(i)

    def get_n_splits(self, X, y=None, groups=None):

        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return int(self.iters)


class OuterCVSetup(_BaseKFold):
    
    """ Cross validation object for the outer CV phase.
    
        Adjusted K-fold such that no test set will have more than one record 
        per athlete ID, passed in as a group.
        
        Parameters
        ----------
        n_splits : number of folds
        
        random_state : random_state to be passed on to random.seed
        
        shuffle : not used, implemented for compatiblity with inheritance
        
        Returns
        -------
        CV object
    """
    
    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(OuterCVSetup, self).__init__(n_splits, shuffle, random_state)
            
        self.seed = random_state
        
    def _iter_test_indices(self, X, y=None, groups=None):  
        
        if self.seed is not None:
            random.seed(self.seed)
            
        n_samples = _num_samples(X)
        fold_size = n_samples // self.n_splits

        tests = [list() for _ in range(self.n_splits)]
        present_IDs = [set() for _ in range(self.n_splits)]
        
        for i in random.sample(range(n_samples), n_samples): 
        
            for j in random.sample(range(self.n_splits), self.n_splits):   
        
                if len(tests[j]) == fold_size or groups.iloc[i] in present_IDs[j]:
                    
                    continue
            
                tests[j].append(i)
                present_IDs[j].add(groups.iloc[i])
                
                break
         
        for test in tests:
            yield test
            
    def get_n_splits(self, X, y=None, groups=None):
        
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
            
        return int(self.n_splits)
    

def mse_variance(y, y_pred, **kwargs):
    
    """ MSE variance estimator. **kwargs for compatibility """
    
    m = _num_samples(y)
    
    mse_hat = mean_squared_error(y, y_pred)
    
    loss = (y - y_pred) ** 2
    
    mse_var = sum((mse_hat - loss) ** 2) / (m * (m-1))
    
    return mse_var








