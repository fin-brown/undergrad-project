#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:42:28 2018

@author: finbrown1
"""

import numpy as np
import pandas as pd
import random

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from utils import index_nearest, int_distance_to_event

class MeanEstimator(BaseEstimator, RegressorMixin):
    
    """ Event mean predictor. """
                
    def fit(self, X, y):

        self.X_ = X
        self.y_ = y
        
        means = [0]*10
        
        for i in range(1, 11):
            
            means[i - 1] = np.mean(self.y_[self.X_.Event==i])               
        
        self._means = means
        return self

    
    def predict(self, X):
        
        self.X = X
        
        check_is_fitted(self, ["X", "X_", "y_", "_means"])
        
        self.y = [self._means[event - 1] for event in self.X.Event.tolist()]
        
        return self.y








