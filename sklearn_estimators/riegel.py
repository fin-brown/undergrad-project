#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:54:33 2018

@author: finbrown1
"""

import numpy as np
import pandas as pd
import random

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from utils import index_nearest, int_distance_to_event

class RiegelEstimator(BaseEstimator, RegressorMixin):
    
    """ A Reigel Power Law estimator for running performance. 
            
        Parameters
        ----------
        select_method : method for selecting how to sample from the available
            past records of a given athlete.
              - random : sample from the past records randomly
              - nearest : chose the nearest past record by event code, 
                  defaulting to the lower option if a tie between two.
        
        log : boolean flag to indicate whether the data has been log scaled.  
    """
            
    def __init__(self, riegel_exponent=1.06, sample_method="random", log=False):
        
        self.riegel_exponent = riegel_exponent
        self.sample_method = sample_method
        self.log = log
        
    def fit(self, X, y):
        
        """ Fit method for the estimator, performs simple checks.
            
            Parameters
            ----------
            X : features to be fitted, must have attributes ID and Distance. 
                Event will also be added if not given and is necessary.
            
            y : targets to be fitted. If self.log==True targets will be scaled
                back to standard scale for prediction.
                
            Returns
            -------
            self
        """
        
        if not hasattr(X, "ID") and not hasattr(X, "Distance"):
            raise AttributeError("X must have attributes ID and Distance")
        
        methods = {"random", "nearest", "average", "weighted_average"}
        if self.sample_method not in methods:
            raise ValueError("{} not in {}".format(self.sample_method, 
                                                   methods))

        self.X_ = X
        self.y_ = y if self.log is not True else np.exp(y)
          
        # if we want to use "nearest" check Event is defined
        if self.sample_method == "nearest":
            
            try: 
                self.X_.loc[:, "Event"] = X.Event
            
            except KeyError:                
                
                self.X_.loc[:, "Event"] = int_distance_to_event(
                        X.Distance.astype(int))
                    
        return self

    
    def predict(self, X):

        """ Predict method for the estimator. Uses Riegel's formula with a past 
            record selected from self.X_.
            
            Parameters
            ----------
            X : the test data. Must have attributes ID and Distance.
            
            Returns
            -------
            y : predictions for X
        """
        
        self.X = X
        
        check_is_fitted(self, ["X", "X_", "y_"])
        
        # create an empty dataframe to fill with things
        temp = pd.DataFrame(index=self.X.index)
        temp.loc[:, "ID"] = self.X.ID

        # define sEvent as index of sampled past Event
        if self.sample_method == "random":
            
            temp.loc[:, "sEvent"] = [random.sample(
                    set(self.X_[self.X_.ID==ID_].index), 1)[0] 
                    for ID_ in temp.ID]
        
        elif self.sample_method == "nearest":

            temp.loc[:, "Event"] = self.X.Event
            
            temp.loc[:, "sEvent"] = [index_nearest(
                    Event_, self.X_.Event[self.X_.ID==ID_]) 
                    for Event_, ID_ in list(zip(temp.Event, temp.ID))]
                
        # find t1, d1, d2 from sampled record and X
        temp.loc[:, "t1"] = self.y_[temp.sEvent].tolist()
        temp.loc[:, "d1"] = self.X_.Distance[temp.sEvent].tolist()
        temp.loc[:, "d2"] = self.X.Distance
        
        # convert to log space if required
        if self.log is not True:
            
            self.y = temp.t1 * (temp.d2/temp.d1) ** 1.06
            
        else:
            
            self.y = np.log(temp.t1 * (temp.d2/temp.d1) ** 1.06)
        
        return self.y 










