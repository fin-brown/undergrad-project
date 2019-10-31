#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:57:03 2018

@author: finbrown1
"""

import numpy as np
import pandas as pd

from data.transforms import ConvertLong2Wide

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

class RandomForestWideRegressor(BaseEstimator, RegressorMixin):

    """ RandomForest Regressor adapted to work with the data format.
        Takes n_estimators and max_features as params (see 
        sklearn.ensemble.RandomForests for details).
        
        Parameters
        ----------
        n_estimators : number of trees to use in the ensemble
        
        max_features : method for selecting number of features to use at each
            node split. As in sklearn.ensemble.RandomForestRegressor, takes:
                - auto : use all n_features
                - sqrt : use sqrt(n_features)
                - log2 : use log2(n_features)
    """
    
    def __init__(self, n_estimators=10, max_features="auto"):
        
        self.n_estimators = n_estimators
        self.max_features = max_features

        
    def fit(self, X, y):
        """ Fit method for the estimator. Converts the data into wide format.
        
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
        # fit the data in wide format
        self.wide_ = ConvertLong2Wide().fit_transform(X, y)
    
        return self
        
        
    def predict(self, X, y=None):
        
        """ Predict method for the regressor.
        
            This seems like the best valid application of RF for this problem.
            Fit 10 seperate RF models (1 for each feature-target split) and 
            pool the predictions. 
            
            Impute the median for missing values and only fit models on points
            where the target is a legitimate value rather than an imputation.
            
            Whilst the fitting of the component models should probably be in
            the fit method, only fitting models for events present in the test
            set saves computation and thus is done after seeing the test set.
            
            Parameters
            ----------
            X : the test data. Must have ID attribute.
            
            Returns
            -------
            y : predictions for X
        """    
        
        # create an empty series to fill (use series so we can update by index)
        y = pd.Series(np.NaN, index=X.ID)
               
        # calculate medians for imputation (as suggested by Breiman)
        fill_values = self.wide_.median()
        
        for i in set(X.Event):
            
            # make a list of columns we want as features
            columns = list(range(1, 11))
            columns.remove(i)
            
            # find rows where the target is not missing (thus won't be imputed)
            to_fit_ = ~np.isnan(self.wide_.loc[:, i])
                   
            # assign desired rows and columns to y_i, X_i
            y_i = self.wide_.loc[to_fit_, i]
            X_i = self.wide_.loc[to_fit_, columns].fillna(fill_values)
                
            # assign prediction set as rows where the event==i
            Xi = self.wide_.loc[X[X.Event==i].ID, columns].fillna(fill_values)
            
            # for small X we may have no instances of a particular event
            if len(Xi) != 0:
            
                # fit the model and update y with the predictions
                rf = RandomForestRegressor(n_estimators=self.n_estimators,
                                           max_features=self.max_features).fit(
                                                   X_i, y_i)
                            
                y.update(pd.Series(rf.predict(Xi), index=Xi.index))
      
        self.y = y.as_matrix()
        
        return self.y
                


# abstract wrapper method then pass parameters to gridsearch etc
# move fitting to fit. explain more about why 

