#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:16:56 2018

@author: finbrown1
"""

import numpy as np
import pandas as pd
import random

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from utils import index_nearest, int_distance_to_event, DISTANCES

# Potuguese Table of Speeds
PTS = np.array([(40.0,11.0000), (50.0,10.9960), (60.0,10.9830), 
                (70.0,10.9620), (80.0,10.934), (90.0,10.9000),
                (100.0,10.8600), (110.0,10.8150), (120.0,10.765),
                (130.0,10.7110), (140.0,10.6540), (150.0,10.5940),
                (160.0,10.531), (170.0,10.4650), (180.0,10.3960),
                (200.0,10.2500), (220.0,10.096), (240.0,9.9350),
                (260.0,9.7710), (280.0,9.6100), (300.0,9.455),
                (320.0,9.3070), (340.0,9.1660), (360.0,9.0320), 
                (380.0,8.905), (400.0,8.7850), (450.0,8.5130),
                (500.0,8.2790), (550.0,8.083), (600.0,7.9210),
                (700.0,7.6690), (800.0,7.4960), (900.0,7.32000),
                (1000.0,7.18933), (1200.0,6.98066), (1500.0,6.75319),
                (2000.0,6.50015), (2500.0,6.33424), (3000.0,6.21913), 
                (3500.0,6.13510), (4000.0,6.07040), (4500.0,6.01822), 
                (5000.0,5.97432), (6000.0,5.90181), (7000.0,5.84156), 
                (8000.0,5.78889), (9000.0,5.74211), (10000.0,5.70050), 
                (12000.0,5.62944), (15000.0,5.54300), (20000.0,5.43785),
                (25000.0,5.35842), (30000.0,5.29298), (35000.0,5.23538),
                (40000.0,5.18263), (50000.0,5.08615)])

# constants for standard time formula
(C1, C2, C3) = (.20, .08, .0065)

# helper functions for implementing scoring algorithm   
def a(v): return 85 / (.0654 - .00258 * v)

def b(v): return 1 - 950 / a(v)   

def f(d):

    """ f(d) as defined in [Computer Generated Track Scoring Tables.
        Takes and returns a pandas series.
    """
    
    f = pd.Series(index=d.index)
    
    for i in range(len(d)):
    
        if d.iloc[i] < 110:
            
            f.iloc[i] = 0
            
            continue
        
        l = d.iloc[i] // 400
        m = d.iloc[i] - 400 * l
        
        if m < 50: 
            pl = 0
        elif m < 150: pl = m - 50
        elif m < 250: pl = 100
        elif m < 350: pl = m - 150
        else: pl = 200
        
        tm = l * 200 + pl
        
        f.iloc[i] = tm / d.iloc[i]

    return f


def v(d):

    """ Linear interpolation of the two values in PTS bounding d. 
        Takes and returns pandas series.
    """
    
    v = pd.Series(index=d.index)
    
    for i in range(len(d)):
        
        try:
            v.iloc[i] = float(PTS[np.where(PTS == d.iloc[i])[0], 1])
        
        except:
            # find 2 closest distances in the table
            du = PTS[PTS > d.iloc[i]].min()
            dl = PTS[PTS < d.iloc[i]].max()
    
            vu = PTS[np.where(PTS == du)[0], 1]
            vl = PTS[np.where(PTS == dl)[0], 1]
            
            v.iloc[i] = float(vl + (d.iloc[i] - dl) * (vu - vl) / (du - dl))
  
    return v

# calculates 950 point times for input d
def t950(d): return d / v(d) + C1 + C2 * v(d) + C3 * f(d) * v(d) ** 2


class PurdyEstimator(BaseEstimator, RegressorMixin):

    """ A Purdy Points equivalent score estimator for running performance. 
    
        Parameters
        ----------
        select_method : method for selecting how to sample from the available
            past records of a given athlete.
              - random : sample from the past records randomly
              - nearest : chose the nearest past record by event code, 
                  defaulting to the lower option if a tie between two.
        
        log : boolean flag to indicate whether the data has been log scaled.                
    """
    
    def __init__(self, sample_method="random", log=False):
        
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
        # transform back to standard scale if self.log==True
        self.y_ = y if self.log is not True else np.exp(y)
                
        # if we want to use "nearest" make sure we have event column
        if self.sample_method == "nearest":
            
            try: 
                self.X_.loc[:, "Event"] = X.Event
            
            except KeyError:                
                
                self.X_.loc[:, "Event"] =int_distance_to_event(X.Distance.astype(int))
            
        return self


    def predict(self, X):
        
        """ Predict method for the estimator. Uses Purdy points equivalent
            scoring from a past record selected from self.X_.
            
            Parameters
            ----------
            X : the test data. Must have attributes ID and Distance.
            
            Returns
            -------
            y : predictions for X
        """
        
        self.X = X
        
        check_is_fitted(self, ["X", "X_", "y_"])
        #check_array(X)        
        
        # instantiate an empty dataframe to fill with various things
        temp = pd.DataFrame(index=self.X.index)
        temp.loc[:, "ID"] = self.X.ID

        # define sEvent as index of sampled past Event
        if self.sample_method == "random":
            
            temp.loc[:, "sEvent"] = [random.sample(set(self.X_[self.X_.ID==ID_].index), 1)[0] 
                                     for ID_ in temp.ID]
        
        elif self.sample_method == "nearest":

            temp.loc[:, "Event"] = self.X.Event
            
            temp.loc[:, "sEvent"] = [index_nearest(Event_, self.X_.Event[self.X_.ID==ID_]) 
                                     for Event_, ID_ in list(zip(temp.Event, temp.ID))]
       
        
        # find d_, t_ from index position of sampled event
        d1 = self.X_.Distance[temp.sEvent]
        t1 = self.y_[temp.sEvent]

        # calculate Purdy points of sampled event
        purdy_points = pd.Series(a(v(d1)) * (t950(d1) / t1 - b(v(d1))))
              
        d2 = self.X.Distance 

        # predict using Purdy methodology
        self.y = a(v(d2)) * t950(d2) / (purdy_points.as_matrix() + 
                                      (a(v(d2)) * b(v(d2))).as_matrix())
        
        # convert back to log space if log is True
        if self.log is True:
                        
            self.y = np.log(self.y)

        return self.y
    

    def sampletable(self, p_range=range(1100, 900, -10), d=DISTANCES):
        
        """ A demonstration of the Purdy points tables for a given choice of
            distances and a range of point levels.
        """
    
        _t950 = t950(d)
                
        points = np.fromiter([a(v(d)) * _t950 / (p + a(v(d)) * b(v(d))) 
                              for p in p_range], dtype=float)
    
        table = pd.DataFrame(np.reshape(points, (points.size//10, 10)), 
                             index=p_range)
        
        return table 


