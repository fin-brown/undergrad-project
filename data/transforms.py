#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:51:58 2018

@author: finbrown1
"""

import numpy as np
import pandas as pd

from scipy.stats import rankdata

from utils import event_to_distance

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer


def conv_long_to_long_empties(data_):
    
    """ NOT USED ANYWHERE
        Adds missing values as np.NaNs to long format. 
    """
    
    data = data_.loc[:, ["ID", "Event", "Distance", "Time"]]
    
    distances = sorted(list(set(data_.Distance)))
    
    long = pd.DataFrame([[ID_, i, distances[i-1], np.NaN] 
                         for ID_ in range(1, max(data.ID) + 1) 
                         for i in range(1, 11)], 
                        columns=data.columns)
    
        
    long = long.set_index(pd.MultiIndex.from_tuples(list(zip(long.ID, long.Event)), 
                                                    names=["ID", "Time"]))
    
    
    recs = data.set_index(pd.MultiIndex.from_tuples(list(zip(data.ID, data.Event)), 
                                                    names=["ID", "Time"]))
     
    long.update(recs)
    
    return long.reset_index(drop=True)


def conv_long_to_wide(data_):
    
    """ Converts long data to wide. Used in ConvertLong2Wide. """    
    
    temp = dict()
    for index, row in data_.iterrows():
        
        ID, event, time = int(row.ID), int(row.Event), row.Time
        
        if ID not in temp.keys():
            times = [np.NaN]*10
            times[event-1] = time
            temp.update({ID: times})
        
        else:
            if np.isnan(temp[ID][event-1]) or temp[ID][event-1] < time: 
                temp[ID][event-1] = time
                
    wide = pd.DataFrame.from_dict(temp).transpose()
    wide.columns = range(1,11)
    
    return wide


def conv_wide_to_long(data_):
    
    """ Converts wide data to long. Used in ConvertWide2Long. """
        
    ID, Event, Time = [], [], []
    for ID_, row in data_.iterrows():
        
        recs = ~np.isnan(row.values)
        
        Event += row.index[recs].tolist()
        Time += row.values[recs].tolist()
        ID += [ID_] * sum(recs)
        
    long = pd.DataFrame({"ID": ID, 
                         "Event": Event,
                         "Distance": event_to_distance(Event),
                         "Time": Time})
    
    return long


class ConvertLong2Wide(BaseEstimator, TransformerMixin):
    
    """ Sklearn transformer converting long format data split by X, y into 
        wide format. 
    """
    
    def fit(self, X, y):
    
        self.X = X
        self.y = y
        
        return self
        
    def transform(self, X=None):
        
        long = self.X
        long.loc[:, "Time"] = self.y
        
        return conv_long_to_wide(long)


class ConvertWide2Long(BaseEstimator, TransformerMixin):
    
    """ Sklearn transformer converting wide format data into long format split 
        into X, y. 
    """
    
    def fit(self, X, y=None):
    
        self.X = X
        
        return self
        
    def transform(self, X=None):
    
        long = conv_wide_to_long(self.X)
        
        return long.loc[:, ["ID", "Event", "Distance"]], long.loc[:, "Time"]


class ImputeMeans(BaseEstimator, TransformerMixin):
    
    """ Sklearn mean imputer that returns data in pandas.DataFrame with 
        original column names and index for consistency.
    """
    
    def fit(self, X, y=None):
        
        self.X = X
        
    def transform(self, X=None):
        
        imputer = Imputer()
    
        new = pd.DataFrame(imputer.fit_transform(self.X), 
                           index=self.X.index, 
                           columns=self.X.columns)

        return new


class RemoveOutliers(BaseEstimator, TransformerMixin):
    
    """ Remove outliers based on outlier score defined as difference between
        max and min percentiles over all events for each athlete. 
        
        Also remove athletes scoring in bottom 1%.
    """
    
    def __init__(self, cutoff=.05, abs_cutoff=.01):
        
        self.cutoff = cutoff
        self.abs_cutoff = abs_cutoff

    def fit(self, X, y):
        
        self._wide = ConvertLong2Wide().fit(X, y).transform()
        
        (nrows, ncols) = self._wide.shape
        
        percentiles = pd.DataFrame(index=self._wide.index, 
                                   columns=self._wide.columns)
        
        for i in range(ncols):
            
            column = self._wide.iloc[:, i].dropna()
            
            to_update = pd.Series(1 - (rankdata(column, "average") / 
                                  len(column)), index=column.index)
            
            percentiles.iloc[:, i].update(to_update)
        
        self.bottom_1pc = (percentiles.min(axis=1) < self.abs_cutoff)
        self.outlier_score = percentiles.max(axis=1) - percentiles.min(axis=1)
        
        return self
        
    def transform(self, X=None):
        
        keep = self._wide[(self.outlier_score < 
                          self.outlier_score.quantile(1-self.cutoff)) 
                          & self.bottom_1pc.apply(np.invert)]
        
        return ConvertWide2Long().fit(keep).transform()



class TopXPercentile(BaseEstimator, TransformerMixin):
    
    """ Returns top X percentile of performers based on percentile score.
        Percentile score for a given runner is the max. percentile achieved
        in any event by that athlete.
    """
    
    def __init__(self, cutoff=.95):
        
        self.cutoff = cutoff

    def fit(self, X, y):
        
        self._wide = ConvertLong2Wide().fit(X, y).transform()
        
        (nrows, ncols) = self._wide.shape
        
        percentiles = pd.DataFrame(index=self._wide.index, 
                                   columns=self._wide.columns)
        
        for i in range(ncols):

            column = self._wide.iloc[:, i].dropna()
            
            to_update = pd.Series(rankdata(column, "average") / 
                                  len(column), index=column.index)
            
            percentiles.iloc[:, i].update(to_update)
        
        self.percentiles = percentiles.min(axis=1)
        
        return self
        
    def transform(self, X=None):
        
        keep = self._wide[self.percentiles < 
                          self.percentiles.quantile(self.cutoff)]
        
        return ConvertWide2Long().fit(keep).transform()
        
    



