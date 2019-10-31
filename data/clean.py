#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:30:06 2017

@author: finbrown1

issues to resolve:
    replace bad dates with NaNs
    remove outliers and bottom quartile
    
"""

import numpy as np

from data.load import data

# re-order events that were in wrong order; replace "[1900 1 1]"s with np.NaNs
for index, row in data.iterrows():
    
    if row.DoB == "[1900 1 1]":
        data.at[index, "DoB"] = np.NaN

    if row.Event == 6:
        data.at[index, "Distance"] = 1609.344
        
    elif row.Event == 7: 
        data.at[index, "Distance"] = 5000
    
    elif row.Event == 8:
        data.at[index, "Distance"] = 10000
    
# drop any entries faster than their respective official UKA records
male_records = (
        9.87,
        19.94,
        44.36,
        101.73,
        208.81,
        226.32,
        773.11,
        1606.57,
        3572,
        7633)

female_records = (
        10.99, 
        22.07,
        49.41,
        116.21,
        235.22,
        257.57,
        869.11,
        1801.09,
        4007,
        8125)

outliersu = [False]*len(data)
for index, row in data.iterrows():
    
    if row.Sex == "Male":
        if row.Time < male_records[row.Event-1]:
            
            outliersu[index] = True
            
    elif row.Time < female_records[row.Event-1]:
        
        outliersu[index] = True
        
non_outliers = np.invert(np.array(outliersu))

clean_data = data.loc[(data.Sex=="Male").values & non_outliers, 
                      ["ID", "Event", "Distance", "Time"]]














